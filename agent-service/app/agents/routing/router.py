"""意图识别与路由"""

import asyncio
import re

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.models.resilience import (
    CircuitBreaker,
    ainvoke_with_fallback,
)
from app.agents.routing.schemas import Intent, RouteLayer, RouteResult
from app.agents.routing.semantic import SemanticIntentIndex
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 本地意图分类模型，复用 RAG 使用的 Ollama 服务
OLLAMA_BASE_URL = settings.llm.ollama_base_url
INTENT_MODEL_NAME = settings.llm.intent_model_name

# 本地小模型默认关闭 thinking：qwen3.5:0.8b 开启 thinking 时结构化输出
# 经常只返回空 content，导致 JSON 解析失败（评测失败率 75%）。
LOCAL_MODEL_REASONING = False

# L2 主模型失败后直接切换兜底大模型，避免本地模型重试放大首响延迟。
L2_RETRY_ATTEMPTS = 1

# 各意图的最低置信度阈值；文档 §5.1 的起步值，上线前用标注集校准
INTENT_CONFIDENCE_THRESHOLDS: dict[Intent, float] = {
    "chitchat": 0.60,
    "recommendation_plan": 0.65,
    "claim": 0.75,
    "human_handoff": 0.85,
    "fallback": 0.00,
}

# L0 规则：显式转人工命令，不依赖模型判断
_HANDOFF_PATTERN = re.compile(r"(转人工|人工客服|找人工|真人客服)")

# 短追问特征：只有带这些追问信号时，才继承上一轮主流程
_FOLLOWUP_PATTERN = re.compile(
    r"(多少钱|多久|怎么|如何|为什么|为啥|然后|还有|这个|那个|"
    r"可以吗|行不行|具体|呢|吗|？|\?)"
)

# 模型全部失败时的关键词兜底；顺序即优先级
_INTENT_KEYWORDS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        "human_handoff",
        ("转人工", "人工客服", "找人工", "投诉", "经理", "主管", "客服"),
    ),
    (
        "claim",
        ("理赔", "报销", "赔付", "出险", "住院", "门诊", "报案", "拒赔", "材料"),
    ),
    (
        "recommendation_plan",
        (
            "推荐", "产品", "保额", "保费", "预算", "重疾", "医疗险",
            "意外险", "寿险", "投保", "方案",
        ),
    ),
    (
        "chitchat",
        ("你好", "您好", "谢谢", "再见", "天气", "聊天", "笑话", "名字"),
    ),
)

# 低信息量输入的最大长度；只兜住极短、没有业务对象的模糊表达
LOW_INFO_MAX_CHARS = 12

# 模糊诉求特征：这些表达缺少具体业务对象，模型容易强行归类
_VAGUE_REQUEST_PATTERN = re.compile(
    r"(怎么办|怎么弄|怎么处理|怎么解决|什么情况|有什么办法|有没有办法|"
    r"要准备什么|需要什么|要多少钱|要花多少钱|大概要多久|"
    r"靠不靠谱|值不值|能不能行|行不行|可以帮我|帮我弄|帮我处理|"
    r"这样对吗|是怎么回事|不太明白|先看看吧|我还有问题|"
    r"这个能办吗|这个有用吗|这个需要什么东西|哪里能办)"
)

# 明显闲聊信号：带这些词的短句交给模型判断，不按低信息量兜底
_CHITCHAT_HINTS: tuple[str, ...] = (
    "嗨", "哈喽", "早上好", "下午好", "晚上好", "晚安", "早安",
    "嘿嘿", "哈哈", "拜拜", "节日快乐", "下次再聊", "天气", "下雨",
    "电影", "音乐", "唱歌", "猫", "狗", "新闻", "旅游", "爬山",
)

# 低信息量输入的澄清话术，由规则层直接返回
LOW_INFORMATION_QUESTION = (
    "请问您具体想咨询什么？例如保险产品推荐、理赔流程，"
    "或回复「转人工」联系人工客服。"
)

# 通用业务词：规则关键词没覆盖，但足以说明用户有明确保险诉求
_GENERIC_BUSINESS_KEYWORDS: frozenset[str] = frozenset({
    "保险", "保单", "条款", "退保", "险种", "保障", "产品",
})

# 所有规则关键词与通用业务词的并集，用于判断输入是否带有明确方向
_RULE_KEYWORDS: frozenset[str] = frozenset(
    keyword
    for _, keywords in _INTENT_KEYWORDS
    for keyword in keywords
) | _GENERIC_BUSINESS_KEYWORDS


def _is_low_information(text: str) -> bool:
    """判断输入是否信息量不足、无法判断意图。

    只识别“怎么办”“要准备什么”这类没有业务对象的模糊表达；
    带业务词或明显闲聊信号的短句交给模型，避免误伤正常短问句。

    Args:
        text: 用户本轮输入。

    Returns:
        极短、不含业务/闲聊关键词且带有模糊诉求特征时返回 True。
    """
    stripped = text.strip()
    if not stripped or len(stripped) > LOW_INFO_MAX_CHARS:
        return False
    if any(keyword in stripped for keyword in _RULE_KEYWORDS):
        return False
    if any(hint in stripped for hint in _CHITCHAT_HINTS):
        return False
    return _VAGUE_REQUEST_PATTERN.search(stripped) is not None


SYSTEM_PROMPT = """
你是保险商城智能客服的意图识别节点，请判断用户消息应该交给哪个工作流。

intent只能从以下选项中选择：
- recommendation_plan：保险产品咨询、产品条款、保险推荐、方案管理和投保咨询
- claim：报案、理赔责任、理赔材料、理赔流程和理赔进度
- human_handoff：投诉、高风险问题或用户明确要求人工服务
- chitchat：与保险业务无关的普通聊天
- fallback：用户表达笼统、缺少具体业务对象或诉求不明确（如“这个怎么办”“要准备什么”“靠不靠谱”），或无法确定属于以上任何一类

分类规则：
1. 结合历史上下文判断；用户是短追问（如“多少钱”“多久”）且上一轮有主流程时，继承上一轮的主流程意图；没有历史上下文时，短追问应输出 fallback。
2. 一句话包含多个意图时，选择当前最需要处理的主意图。
3. 无法判断时输出 fallback，不要猜测；宁可让用户补充一句，也不要强行归入某个业务意图。
4. confidence 要反映真实把握程度：0.9 以上为表述明确，0.6~0.8 为存在歧义，低于 0.5 为无法判断。
5. 只有缺少关键信息、无法继续处理时才把 needs_clarification 设为 true，并给出具体的 clarification_question；不要为了追问而追问。
6. 只抽取用户明确提到的槽位，不要编造。

示例：
“住院了怎么理赔” -> claim
“推荐一款医疗险” -> recommendation_plan
“我想投诉业务员” -> human_handoff
“今天天气不错” -> chitchat
“这个怎么办” -> fallback
“要准备什么” -> fallback

槽位参考：
- recommendation_plan：category（medical/critical_illness/life/accident）、age、budget
- claim：category、incident_type
""".strip()


class IntentRouter:
    """意图路由器。

    按 L0 规则 → 低信息量规则 → L1 语义 → L2 小模型 的顺序识别意图：
    - L0 命中直接返回，不调模型
    - 低信息量且无上下文可继承时直接进入澄清
    - L1 相似度达标直接返回，不调模型
    - L2 主模型失败时，切换兜底模型

    只输出 RouteResult，不关心具体跳转到哪个工作流节点。

    Attributes:
        model: 主分类模型，供合并输入分析复用。
        route_model: 绑定 RouteResult 结构化输出的主模型调用链。
        fallback_route_model: 主模型失败时使用的兜底调用链。
        _route_breaker: 主模型连续失败时的熔断器。
        _semantic_index: L1 语义索引。
    """

    def __init__(
            self,
            model: BaseChatModel | None = None,
            fallback_model: BaseChatModel | None = None,
            enable_l1: bool = True,
            embeddings: Embeddings | None = None,
    ):
        """初始化意图路由器。

        Args:
            model: 主分类模型，便于测试时注入；为空时使用本地 Ollama 模型。
            fallback_model: 可选的大模型，主模型分类失败时重试一次。
            enable_l1: 是否启用 L1 语义路由。
            embeddings: 可选的嵌入模型，便于测试或替换向量模型。
        """

        # 1. 初始化模型：默认使用本地 ollama qwen3.5:0.8b 模型
        if model is None:
            model = init_chat_model(
                model=INTENT_MODEL_NAME,
                model_provider="ollama",
                base_url=OLLAMA_BASE_URL,
                temperature=0,
                reasoning=LOCAL_MODEL_REASONING,
            )

        # 2. 配置模型输出结构化数据
        self.model = model
        self.route_model = model.with_structured_output(RouteResult)
        self.fallback_route_model = (
            fallback_model.with_structured_output(RouteResult)
            if fallback_model is not None
            else None
        )
        self._route_breaker = CircuitBreaker()

        # 3. 准备 L1 语义索引，实际构建在 initialize() 里完成
        self._enable_l1 = enable_l1
        self._semantic_index = (
            SemanticIntentIndex(embeddings=embeddings)
            if enable_l1
            else None
        )
        self._semantic_ready = not enable_l1
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """构建 L1 语义索引；构建失败时自动降级为只走 L2。"""
        async with self._init_lock:
            if self._semantic_ready:
                return

            if self._semantic_index is None:
                self._semantic_ready = True
                return

            try:
                await self._semantic_index.build()
            except Exception as exc:
                logger.warning(f"L1 语义索引构建失败，降级为仅 L2: {exc}")
                self._enable_l1 = False
                self._semantic_index = None
            finally:
                self._semantic_ready = True

    async def _ensure_l1_ready(self) -> None:
        """确保 L1 已初始化；未初始化时惰性构建一次。"""
        if self._enable_l1 and not self._semantic_ready:
            await self.initialize()

    async def classify_fast(
            self,
            query: str,
            active_workflow: Intent | None = None,
    ) -> tuple[RouteResult, RouteLayer] | None:
        """执行 L0 规则、低信息量判断和 L1 语义匹配。

        Args:
            query: 用户本轮输入。
            active_workflow: 上一轮主流程意图，用于判断短追问能否继承。

        Returns:
            (分类结果, 路由来源)；规则和 L1 都未命中时返回 None。
        """

        # 1. L0 规则优先：显式转人工直接命中，不再调用模型
        if _HANDOFF_PATTERN.search(query):
            return (
                RouteResult(
                    intent="human_handoff",
                    confidence=1.0,
                    reason="命中转人工规则",
                ),
                "rule",
            )

        # 2. 低信息量：无上下文可继承时直接澄清；可继承的短追问交给 L2
        if _is_low_information(query):
            if (
                    active_workflow is not None
                    and _FOLLOWUP_PATTERN.search(query)
            ):
                return None
            return (
                RouteResult(
                    intent="fallback",
                    confidence=0.3,
                    reason="输入信息量不足，无法判断意图",
                    needs_clarification=True,
                    clarification_question=LOW_INFORMATION_QUESTION,
                ),
                "rule",
            )

        # 3. L1 语义匹配：相似度和领先优势都达标才命中
        if not self._enable_l1:
            return None

        await self._ensure_l1_ready()
        if self._semantic_index is None:
            return None

        try:
            hit = await self._semantic_index.search(query)
        except Exception as exc:
            logger.warning(f"L1 语义匹配失败，降级到 L2: {exc}")
            return None

        if hit is None:
            return None

        intent, score = hit
        return (
            RouteResult(
                intent=intent,
                confidence=score,
                reason=f"L1 语义相似度命中（{score:.3f}）",
            ),
            "semantic",
        )

    async def classify_l2(self, query: str, context: str = "") -> RouteResult:
        """调用模型完成 L2 意图分类，失败时切换兜底模型。

        Args:
            query: 用户本轮输入。
            context: 历史上下文。

        Returns:
            模型输出的 RouteResult。

        Raises:
            Exception: 主模型和兜底模型都失败时抛出。
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：{query}\n历史上下文：{context or '无'}"),
        ]

        return await ainvoke_with_fallback(
            self.route_model,
            messages,
            self.fallback_route_model,
            breaker=self._route_breaker,
            attempts=L2_RETRY_ATTEMPTS,
            label="意图分类",
        )

    async def route(
            self,
            query: str,
            context: str = "",
            active_workflow: Intent | None = None,
    ) -> tuple[RouteResult, RouteLayer]:
        """按 L0 → 低信息量 → L1 → L2 的顺序识别意图。

        Args:
            query: 用户输入的查询。
            context: 历史上下文，包含上一轮主流程意图和历史消息。
            active_workflow: 上一轮主流程意图，用于短追问继承。

        Returns:
            意图分类结果和本次路由来源的二元组。
        """

        # 1. 便宜的规则/语义层先跑
        fast_result = await self.classify_fast(query, active_workflow)
        if fast_result is not None:
            return fast_result

        # 2. 未命中再调用模型
        result = await self.classify_l2(query, context)
        return result, "llm"

    def infer_intent_by_rules(
            self,
            query: str,
            active_workflow: Intent | None = None,
    ) -> RouteResult | None:
        """模型全部失败时，用关键词和上下文推断意图。

        Args:
            query: 用户本轮输入。
            active_workflow: 上一轮主流程意图。

        Returns:
            规则推断出的意图结果；无法判断时返回 None。
        """
        text = query.strip()
        if not text:
            return None

        # 1. 关键词兜底，按高风险到低风险排序
        for intent, keywords in _INTENT_KEYWORDS:
            for keyword in keywords:
                if keyword in text:
                    return RouteResult(
                        intent=intent,
                        confidence=0.50,
                        reason=f"模型不可用，关键词命中「{keyword}」",
                    )

        # 2. 只有带明显追问信号的短句，才继承上一轮主流程
        if (
                active_workflow is not None
                and len(text) <= 8
                and _FOLLOWUP_PATTERN.search(text)
        ):
            return RouteResult(
                intent=active_workflow,
                confidence=0.40,
                reason="模型不可用，继承上一轮主流程",
            )

        return None

    def should_clarify(
            self,
            result: RouteResult,
            route_layer: RouteLayer = "llm",
    ) -> bool:
        """判断分类结果是否需要先向用户追问。

        L0/L1 各自有独立阈值，命中即视为可信，只尊重模型给出的追问建议；
        L2 才使用各意图的置信度阈值。

        Args:
            result: 意图分类结果。
            route_layer: 本次路由来源。

        Returns:
            需要追问时返回 True。
        """
        if route_layer in ("rule", "semantic"):
            return result.needs_clarification

        threshold = INTENT_CONFIDENCE_THRESHOLDS[result.intent]

        return (
            result.needs_clarification
            or result.intent == "fallback"
            or result.confidence < threshold
        )


# 模块级单例，和 RAG、guardrails 的初始化方式保持一致
_intent_router: IntentRouter | None = None


async def init_intent_router(router: IntentRouter | None = None) -> IntentRouter:
    """初始化全局意图路由器。

    Args:
        router: 可选的意图路由器实例；为空时使用默认本地模型创建。

    Returns:
        初始化后的 IntentRouter 实例。
    """
    global _intent_router
    _intent_router = router or IntentRouter()
    await _intent_router.initialize()
    logger.info("意图路由器初始化完成~☑️")
    return _intent_router


def get_intent_router() -> IntentRouter:
    """获取全局意图路由器。

    Returns:
        已初始化的 IntentRouter 实例。

    Raises:
        RuntimeError: 尚未调用 init_intent_router 初始化时抛出。
    """
    if _intent_router is None:
        raise RuntimeError("意图路由器尚未初始化，请先调用 init_intent_router()")
    return _intent_router
