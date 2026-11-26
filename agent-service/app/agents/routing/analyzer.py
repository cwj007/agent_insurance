"""输入合并分析：一次模型调用同时完成安全校验和意图分类。"""

from langchain_core.language_models import BaseChatModel

from app.agents.models.resilience import CircuitBreaker, ainvoke_with_fallback
from app.agents.routing.router import SYSTEM_PROMPT as INTENT_SYSTEM_PROMPT
from app.agents.routing.schemas import InputAnalysis
from app.agents.safety.guardrails import (
    INPUT_GUARD_FAIL_OPEN,
    INPUT_GUARD_PROMPT,
    check_input_rules,
    wrap_untrusted_content,
)
from app.agents.safety.schemas import GuardDecision
from app.core.logging import get_logger

logger = get_logger(__name__)

# 合并分析主模型失败后直接切换兜底大模型，避免重试放大首响延迟。
ANALYSIS_RETRY_ATTEMPTS = 1

_OUTPUT_EXAMPLES = """
示例（放行）：
{"guard": {"allowed": true, "category": "none", "reason": "正常的理赔咨询"}, "route": {"intent": "claim", "confidence": 0.92, "reason": "咨询理赔流程", "slots": {}, "missing_slots": [], "needs_clarification": false, "clarification_question": null}}

示例（拦截）：
{"guard": {"allowed": false, "category": "illegal", "reason": "教唆骗保"}, "route": null}
""".strip()

SYSTEM_PROMPT = f"""你是保险客服的输入分析节点，需要一次性完成两件事：
1. 安全合规校验：判断用户输入是否可以进入后续业务流程。
2. 意图分类：在允许放行时，判断用户消息应该交给哪个工作流。

===== 安全合规校验规则 =====
{INPUT_GUARD_PROMPT}

===== 意图分类规则 =====
{INTENT_SYSTEM_PROMPT}

===== 合并输出要求 =====
- guard：安全校验结果，必填。
- route：意图分类结果，必填字段。
- 当 guard.allowed 为 true 时，route 必须完整填写 intent、confidence、reason 等字段。
- 当 guard.allowed 为 false 时，route 必须为 null。
- 不要回答用户问题，只输出结构化结果。

{_OUTPUT_EXAMPLES}""".strip()


class InputAnalyzer:
    """输入安全校验与意图分类的合并分析器。

    先走规则检查，未命中再调用一次模型同时返回 GuardDecision 和 RouteResult。
    主模型失败时切换兜底模型重试一次，仍失败则按输入侧降级策略返回。

    Attributes:
        _chain: 绑定 InputAnalysis 结构化输出的主模型调用链。
        _fallback_chain: 主模型失败时使用的兜底调用链。
        _breaker: 主模型连续失败时的熔断器。
    """

    def __init__(
            self,
            model: BaseChatModel,
            fallback_model: BaseChatModel | None = None,
    ):
        """初始化合并分析器。

        Args:
            model: 主分析模型，建议使用低温度的小模型。
            fallback_model: 可选的大模型，主模型失败时重试一次。
        """
        self._chain = model.with_structured_output(InputAnalysis)
        self._fallback_chain = (
            fallback_model.with_structured_output(InputAnalysis)
            if fallback_model is not None
            else None
        )
        self._breaker = CircuitBreaker()

    async def analyze(self, query: str, context: str = "") -> InputAnalysis:
        """同时完成输入安全校验和意图分类。

        Args:
            query: 用户本轮输入。
            context: 历史上下文，包含上一轮主流程意图和历史消息。

        Returns:
            InputAnalysis：包含 guard 和可选的 route。
        """

        # 1. 规则优先：明显违规的输入不消耗模型调用
        rule_decision = check_input_rules(query)
        if rule_decision is not None:
            return InputAnalysis(guard=rule_decision, route=None)

        # 2. 合并调用：安全校验 + 意图分类
        content = f"用户问题：{query}\n历史上下文：{context or '无'}"
        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", wrap_untrusted_content(content)),
        ]

        try:
            return await ainvoke_with_fallback(
                self._chain,
                messages,
                self._fallback_chain,
                breaker=self._breaker,
                attempts=ANALYSIS_RETRY_ATTEMPTS,
                validate=lambda result: not (
                        result.guard.allowed and result.route is None
                ),
                label="合并输入分析",
            )
        except Exception as exc:
            logger.error(f"合并输入分析失败: {exc}")

        # 3. 两个模型都失败：按输入侧降级策略返回，route 留空交给 route 节点
        if INPUT_GUARD_FAIL_OPEN:
            logger.warning("合并输入分析失败，按策略降级放行")
            guard = GuardDecision(
                allowed=True,
                category="none",
                reason="合并分析异常，降级放行",
            )
        else:
            guard = GuardDecision(
                allowed=False,
                category="other",
                reason="合并分析异常，按不合规处理",
            )

        return InputAnalysis(guard=guard, route=None)


# 模块级单例，和 RAG、guardrails 的初始化方式保持一致
_input_analyzer: InputAnalyzer | None = None


def init_input_analyzer(analyzer: InputAnalyzer) -> InputAnalyzer:
    """初始化全局输入分析器。

    Args:
        analyzer: 输入分析器实例。

    Returns:
        初始化后的 InputAnalyzer 实例。
    """
    global _input_analyzer
    _input_analyzer = analyzer
    logger.info("输入合并分析器初始化完成~☑️")
    return _input_analyzer


def get_input_analyzer() -> InputAnalyzer:
    """获取全局输入分析器。

    Returns:
        已初始化的 InputAnalyzer 实例。

    Raises:
        RuntimeError: 尚未调用 init_input_analyzer 初始化时抛出。
    """
    if _input_analyzer is None:
        raise RuntimeError("输入分析器尚未初始化，请先调用 init_input_analyzer()")
    return _input_analyzer
