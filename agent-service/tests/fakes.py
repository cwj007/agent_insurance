"""Agents 模块测试用的假模型、假嵌入与脚本化组件。

所有假组件只模拟被测试代码依赖的最小接口，不发起任何网络请求。
"""

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from app.agents.routing.analyzer import InputAnalyzer
from app.agents.routing.router import IntentRouter
from app.agents.routing.schemas import InputAnalysis, RouteResult
from app.agents.safety.schemas import GuardDecision


def run_async(coro):
    """同步执行协程，避免测试依赖 pytest-asyncio。

    Args:
        coro: 待执行的协程对象。

    Returns:
        协程的返回值。
    """
    return asyncio.run(coro)


class FakeStructuredRunnable:
    """结构化输出调用链的测试桩。

    Attributes:
        schema: 结构化输出的目标类型。
        _results: schema 到返回值（或返回值工厂）的映射。
    """

    def __init__(self, schema: type, results: dict[Any, Any]):
        """初始化测试桩。

        Args:
            schema: 结构化输出的目标类型。
            results: schema 到返回值或 value(messages) 工厂的映射。
        """
        self.schema = schema
        self._results = results

    async def ainvoke(self, messages, config=None, **kwargs):
        """返回预置的结构化结果。

        Args:
            messages: 传给模型的 messages，工厂函数会收到该参数。
            config: 运行时配置，测试桩忽略。
            **kwargs: 其他调用参数，测试桩忽略。

        Returns:
            预置的结构化结果。

        Raises:
            AssertionError: 未为该 schema 配置返回值时抛出。
        """
        value = self._results.get(self.schema)
        if value is None:
            try:
                return self.schema()
            except Exception as exc:
                raise AssertionError(
                    f"未为 {self.schema.__name__} 配置返回值"
                ) from exc

        if callable(value):
            return value(messages)
        return value


class FakeChatModel(BaseChatModel):
    """脚本化聊天模型。

    普通调用按顺序返回预置 AIMessage；结构化调用返回预置结果。
    继承 BaseChatModel 后可直接用于 create_agent 等需要模型对象的场景。

    Attributes:
        responses: 普通调用依次返回的消息；耗尽后重复最后一条。
        structured_results: schema 到结构化返回值的映射。
        generate_error: 设置后普通调用直接抛出该异常。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: list[AIMessage] = Field(default_factory=list)
    structured_results: dict[Any, Any] = Field(default_factory=dict)
    generate_error: Any = None

    _cursor: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        """模型类型标识。

        Returns:
            固定返回 fake-chat-model。
        """
        return "fake-chat-model"

    def _next_message(self) -> AIMessage:
        """取下一条预置消息。

        Returns:
            预置 AIMessage；没有预置时返回固定测试回答。
        """
        if not self.responses:
            return AIMessage(content="测试回答")
        index = min(self._cursor, len(self.responses) - 1)
        self._cursor += 1
        return self.responses[index]

    def _generate(
            self,
            messages,
            stop=None,
            run_manager=None,
            **kwargs,
    ) -> ChatResult:
        """生成一条消息。

        Args:
            messages: 输入消息。
            stop: 停止词，测试桩忽略。
            run_manager: 回调管理器，测试桩忽略。
            **kwargs: 其他参数，测试桩忽略。

        Returns:
            包含一条预置消息的 ChatResult。

        Raises:
            Exception: generate_error 已设置时原样抛出。
        """
        if self.generate_error is not None:
            raise self.generate_error
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])

    def bind_tools(self, tools, **kwargs):
        """绑定工具，测试桩直接返回自身。

        Args:
            tools: 工具列表。
            **kwargs: 其他绑定参数。

        Returns:
            模型自身。
        """
        return self

    def with_structured_output(self, schema, **kwargs):
        """返回结构化输出测试桩。

        Args:
            schema: 结构化输出的目标类型。
            **kwargs: 其他参数，测试桩忽略。

        Returns:
            读取 structured_results 的调用链。
        """
        return FakeStructuredRunnable(schema, self.structured_results)


class FakeEmbeddings(Embeddings):
    """按映射函数生成向量的嵌入模型。

    Attributes:
        _mapper: 文本到向量的映射函数。
    """

    def __init__(self, mapper: Callable[[str], list[float]]):
        """初始化嵌入模型。

        Args:
            mapper: 文本到向量的映射函数。
        """
        self._mapper = mapper

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量。

        Args:
            texts: 文档文本列表。

        Returns:
            与输入一一对应的向量列表。
        """
        return [self._mapper(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量。

        Args:
            text: 查询文本。

        Returns:
            查询向量。
        """
        return self._mapper(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """异步批量生成文档向量。

        Args:
            texts: 文档文本列表。

        Returns:
            与输入一一对应的向量列表。
        """
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """异步生成查询向量。

        Args:
            text: 查询文本。

        Returns:
            查询向量。
        """
        return self.embed_query(text)


def keyword_embedding(text: str) -> list[float]:
    """按保险关键词生成 4 维向量，供 L1 语义测试使用。

    Args:
        text: 待编码文本。

    Returns:
        4 维向量，各维依次表示推荐、理赔、闲聊、转人工的关键词命中数。
    """
    buckets = (
        (
            "recommend",
            (
                "推荐", "产品", "保额", "保费", "重疾", "医疗险", "意外险",
                "寿险", "投保", "方案", "保险", "买",
            ),
        ),
        (
            "claim",
            (
                "理赔", "报销", "赔付", "出险", "住院", "门诊", "报案",
                "拒赔", "材料", "审核",
            ),
        ),
        (
            "chitchat",
            (
                "你好", "您好", "谢谢", "再见", "天气", "聊天", "笑话",
                "名字", "无聊", "电影",
            ),
        ),
        (
            "handoff",
            ("人工", "客服", "投诉", "经理", "主管", "领导", "热线"),
        ),
    )
    scores = [sum(1 for keyword in keywords if keyword in text) for _, keywords in buckets]
    if max(scores) == 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [float(score) for score in scores]


class ScriptedIntentRouter(IntentRouter):
    """可控的意图路由器。

    L0 默认走真实规则；L2 与关键词兜底可按用例脚本化。

    Attributes:
        _l2_result: L2 分类的预置结果。
        _l2_error: 设置后 L2 调用直接抛出该异常。
        _rule_result: 规则兜底的预置结果。
        _fast_override: 设置后覆盖 classify_fast 的返回值。
    """

    def __init__(
            self,
            l2_result: RouteResult | None = None,
            l2_error: Exception | None = None,
            rule_result: RouteResult | None = None,
            fast_override=None,
    ):
        """初始化脚本化路由器。

        Args:
            l2_result: L2 分类的预置结果。
            l2_error: L2 调用需要抛出的异常。
            rule_result: 规则兜底的预置结果。
            fast_override: classify_fast 的固定返回值。
        """
        super().__init__(model=FakeChatModel(), enable_l1=False)
        self._l2_result = l2_result
        self._l2_error = l2_error
        self._rule_result = rule_result
        self._fast_override = fast_override

    async def classify_fast(self, query, active_workflow=None):
        """执行 L0 规则和低信息量判断，或返回预设结果。

        Args:
            query: 用户本轮输入。
            active_workflow: 上一轮主流程意图。

        Returns:
            (分类结果, 路由来源)；未命中时返回 None。
        """
        if self._fast_override is not None:
            return self._fast_override
        return await super().classify_fast(query, active_workflow)

    async def classify_l2(self, query, context=""):
        """返回预设的 L2 结果或抛出预设异常。

        Args:
            query: 用户本轮输入。
            context: 历史上下文。

        Returns:
            预设的分类结果。

        Raises:
            Exception: 配置了 l2_error 时抛出。
            RuntimeError: 未配置结果时抛出。
        """
        if self._l2_error is not None:
            raise self._l2_error
        if self._l2_result is None:
            raise RuntimeError("测试未配置 L2 结果")
        return self._l2_result

    def infer_intent_by_rules(self, query, active_workflow=None):
        """返回预设结果，否则使用真实的规则兜底。

        Args:
            query: 用户本轮输入。
            active_workflow: 上一轮主流程意图。

        Returns:
            规则推断结果；无法判断时返回 None。
        """
        if self._rule_result is not None:
            return self._rule_result
        return super().infer_intent_by_rules(query, active_workflow)


class ScriptedInputAnalyzer(InputAnalyzer):
    """返回预设结果的合并输入分析器。

    Attributes:
        _guard: 预设的安全校验结果。
        _route: 预设的意图分类结果。
        _error: 设置后 analyze 直接抛出该异常。
    """

    def __init__(
            self,
            guard: GuardDecision,
            route: RouteResult | None = None,
            error: Exception | None = None,
    ):
        """初始化脚本化分析器。

        Args:
            guard: 预设的安全校验结果。
            route: 预设的意图分类结果。
            error: analyze 需要抛出的异常。
        """
        # 不调用父类构造，避免创建真实模型调用链
        self._guard = guard
        self._route = route
        self._error = error

    async def analyze(self, query: str, context: str = "") -> InputAnalysis:
        """返回预设的合并分析结果。

        Args:
            query: 用户本轮输入。
            context: 历史上下文。

        Returns:
            预设的 InputAnalysis。

        Raises:
            Exception: 配置了 error 时抛出。
        """
        if self._error is not None:
            raise self._error
        return InputAnalysis(guard=self._guard, route=self._route)


class ScriptedChain:
    """按顺序返回结果或抛出异常的调用链。

    Attributes:
        outcomes: 每次调用对应的返回值或异常。
        calls: 已调用次数。
    """

    def __init__(self, outcomes: list[Any]):
        """初始化调用链。

        Args:
            outcomes: 每次调用对应的返回值或异常。
        """
        self.outcomes = list(outcomes)
        self.calls = 0

    async def ainvoke(self, messages):
        """执行一次预设结果。

        Args:
            messages: 输入消息，测试桩忽略。

        Returns:
            本次调用对应的返回值。

        Raises:
            Exception: 本次结果是异常时抛出。
        """
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def allowed_guard(reason: str = "正常咨询") -> GuardDecision:
    """构造放行的安全校验结果。

    Args:
        reason: 判断理由。

    Returns:
        放行的 GuardDecision。
    """
    return GuardDecision(allowed=True, category="none", reason=reason)


def route_result(
        intent: str,
        confidence: float = 0.9,
        *,
        needs_clarification: bool = False,
        clarification_question: str | None = None,
) -> RouteResult:
    """构造意图分类结果。

    Args:
        intent: 意图标签。
        confidence: 置信度。
        needs_clarification: 是否需要澄清。
        clarification_question: 澄清问题。

    Returns:
        RouteResult 实例。
    """
    return RouteResult(
        intent=intent,
        confidence=confidence,
        reason="测试构造",
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )
