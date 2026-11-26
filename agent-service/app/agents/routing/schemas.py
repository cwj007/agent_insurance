"""意图识别与输入分析的数据结构。"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.agents.safety.schemas import GuardDecision

# 一级意图：与工作流、工具白名单一一对应；先有工作流，再加意图
Intent = Literal[
    "chitchat",  # 闲聊
    "recommendation_plan",  # 推荐方案
    "claim",  # 理赔服务
    "human_handoff",  # 转人工
    "fallback",  # 无法判断，进入兜底或澄清
]

# 路由来源：规则 → 语义 → LLM → 兜底
RouteLayer = Literal["rule", "semantic", "llm", "fallback"]


class RouteResult(BaseModel):
    """一轮意图识别结果。

    分类器只输出意图标签和判断依据，不输出目标节点；节点映射由路由函数完成。
    置信度阈值和澄清策略属于路由层，不写进本模型。

    Attributes:
        intent: 本轮消息应进入的工作流。
        confidence: 模型对 intent 的置信度，0~1。
        reason: 一句话判断理由，仅用于排查，不返回给用户。
        slots: 本轮抽取到的槽位。
        missing_slots: 当前意图下仍然缺失的必填槽位。
        needs_clarification: 是否建议先向用户追问。
        clarification_question: 建议的追问话术。
    """

    intent: Intent = Field(
        description="本轮消息应进入的工作流；无法判断时输出 fallback",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "对 intent 的置信度，0~1。"
            "0.9 以上表示用户表述明确；0.6~0.8 表示存在歧义；"
            "低于 0.5 表示无法判断，此时 intent 应为 fallback"
        ),
    )
    reason: str = Field(
        max_length=100,
        description="一句话判断理由，仅用于排查，不返回给用户",
    )
    slots: dict[str, Any] = Field(
        default_factory=dict,
        description="本轮抽取到的槽位，例如 {'category': 'medical', 'age': 30}",
    )
    missing_slots: list[str] = Field(
        default_factory=list,
        description="当前意图下仍然缺失的必填槽位名称",
    )
    needs_clarification: bool = Field(
        default=False,
        description="信息不足或有歧义、建议先追问用户时为 true",
    )
    clarification_question: str | None = Field(
        default=None,
        description="needs_clarification 为 true 时给出的具体追问，否则为 null",
    )

    @model_validator(mode="after")
    def _check_clarification(self) -> "RouteResult":
        """校验澄清字段是否自洽。

        Returns:
            校验通过的实例自身。

        Raises:
            ValueError: needs_clarification 与 clarification_question 不一致时抛出。
        """
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("needs_clarification 为 true 时必须提供 clarification_question")
        if not self.needs_clarification and self.clarification_question:
            raise ValueError("clarification_question 仅在 needs_clarification 为 true 时提供")
        return self

    def to_state_update(self) -> dict[str, Any]:
        """转换为可直接合并进 Graph state 的更新字段。

        Returns:
            dict[str, Any]: 与 InsuranceAgentState 字段同名的更新字典。
        """
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "route_reason": self.reason,
            "slots": self.slots,
            "missing_slots": self.missing_slots,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
        }


class InputAnalysis(BaseModel):
    """输入安全校验与意图分类的合并结果。

    一次模型调用同时完成安全判断和意图识别，减少一次往返。
    guard 拦截时 route 必须为空。

    Attributes:
        guard: 安全合规校验结果。
        route: 意图分类结果；guard 拦截时为 None。
    """

    guard: GuardDecision = Field(description="安全合规校验结果")
    route: RouteResult | None = Field(
        default=None,
        description="意图分类结果；guard.allowed 为 false 时必须为 null",
    )

    @model_validator(mode="after")
    def _check_blocked_route(self) -> "InputAnalysis":
        """校验拦截时不再返回意图分类结果。

        Returns:
            校验通过的实例自身。

        Raises:
            ValueError: guard 拦截但 route 不为空时抛出。
        """
        if not self.guard.allowed and self.route is not None:
            raise ValueError("guard 拦截时 route 必须为 null")
        return self
