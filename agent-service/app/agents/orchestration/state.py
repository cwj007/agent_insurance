"""保险顾问主 Graph 状态。"""

from typing import Any, NotRequired

from langgraph.graph import MessagesState

from app.agents.routing.schemas import Intent, RouteLayer
from app.agents.safety.schemas import GuardCategory, GuardStage


class InsuranceAgentState(MessagesState):
    """保险顾问主 Graph 状态。

    Attributes:
        previous_workflow: 上一轮的主流程意图，用于短追问继承。
        active_workflow: 当前多轮对话的主流程意图。
        intent: 最近一轮分类得到的一级意图。
        confidence: 最近一轮意图识别的置信度。
        route_reason: 最近一轮路由理由，便于排查。
        slots: 已收集的槽位。
        missing_slots: 当前意图下仍然缺失的必填槽位。
        needs_clarification: 最近一轮是否需要澄清。
        clarification_question: 建议的澄清问题。
        route_layer: 最近一轮路由来源。
        route_similarity: L1 语义路由的最高相似度。
        precomputed_route: 输入分析阶段预计算的意图结果，供 route 节点复用。
        precomputed_layer: 预计算意图结果来自哪一层（rule / semantic / llm）。
        clarification_count: 连续澄清次数，超过上限后强制兜底或转人工。
        guard_stage: 最近一次安全校验发生在输入侧还是输出侧。
        guard_blocked: 最近一次安全校验是否拦截。
        guard_category: 最近一次校验命中的风险类别。
        guard_reason: 最近一次校验的判断理由，便于排查。
        handoff_requested: 是否已记录用户的人工转接请求。
        summary: 较早会话历史的摘要，由上下文治理节点生成。
        workflow_trace: 上一轮工具执行轨迹的精简摘要，只保留元数据。
        workflow_failed: 当前轮工作流是否执行失败。
        workflow_error: 当前轮工作流失败说明，仅用于排查和事件。
        final_answer: 输出安全校验后的最终回答，供 runner 流式推送。
    """

    previous_workflow: NotRequired[Intent]
    active_workflow: NotRequired[Intent]

    # 当前轮分类结果：使用基础类型扁平存储，
    # 避免 Pydantic 对象在严格模式下从 checkpointer 读出时退化成 dict
    intent: NotRequired[Intent]
    confidence: NotRequired[float]
    route_reason: NotRequired[str]
    slots: NotRequired[dict[str, Any]]
    missing_slots: NotRequired[list[str]]
    needs_clarification: NotRequired[bool]
    clarification_question: NotRequired[str | None]

    # 路由与兜底
    route_layer: NotRequired[RouteLayer]
    route_similarity: NotRequired[float | None]
    precomputed_route: NotRequired[dict[str, Any] | None]
    precomputed_layer: NotRequired[RouteLayer | None]
    clarification_count: NotRequired[int]

    # 安全合规校验结果
    guard_stage: NotRequired[GuardStage]
    guard_blocked: NotRequired[bool]
    guard_category: NotRequired[GuardCategory]
    guard_reason: NotRequired[str]

    # 人工转接
    handoff_requested: NotRequired[bool]

    # 上下文治理
    summary: NotRequired[str]

    # 工作流执行轨迹与失败标记
    workflow_trace: NotRequired[list[dict[str, Any]]]
    workflow_failed: NotRequired[bool]
    workflow_error: NotRequired[str | None]

    # 输出安全校验后的最终回答
    final_answer: NotRequired[str]
