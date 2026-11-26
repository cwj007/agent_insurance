"""保险顾问主 Graph 编排。

整体流程：

    START → input_guard ──拦截──→ guard_reply → END
                │
                └──放行──→ route ──澄清──→ clarify_node → output_guard → END
                              │
                              └──意图──→ 工作流节点 → output_guard → END
"""

from collections.abc import Callable
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.agents.context.summarizer import init_context_manager, prepare_context_update
from app.agents.models.resilience import build_fallback_chat_model
from app.agents.orchestration.state import InsuranceAgentState
from app.agents.routing.analyzer import (
    InputAnalyzer,
    get_input_analyzer,
    init_input_analyzer,
)
from app.agents.routing.router import (
    IntentRouter,
    get_intent_router,
    init_intent_router,
)
from app.agents.routing.schemas import Intent, RouteResult
from app.agents.safety.guardrails import (
    get_guardrails,
    init_guardrails,
    input_block_reply,
    output_block_reply,
)
from app.agents.tools.contracts import WorkflowTools
from app.agents.workflows import (
    chitchat_node,
    claim_node,
    clarify_node,
    fallback_node,
    human_handoff_node,
    init_workflows,
    recommendation_node,
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 工作流节点名
WorkflowNode = Literal[
    "recommendation_node",
    "claim_node",
    "chitchat_node",
    "human_handoff_node",
    "fallback_node",
]

# route 节点可能跳转的目标（工作流节点 + 澄清节点）
RouteTarget = Literal[
    "recommendation_node",
    "claim_node",
    "chitchat_node",
    "human_handoff_node",
    "fallback_node",
    "clarify_node",
]

# 意图 → 工作流节点，分类和路由解耦，分类器只输出 intent
INTENT_TO_NODE: dict[Intent, WorkflowNode] = {
    "recommendation_plan": "recommendation_node",
    "claim": "claim_node",
    "chitchat": "chitchat_node",
    "human_handoff": "human_handoff_node",
    "fallback": "fallback_node",
}

# 最大澄清次数，超过后直接兜底，避免死循环
MAX_CLARIFICATION_ROUNDS = 2

# 是否把「输入安全校验」和「L2 意图分类」合并成一次模型调用
ENABLE_COMBINED_INPUT_ANALYSIS = True


def _message_text(message: BaseMessage) -> str:
    """提取消息中的纯文本内容。

    Args:
        message: LangChain 消息对象。

    Returns:
        消息的文本内容；多模态内容会拼接其中的文本块。
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _build_context(
        messages: list[BaseMessage],
        active_workflow: Intent | None,
        summary: str = "",
) -> str:
    """构建意图分类用的历史上下文。

    Args:
        messages: 当前会话的消息列表，最后一条为本次用户输入。
        active_workflow: 上一轮的主流程意图，用于短追问继承。
        summary: 较早会话历史的摘要。

    Returns:
        格式化的历史上下文文本；没有历史消息时返回“无”。
    """
    # 最后一条是本次 query，由 route() 单独传入，这里只取历史部分
    recent = messages[-8:-1]
    history_lines = []
    for message in recent:
        text = _message_text(message).strip()
        if not text:
            continue
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            role = message.type
        history_lines.append(f"{role}: {text}")

    history = "\n".join(history_lines) if history_lines else "无"
    summary_text = summary.strip()[:500] or "无"
    return (
        f"上一轮主流程意图: {active_workflow or '无'}\n"
        f"历史摘要: {summary_text}\n"
        f"最近对话:\n{history}"
    )


# 1. 定义节点
# 1.0 上下文治理节点：压缩过长的会话历史
async def prepare_context_node(state: InsuranceAgentState) -> dict:
    """在每轮开始前压缩过长的会话历史。

    Args:
        state: 全局状态。

    Returns:
        摘要与待删除消息的更新；未触发摘要时为空字典。
    """
    return await prepare_context_update(state)


# 1.1 用户输入安全合规校验节点
async def input_guard_node(
        state: InsuranceAgentState,
) -> Command[Literal["route", "guard_reply"]]:
    """校验用户输入是否安全、合法、合规。

    先跑 L0/L1：命中时只做安全校验，意图结果直接缓存；
    未命中时把安全校验和 L2 意图分类合并成一次模型调用。
    校验结果写入 guard_* 字段，意图结果存入 precomputed_route 供 route 节点复用。

    Args:
        state: 全局状态。

    Returns:
        放行时跳转到 route，拦截时跳转到 guard_reply。
    """
    messages = state.get("messages", [])
    active_workflow = state.get("active_workflow") or None
    user_text = _message_text(messages[-1]) if messages else ""
    context = _build_context(
        messages,
        active_workflow,
        str(state.get("summary") or ""),
    )
    intent_router = get_intent_router()

    # 1. 先跑便宜的 L0 规则、低信息量判断和 L1 语义
    try:
        fast_result = await intent_router.classify_fast(user_text, active_workflow)
    except Exception as exc:
        logger.warning(f"L0/L1 路由失败，等待合并分析: {exc}")
        fast_result = None

    precomputed_route = None
    precomputed_layer = None

    if fast_result is not None:
        # 2a. L0/L1 命中：只做安全校验，意图结果直接缓存
        fast_route, fast_layer = fast_result
        precomputed_route = fast_route.model_dump()
        precomputed_layer = fast_layer
        decision = await get_guardrails().check_input(user_text)
    elif ENABLE_COMBINED_INPUT_ANALYSIS:
        # 2b. L0/L1 未命中：安全校验 + L2 意图分类合并成一次调用
        analysis = await get_input_analyzer().analyze(user_text, context)
        decision = analysis.guard
        if analysis.route is not None:
            precomputed_route = analysis.route.model_dump()
            precomputed_layer = "llm"
    else:
        # 2c. 降级路径：只做安全校验，意图完全交给 route 节点
        decision = await get_guardrails().check_input(user_text)

    update = {
        "guard_stage": "input",
        "guard_blocked": not decision.allowed,
        "guard_category": decision.category,
        "guard_reason": decision.reason,
        "precomputed_route": precomputed_route,
        "precomputed_layer": precomputed_layer,
        # 每轮重置，避免上一轮的转人工/失败标记影响本轮
        "handoff_requested": False,
        "workflow_failed": False,
        "workflow_error": None,
        "final_answer": "",
    }

    if not decision.allowed:
        logger.warning(
            f"用户输入被安全校验拦截: category={decision.category}, reason={decision.reason}"
        )
        return Command(update=update, goto="guard_reply")

    return Command(update=update, goto="route")


# 1.2 输入拦截回复节点
def guard_reply_node(state: InsuranceAgentState) -> dict:
    """返回固定的输入拦截话术。

    Args:
        state: 全局状态。

    Returns:
        包含安全话术的 AIMessage。
    """
    category = state.get("guard_category", "other")
    answer = input_block_reply(category)
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
    }


# 1.3 意图识别节点
async def route_node(state: InsuranceAgentState) -> Command[RouteTarget]:
    """识别用户意图并路由到对应工作流。

    Args:
        state: 全局状态。

    Returns:
        更新分类结果并跳转到工作流节点或澄清节点。
    """
    messages = state.get("messages", [])
    active_workflow = state.get("active_workflow") or None
    query = _message_text(messages[-1]) if messages else ""
    context = _build_context(
        messages,
        active_workflow,
        str(state.get("summary") or ""),
    )
    intent_router = get_intent_router()

    # 1. 优先复用输入分析阶段已经算好的结果（L0 / L1 / L2）
    result = None
    route_layer = None
    precomputed = state.get("precomputed_route")
    if precomputed:
        try:
            result = RouteResult.model_validate(precomputed)
            route_layer = state.get("precomputed_layer") or "llm"
        except Exception as exc:
            logger.warning(f"预计算意图结果无效，重新路由: {exc}")

    # 2. 没有预计算结果时，重新跑 L0 → L1 → L2
    if result is None:
        try:
            fast_result = await intent_router.classify_fast(query, active_workflow)
        except Exception as exc:
            logger.warning(f"L0/L1 路由失败，降级到 L2: {exc}")
            fast_result = None

        if fast_result is not None:
            result, route_layer = fast_result
        else:
            try:
                result = await intent_router.classify_l2(query, context)
                route_layer = "llm"
            except Exception as exc:
                # 模型都失败时，先用关键词/上下文规则兜底，避免直接甩给用户
                logger.error(f"意图分类失败，尝试规则兜底: {exc}")
                rule_result = intent_router.infer_intent_by_rules(
                    query,
                    active_workflow,
                )
                if rule_result is not None:
                    result = rule_result
                    route_layer = "rule"
                else:
                    result = RouteResult(
                        intent="fallback",
                        confidence=0.0,
                        reason="模型不可用且规则未命中，进入兜底",
                    )
                    route_layer = "fallback"

    # 3. 模型能调通但给出 fallback 时，再用关键词/上下文规则补一刀
    if (
            result.intent == "fallback"
            and not result.needs_clarification
            and route_layer == "llm"
    ):
        rule_result = intent_router.infer_intent_by_rules(query, active_workflow)
        if rule_result is not None:
            logger.info(f"模型返回 fallback，规则接管: intent={rule_result.intent}")
            result = rule_result
            route_layer = "rule"

    # 当前轮分类结果始终写入；active_workflow 只在进入工作流时更新，
    # 澄清/兜底期间保留上一轮主流程，避免短追问丢失上下文
    update = result.to_state_update()
    update["route_layer"] = route_layer
    update["route_similarity"] = result.confidence if route_layer == "semantic" else None

    clarification_count = state.get("clarification_count", 0)
    should_clarify = intent_router.should_clarify(result, route_layer)

    # 低置信度先澄清；达到次数上限后兜底，避免死循环
    if should_clarify and clarification_count < MAX_CLARIFICATION_ROUNDS:
        update["clarification_count"] = clarification_count + 1
        logger.info(
            f"意图置信度不足，进入澄清: intent={result.intent}, "
            f"confidence={result.confidence:.2f}"
        )
        return Command(update=update, goto="clarify_node")

    if should_clarify:
        logger.info("澄清次数达到上限，转入兜底")
        return Command(update=update, goto="fallback_node")

    # 明确进入工作流：记录工作流切换，并重置澄清计数
    if active_workflow is not None and active_workflow != result.intent:
        update["previous_workflow"] = active_workflow
    update["active_workflow"] = result.intent
    update["clarification_count"] = 0
    return Command(update=update, goto=INTENT_TO_NODE[result.intent])


# 1.5 模型输出安全合规校验节点
async def output_guard_node(state: InsuranceAgentState) -> dict:
    """校验模型输出，命中合规风险时原位替换为安全话术。

    Args:
        state: 全局状态。

    Returns:
        校验结果；需要拦截时返回同 id 的新 AIMessage 覆盖原回答。
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {}

    original_answer = _message_text(last)
    decision = await get_guardrails().check_output(original_answer)
    update = {
        "guard_stage": "output",
        "guard_blocked": not decision.allowed,
        "guard_category": decision.category,
        "guard_reason": decision.reason,
    }

    if decision.allowed:
        # final_answer 供 runner 在安全校验通过后流式推送
        update["final_answer"] = original_answer
        return update

    logger.warning(
        f"模型输出被安全校验拦截: category={decision.category}, reason={decision.reason}"
    )
    replacement = output_block_reply(decision.category)
    # 使用相同 id 覆盖原消息，避免不安全内容留在会话历史里
    update["messages"] = [AIMessage(id=last.id, content=replacement)]
    update["final_answer"] = replacement
    return update


# 2. 创建 Graph，用 Edge 和 Command 表示工作流之间的关系
async def init_insurance_agent_graph(
        checkpointer=None,
        model: BaseChatModel | None = None,
        intent_router: IntentRouter | None = None,
        input_analyzer: InputAnalyzer | None = None,
        fallback_model: BaseChatModel | None = None,
        tools: WorkflowTools | None = None,
        tool_middlewares: list[Any] | None = None,
        model_ready: Callable[[BaseChatModel], None] | None = None,
) -> CompiledStateGraph:
    """初始化并编译保险顾问主 Graph。

    Args:
        checkpointer: 会话记忆存储器，用于持久化对话状态。
        model: 可选的聊天模型，便于测试时注入；为空时使用默认 DeepSeek 配置。
        intent_router: 可选的意图路由器，便于测试时注入；为空时使用默认本地模型。
        input_analyzer: 可选的输入分析器，便于测试时注入；为空时自动创建。
        fallback_model: 主模型故障时的兜底模型，可选。
        tools: 各工作流可用的工具集合，由 harness 注入。
        tool_middlewares: 工具中间件，由 harness 注入。
        model_ready: 模型就绪后的回调，供 harness 初始化画像抽取等组件。

    Returns:
        编译后的 LangGraph 图。
    """
    # 1. 初始化模型；注入模型时默认不创建真实兜底，保证测试离线
    if model is None:
        model = init_chat_model(
            model=settings.llm.deepseek_chat_model,
            model_provider="deepseek",
            api_key=settings.llm.deepseek_api_key,
            temperature=0,  # 分类和校验需要稳定输出
            extra_body={"thinking": {"type": "disabled"}},
        )
        if fallback_model is None:
            fallback_model = build_fallback_chat_model()

    # 模型确定后通知装配层，避免 agents 直接依赖画像抽取等业务组件
    if model_ready is not None:
        model_ready(model)

    # 2. 初始化工作流：业务工具由 harness 注入，agents 不直接依赖 modules
    await init_workflows(
        model,
        fallback_model,
        tools=tools,
        tool_middlewares=tool_middlewares,
    )

    # 2.1 初始化上下文摘要模型，历史过长时压缩
    init_context_manager(model)

    # 3. 意图路由器：默认本地小模型，DeepSeek 作为 L2 兜底
    intent_router = intent_router or IntentRouter(fallback_model=model)
    await init_intent_router(intent_router)

    # 4. 安全校验器：负责输出侧校验和降级路径
    init_guardrails(model, fallback_model)

    # 5. 合并输入分析：优先复用意图路由器的主模型，DeepSeek 兜底
    if input_analyzer is None:
        analyzer_model = getattr(intent_router, "model", model)
        input_analyzer = InputAnalyzer(
            analyzer_model,
            fallback_model=model,
        )
    init_input_analyzer(input_analyzer)

    # 6. 构造器
    builder = StateGraph(InsuranceAgentState)

    # 7. 添加节点
    builder.add_node("prepare_context", prepare_context_node)
    builder.add_node("input_guard", input_guard_node)
    builder.add_node("guard_reply", guard_reply_node)
    builder.add_node("route", route_node)
    builder.add_node("clarify_node", clarify_node)
    builder.add_node("recommendation_node", recommendation_node)
    builder.add_node("claim_node", claim_node)
    builder.add_node("chitchat_node", chitchat_node)
    builder.add_node("human_handoff_node", human_handoff_node)
    builder.add_node("fallback_node", fallback_node)
    builder.add_node("output_guard", output_guard_node)

    # 8. 添加边；input_guard 和 route 通过 Command(goto=...) 动态跳转
    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "input_guard")
    for workflow_node in (
            "recommendation_node",
            "claim_node",
            "chitchat_node",
            "human_handoff_node",
            "fallback_node",
            "clarify_node",
    ):
        builder.add_edge(workflow_node, "output_guard")
    builder.add_edge("output_guard", END)
    builder.add_edge("guard_reply", END)

    # 9. 编译
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("保险顾问主 Graph 初始化完成~☑️")
    return graph
