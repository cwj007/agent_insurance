"""工作流运行时：子 Agent 初始化与执行辅助。"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.agents.context.profile_middleware import ProfileContext
from app.agents.models.resilience import (
    ainvoke_with_fallback,
    build_model_reliability_middlewares,
)
from app.agents.orchestration.state import InsuranceAgentState
from app.agents.tools.contracts import WorkflowTools
from app.agents.tools.middleware import build_tool_middlewares
from app.agents.workflows.prompts import CLAIM_PROMPT, WORKFLOW_ERROR_MESSAGE
from app.agents.workflows.subagents import (
    SUBAGENT_SUMMARY_KEEP_MESSAGES,
    SUBAGENT_SUMMARY_TRIGGER_MESSAGES,
    init_insurance_agent,
)
from app.agents.workflows.trace import (
    WORKFLOW_TRACE_LIMIT,
    _last_ai_message,
    _prepend_summary,
    _tool_trace_item,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# 运行时组件，由 init_workflows 初始化
_workflow_model: BaseChatModel | None = None
_fallback_model: BaseChatModel | None = None
_recommendation_agent: CompiledStateGraph | None = None
_claim_agent: CompiledStateGraph | None = None


def get_workflow_model() -> BaseChatModel | None:
    """获取文本工作流模型。

    Returns:
        当前文本工作流模型；未初始化时返回 None。
    """
    return _workflow_model


def get_recommendation_agent() -> CompiledStateGraph | None:
    """获取推荐子 Agent。

    Returns:
        推荐子 Agent；未初始化时返回 None。
    """
    return _recommendation_agent


def get_claim_agent() -> CompiledStateGraph | None:
    """获取理赔子 Agent。

    Returns:
        理赔子 Agent；未初始化时返回 None。
    """
    return _claim_agent


async def init_workflows(
        model: BaseChatModel,
        fallback_model: BaseChatModel | None = None,
        tools: WorkflowTools | None = None,
        tool_middlewares: list[Any] | None = None,
) -> None:
    """初始化工作流运行时组件。

    Args:
        model: 工作流共用的聊天模型。
        fallback_model: 主模型故障时的兜底模型，可选。
        tools: 各工作流可用的工具集合，由 harness 注入。
        tool_middlewares: 工具中间件，由 harness 注入；为空时使用默认通用中间件。
    """
    global _workflow_model, _fallback_model, _recommendation_agent, _claim_agent
    _workflow_model = model
    _fallback_model = fallback_model
    workflow_tools = tools or WorkflowTools()
    middlewares = (
        tool_middlewares
        if tool_middlewares is not None
        else build_tool_middlewares()
    )

    # 推荐工作流需要产品查询和条款检索工具
    try:
        _recommendation_agent = await init_insurance_agent(
            model=model,
            fallback_model=fallback_model,
            tools=workflow_tools.recommendation,
            tool_middlewares=middlewares,
        )
    except Exception as exc:
        logger.error(f"推荐 Agent 初始化失败: {exc}")
        _recommendation_agent = None

    # 理赔工作流只需要条款检索工具（底层走 app.rag.retriever），
    # 同样接入模型与工具可靠性中间件
    try:
        _claim_agent = create_agent(
            model=model,
            tools=workflow_tools.claim,
            system_prompt=CLAIM_PROMPT,
            middleware=[
                *build_model_reliability_middlewares(fallback_model),
                SummarizationMiddleware(
                    model=model,
                    trigger=("messages", SUBAGENT_SUMMARY_TRIGGER_MESSAGES),
                    keep=("messages", SUBAGENT_SUMMARY_KEEP_MESSAGES),
                ),
                *middlewares,
            ],
            context_schema=ProfileContext,
        )
    except Exception as exc:
        logger.error(f"理赔 Agent 初始化失败: {exc}")
        _claim_agent = None


def _build_agent_context(config: RunnableConfig | None) -> ProfileContext:
    """从 LangGraph 运行时配置构建子 Agent 上下文。

    Args:
        config: LangGraph 运行时配置。

    Returns:
        子 Agent 的用户、会话与画像上下文。
    """
    configurable = (config or {}).get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return ProfileContext(
        user_id=configurable.get("user_id"),
        thread_id=str(thread_id) if thread_id else None,
        profile_text=str(configurable.get("user_profile_text") or ""),
        memory_text=str(configurable.get("user_memory_text") or ""),
    )


async def _run_text_workflow(
        state: InsuranceAgentState,
        system_prompt: str,
        label: str,
) -> dict:
    """执行不需要工具的文本工作流。

    Args:
        state: 全局状态。
        system_prompt: 工作流专用系统提示词。
        label: 日志中使用的名称。

    Returns:
        包含最终 AIMessage、失败标记和错误说明的状态更新。
    """
    model = _workflow_model
    if model is None:
        logger.error(f"{label}工作流模型未初始化")
        return {
            "messages": [AIMessage(content=WORKFLOW_ERROR_MESSAGE)],
            "workflow_failed": True,
            "workflow_error": f"{label}工作流模型未初始化",
        }

    messages = [
        SystemMessage(content=system_prompt),
        *_prepend_summary(state, state.get("messages", [])),
    ]
    try:
        response = await ainvoke_with_fallback(
            model,
            messages,
            _fallback_model,
            attempts=2,
            backoff=0.3,
            label=f"{label}工作流",
        )
        if isinstance(response, AIMessage) and response.content:
            return {
                "messages": [response],
                "workflow_failed": False,
                "workflow_error": None,
            }

        logger.warning(f"{label}工作流返回空回答")
        error_text = "工作流返回空回答"
    except Exception as exc:
        logger.error(f"{label}工作流执行失败: {exc}")
        error_text = str(exc)[:200]

    return {
        "messages": [AIMessage(content=WORKFLOW_ERROR_MESSAGE)],
        "workflow_failed": True,
        "workflow_error": error_text,
    }


async def _run_agent_workflow(
        agent: CompiledStateGraph | None,
        state: InsuranceAgentState,
        label: str,
        *,
        context: ProfileContext | None = None,
) -> dict:
    """执行绑定工具的 Agent 工作流，并保留工具执行轨迹。

    Args:
        agent: 已编译的 Agent；未初始化时返回兜底话术。
        state: 全局状态。
        label: 日志中使用的名称。
        context: 子 Agent 的运行时上下文，用于注入用户画像。

    Returns:
        包含最终 AIMessage、工具轨迹和失败标记的状态更新。
    """
    if agent is None:
        logger.error(f"{label} Agent 未初始化")
        return {
            "messages": [AIMessage(content=WORKFLOW_ERROR_MESSAGE)],
            "workflow_trace": [],
            "workflow_failed": True,
            "workflow_error": f"{label} Agent 未初始化",
        }

    trace: list[dict[str, Any]] = []
    final_output: dict[str, Any] | None = None
    error_text: str | None = None
    root_run_id: str | None = None

    try:
        async for event in agent.astream_events(
                {"messages": _prepend_summary(state, state.get("messages", []))},
                context=context,
                version="v2",
        ):
            kind = event.get("event")

            # 1. 记录工具执行轨迹，失败也保留，供下一轮参考
            if kind == "on_tool_end":
                trace.append(_tool_trace_item(label, event, "succeeded"))
            elif kind == "on_tool_error":
                trace.append(_tool_trace_item(label, event, "failed"))

            # 2. 用根 run_id 精确匹配最终状态；
            # 作为父 Graph 的节点运行时，根事件会带 parent_ids，不能只看 parent_ids
            elif kind == "on_chain_start" and root_run_id is None:
                root_run_id = str(event.get("run_id") or "") or None
            elif (
                    kind == "on_chain_end"
                    and root_run_id is not None
                    and str(event.get("run_id") or "") == root_run_id
            ):
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict) and "messages" in output:
                    final_output = output
    except Exception as exc:
        logger.error(f"{label}工作流执行失败: {exc}")
        error_text = str(exc)[:200]
        final_output = None

    trace = trace[-WORKFLOW_TRACE_LIMIT:]

    if final_output is not None:
        answer = _last_ai_message(final_output.get("messages", []))
        if answer is not None:
            return {
                "messages": [answer],
                "workflow_trace": trace,
                "workflow_failed": False,
                "workflow_error": None,
            }
        error_text = "工作流没有返回有效回答"

    return {
        "messages": [AIMessage(content=WORKFLOW_ERROR_MESSAGE)],
        "workflow_trace": trace,
        "workflow_failed": True,
        "workflow_error": error_text or "工作流执行失败",
    }
