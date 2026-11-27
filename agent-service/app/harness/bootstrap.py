"""Agent 装配：把 harness 的具体工具与中间件注入 agents Graph。"""

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from app.agents.orchestration.graph import init_insurance_agent_graph
from app.agents.routing.analyzer import InputAnalyzer
from app.agents.routing.router import IntentRouter
from app.harness.profile_extractor import init_profile_extractor
from app.harness.tools import build_agent_tool_middlewares, build_workflow_tools


async def build_insurance_agent_graph(
        checkpointer=None,
        model: BaseChatModel | None = None,
        intent_router: IntentRouter | None = None,
        input_analyzer: InputAnalyzer | None = None,
        fallback_model: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """构建生产用保险顾问主 Graph，装配具体工具与工具中间件。

    Args:
        checkpointer: 会话记忆存储器。
        model: 可选的聊天模型；为空时使用默认 DeepSeek 配置。
        intent_router: 可选的意图路由器。
        input_analyzer: 可选的输入分析器。
        fallback_model: 主模型故障时的兜底模型。

    Returns:
        编译后的 LangGraph 图。
    """
    return await init_insurance_agent_graph(
        checkpointer,
        model=model,
        intent_router=intent_router,
        input_analyzer=input_analyzer,
        fallback_model=fallback_model,
        tools=build_workflow_tools(),
        tool_middlewares=build_agent_tool_middlewares(),
        model_ready=init_profile_extractor,
        draw_graph=True,
    )
