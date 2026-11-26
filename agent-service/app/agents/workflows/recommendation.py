"""保险方案推荐工作流节点。"""

from langchain_core.runnables import RunnableConfig

from app.agents.orchestration.state import InsuranceAgentState
from app.agents.workflows.runtime import (
    _build_agent_context,
    _run_agent_workflow,
    get_recommendation_agent,
)


async def recommendation_node(
        state: InsuranceAgentState,
        config: RunnableConfig | None = None,
) -> dict:
    """保险方案推荐咨询节点。

    Args:
        state: 全局状态。
        config: LangGraph 运行时配置，包含本轮的用户 ID 与画像文本。

    Returns:
        包含推荐方案的 AIMessage。
    """
    # 从运行时配置读取用户画像，由子 Agent 的 middleware 注入
    return await _run_agent_workflow(
        get_recommendation_agent(),
        state,
        "推荐",
        context=_build_agent_context(config),
    )
