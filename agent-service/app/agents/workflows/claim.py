"""理赔咨询工作流节点。"""

from langchain_core.runnables import RunnableConfig

from app.agents.orchestration.state import InsuranceAgentState
from app.agents.workflows.prompts import CLAIM_PROMPT
from app.agents.workflows.runtime import (
    _build_agent_context,
    _run_agent_workflow,
    _run_text_workflow,
    get_claim_agent,
)


async def claim_node(
        state: InsuranceAgentState,
        config: RunnableConfig | None = None,
) -> dict:
    """理赔咨询节点。

    使用条款检索工具回答具体赔付责任；Agent 初始化失败时降级为纯提示词回答。

    Args:
        state: 全局状态。
        config: LangGraph 运行时配置，包含用户与会话上下文。

    Returns:
        包含理赔说明的 AIMessage。
    """
    agent = get_claim_agent()
    if agent is None:
        return await _run_text_workflow(state, CLAIM_PROMPT, "理赔")

    return await _run_agent_workflow(
        agent,
        state,
        "理赔",
        context=_build_agent_context(config),
    )
