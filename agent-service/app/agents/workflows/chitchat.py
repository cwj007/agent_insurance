"""闲聊工作流节点。"""

from app.agents.orchestration.state import InsuranceAgentState
from app.agents.workflows.prompts import CHITCHAT_PROMPT
from app.agents.workflows.runtime import _run_text_workflow


async def chitchat_node(state: InsuranceAgentState) -> dict:
    """闲聊节点。

    Args:
        state: 全局状态。

    Returns:
        包含闲聊回复的 AIMessage。
    """
    return await _run_text_workflow(state, CHITCHAT_PROMPT, "闲聊")
