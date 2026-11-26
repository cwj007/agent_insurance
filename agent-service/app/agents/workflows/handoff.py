"""人工转接工作流节点。"""

from langchain_core.messages import AIMessage

from app.agents.orchestration.state import InsuranceAgentState
from app.agents.workflows.prompts import HANDOFF_MESSAGE
from app.core.logging import get_logger

logger = get_logger(__name__)


def human_handoff_node(state: InsuranceAgentState) -> dict:
    """人工转接节点。

    当前先记录请求和状态标记；实际客服系统接入后，
    在这里创建工单或通知人工坐席。

    Args:
        state: 全局状态。

    Returns:
        包含转接话术的 AIMessage 和 handoff_requested 标记。
    """
    # 不在日志里打印用户原文，避免保单号、健康状况等敏感信息落盘
    logger.warning("用户请求转人工，已设置 handoff_requested 标记")

    return {
        "messages": [AIMessage(content=HANDOFF_MESSAGE)],
        "handoff_requested": True,
    }
