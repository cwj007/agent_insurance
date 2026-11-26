"""兜底工作流节点。"""

from langchain_core.messages import AIMessage

from app.agents.orchestration.state import InsuranceAgentState


def fallback_node(state: InsuranceAgentState) -> dict:
    """兜底节点。

    不直接转人工，而是给出可选方向，让用户继续表达或主动选择人工。

    Args:
        state: 全局状态。

    Returns:
        包含引导选项的 AIMessage。
    """
    return {
        "messages": [
            AIMessage(
                content=(
                    "抱歉，我暂时没能准确理解您的需求。您可以：\n"
                    "1. 换个说法描述问题\n"
                    "2. 咨询保险产品/方案\n"
                    "3. 咨询理赔流程/材料\n"
                    "4. 回复「转人工」联系人工客服"
                )
            )
        ]
    }
