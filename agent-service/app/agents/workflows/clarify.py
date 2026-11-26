"""澄清工作流节点。"""

from langchain_core.messages import AIMessage

from app.agents.orchestration.state import InsuranceAgentState


def clarify_node(state: InsuranceAgentState) -> dict:
    """向用户追问缺失信息。

    Args:
        state: 全局状态。

    Returns:
        包含澄清问题的 AIMessage。
    """
    question = state.get("clarification_question") or (
        "为了更准确地帮您，请补充一下您想咨询的方向：\n"
        "1. 保险产品/方案推荐\n"
        "2. 理赔流程/材料\n"
        "3. 回复「转人工」联系人工客服"
    )
    return {"messages": [AIMessage(content=question)]}
