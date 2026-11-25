"""保险顾问智能体。"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model as _init_chat_model
from langgraph.graph.state import CompiledStateGraph

from app.agents.tools import query_candidate_products
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
你是“安心保”的智能保险顾问。
你需要使用专业、准确、容易理解的语言回答用户的保险问题。
当信息不足时，应先向用户追问，不要编造保险产品或保障内容。
你可以使用 query_candidate_products 工具查询保险产品。
"""


async def init_insurance_agent(checkpointer) -> CompiledStateGraph:
    """创建保险顾问智能体。

    Args:
        checkpointer: 会话记忆存储器，用于持久化对话状态。

    Returns:
        编译后的 LangGraph agent 实例。
    """
    # 1. 初始化大模型（关闭深度思考）
    model = _init_chat_model(
        model=settings.llm.chat_model,
        model_provider="deepseek",
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # 2. 初始化工具（后续补充）
    tools: list = [query_candidate_products]

    # 3. 创建 Agent，绑定模型、工具、会话记忆
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT,
    )

    logger.info("保险顾问智能体初始化成功~☑️")

    return agent