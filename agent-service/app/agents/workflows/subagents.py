"""保险顾问智能体。"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model as _init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from app.agents.context.profile_middleware import ProfileContext, inject_profile_prompt
from app.agents.models.resilience import build_model_reliability_middlewares
from app.agents.tools.middleware import build_tool_middlewares
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 子 Agent 的上下文压缩阈值；单轮最多 10 次工具调用，30 条会在本轮中途触发
SUBAGENT_SUMMARY_TRIGGER_MESSAGES = 60
SUBAGENT_SUMMARY_KEEP_MESSAGES = 12

SYSTEM_PROMPT = """
你是“安心保”的智能保险顾问。
- 使用专业、准确、容易理解的语言回答用户的保险问题。
- 当信息不足时，先向用户追问关键信息（年龄、意向险种、预算、健康状况等）。
- 如果系统提供了「用户档案」，优先使用其中已知的信息，不要重复询问；档案中标注「待确认」的信息需要先向用户确认。
- 给用户推荐保险方案时，必须先调用 query_candidate_products 查询候选产品，再根据用户画像推荐合适的保险组合，并说明推荐理由。
- 回答保险产品的条款问题时，必须先调用 retrieve_insurance_clause 查询对应产品的保险条款，再根据检索结果回答。
- 如果用户用产品名称指定产品但没有产品 ID，调用 retrieve_insurance_clause 时传 product_name，由工具解析产品；不要猜测 product_id。
- 涉及条款和赔付责任时，必须基于检索结果，不要编造；没有相关条款时如实告知用户。
- 不要承诺赔付结果、收益或保证承保；提醒用户以保险合同和公司审核结果为准。
- 回答要简短、精确，不要长篇大论。
"""


async def init_insurance_agent(
        checkpointer=None,
        model: BaseChatModel | None = None,
        fallback_model: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
        tool_middlewares: list[Any] | None = None,
) -> CompiledStateGraph:
    """创建保险顾问智能体。

    Args:
        checkpointer: 会话记忆存储器，用于持久化对话状态；作为子 Agent 时可为空。
        model: 可选的聊天模型，便于复用外部已创建的模型；为空时使用默认 DeepSeek 配置。
        fallback_model: 主模型故障时的兜底模型，可选。
        tools: 业务工具，由 harness 注入；为空时不绑定工具。
        tool_middlewares: 工具中间件，由 harness 注入；为空时使用默认通用中间件。

    Returns:
        编译后的 LangGraph agent 实例。
    """
    # 1. 初始化大模型（关闭深度思考）
    if model is None:
        model = _init_chat_model(
            model=settings.llm.deepseek_chat_model,
            model_provider="deepseek",
            api_key=settings.llm.deepseek_api_key,
            extra_body={"thinking": {"type": "disabled"}},
        )

    # 2. 创建 Agent，绑定模型、注入的工具、会话记忆与画像注入中间件
    middlewares = (
        tool_middlewares
        if tool_middlewares is not None
        else build_tool_middlewares()
    )
    agent = create_agent(
        model=model,
        tools=tools or [],
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            # 1. 模型可靠性：主模型重试后仍失败时切换兜底模型
            *build_model_reliability_middlewares(fallback_model),
            # 2. 画像注入
            inject_profile_prompt,
            # 3. 子 Agent 的上下文安全网，防止单轮工具循环把历史撑爆
            SummarizationMiddleware(
                model=model,
                trigger=("messages", SUBAGENT_SUMMARY_TRIGGER_MESSAGES),
                keep=("messages", SUBAGENT_SUMMARY_KEEP_MESSAGES),
            ),
            # 4. 工具记录、限流、错误转换与重试（由 harness 组装）
            *middlewares,
        ],
        context_schema=ProfileContext,
    )

    logger.info("保险顾问智能体初始化成功~☑️")

    return agent
