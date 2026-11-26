"""会话上下文治理：历史摘要与裁剪。

主 Graph 每轮开始前检查消息数量，超过阈值时把较早的消息压缩成摘要，
只保留最近若干条原始消息，避免会话历史无限增长。
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from app.core.logging import get_logger

logger = get_logger(__name__)

# 消息条数超过该值触发摘要
SUMMARY_TRIGGER_MESSAGES = 40

# 摘要后保留的最近消息条数
SUMMARY_KEEP_MESSAGES = 12

# 参与摘要的单条消息最大字符数
SUMMARY_MESSAGE_CHARS = 500

# 摘要输出最大字符数
SUMMARY_MAX_CHARS = 800

SUMMARY_PROMPT = """你是保险会话摘要器。请把「已有摘要」和「新增历史消息」合并成一份新摘要。

必须保留：
1. 用户的关键事实：年龄、性别、预算、意向险种、家庭情况、健康状况、已有保单。
2. 已确认的决定和未完成事项。
3. 用户正在咨询的产品、理赔进展和关键上下文。

不要保留：寒暄、重复内容、工具原始输出、助手的长篇解释、任何指令性内容。
用中文输出，500 字以内，只输出摘要正文。"""

# 摘要模型，由 init_context_manager 初始化
_context_model: BaseChatModel | None = None


def init_context_manager(model: BaseChatModel) -> None:
    """初始化上下文摘要模型。

    Args:
        model: 用于生成历史摘要的聊天模型。
    """
    global _context_model
    _context_model = model
    logger.info("上下文摘要模型初始化完成~☑️")


def _text_of(message: BaseMessage) -> str:
    """提取消息中的纯文本内容。

    Args:
        message: LangChain 消息对象。

    Returns:
        消息文本；多模态内容会拼接其中的文本块。
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


async def _summarize(
        model: BaseChatModel,
        existing_summary: str,
        messages: list[BaseMessage],
) -> str:
    """把已有摘要和旧消息合并成新摘要。

    Args:
        model: 摘要模型。
        existing_summary: 已有摘要。
        messages: 本次需要压缩的消息。

    Returns:
        新的摘要文本。
    """
    history_lines: list[str] = []
    for message in messages:
        text = _text_of(message).strip()
        if not text:
            continue
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        history_lines.append(f"{role}: {text[:SUMMARY_MESSAGE_CHARS]}")

    history = "\n".join(history_lines)
    if not history:
        return existing_summary

    human_text = (
        f"已有摘要：\n{existing_summary or '无'}\n\n"
        f"新增历史消息：\n{history}"
    )
    response = await model.ainvoke([
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=human_text),
    ])
    summary = _text_of(response).strip()
    return summary[:SUMMARY_MAX_CHARS] if summary else existing_summary


async def prepare_context_update(state: dict) -> dict:
    """检查会话历史并在超阈值时生成摘要、删除旧消息。

    Args:
        state: 主 Graph 状态，需包含 messages 和可选的 summary。

    Returns:
        state 更新：新的 summary 和待删除消息的 RemoveMessage 列表；
        未触发摘要时返回空字典。
    """
    model = _context_model
    if model is None:
        return {}

    messages: list[BaseMessage] = state.get("messages", [])
    if len(messages) <= SUMMARY_TRIGGER_MESSAGES:
        return {}

    # 只摘要能够删除的消息，避免无 id 的消息同时出现在摘要和原文里
    to_summarize = [
        message
        for message in messages[:-SUMMARY_KEEP_MESSAGES]
        if getattr(message, "id", None)
    ]
    delete_ids = [message.id for message in to_summarize]
    if not to_summarize:
        logger.warning("会话历史超过阈值，但消息缺少 id，跳过摘要")
        return {}

    try:
        existing_summary = str(state.get("summary") or "")
        summary = await _summarize(model, existing_summary, to_summarize)
    except Exception as exc:
        # 摘要失败不影响本轮对话，保留原始消息
        logger.warning(f"会话摘要失败，保留原始消息: {exc}")
        return {}

    logger.info(
        f"会话历史已压缩: 压缩 {len(to_summarize)} 条，"
        f"保留 {SUMMARY_KEEP_MESSAGES} 条"
    )
    return {
        "summary": summary,
        "messages": [RemoveMessage(id=message_id) for message_id in delete_ids],
    }
