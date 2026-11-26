"""保险顾问 Agent 的用户画像注入中间件。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import SystemMessage

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileContext:
    """Agent 运行时上下文。

    Attributes:
        user_id: 当前用户 ID。
        thread_id: 当前会话 ID，工具调用记录使用。
        profile_text: 已渲染的用户画像文本；为空表示不注入。
        memory_text: 已渲染的情节记忆文本；为空表示不注入。
    """

    user_id: int | None = None
    thread_id: str | None = None
    profile_text: str = ""
    memory_text: str = ""


def _read_context_value(context: Any, name: str) -> str:
    """从运行时上下文中读取文本字段。

    Args:
        context: LangGraph 运行时上下文，可能是 ProfileContext、dict 或 None。
        name: 字段名。

    Returns:
        字段文本；不存在时返回空字符串。
    """
    if context is None:
        return ""
    if isinstance(context, dict):
        return str(context.get(name) or "")
    return str(getattr(context, name, "") or "")


@dynamic_prompt
def inject_profile_prompt(request: ModelRequest[ProfileContext]) -> SystemMessage:
    """在静态系统提示词后追加用户档案。

    Args:
        request: 当前模型请求，包含静态 system message 与运行时上下文。

    Returns:
        拼接用户档案后的系统消息。
    """
    base_prompt = request.system_message.text if request.system_message else ""
    profile_text = _read_context_value(request.runtime.context, "profile_text")
    memory_text = _read_context_value(request.runtime.context, "memory_text")

    # 1. 静态规则在前，用户档案与历史记忆随后，属于相对稳定的前缀
    sections = [base_prompt]
    if profile_text:
        sections.append(profile_text)
        logger.debug("推荐 Agent 已注入用户画像")
    if memory_text:
        sections.append(memory_text)
        logger.debug("推荐 Agent 已注入历史记忆")

    # 2. 环境快照每轮可能变化，放在最后，减少对前缀缓存的影响
    current_date = datetime.now().strftime("%Y-%m-%d")
    sections.append(f"## 当前环境\n- 当前日期：{current_date}")

    return SystemMessage(
        content="\n\n".join(section for section in sections if section)
    )
