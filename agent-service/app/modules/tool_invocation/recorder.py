"""工具调用记录写入，后台异步执行，不阻塞工具调用。"""

import asyncio
from typing import Any

from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.tool_invocation.models import ToolInvocation

logger = get_logger(__name__)

# 参数和结果预览的最大字符数，避免敏感信息大量落库
MAX_PREVIEW_CHARS = 500

# 持有后台任务引用，避免任务被垃圾回收
_background_tasks: set[asyncio.Task] = set()


def _safe_args(args: dict[str, Any]) -> dict[str, Any]:
    """截断参数中的长文本，避免敏感信息完整落库。

    Args:
        args: 原始工具参数。

    Returns:
        截断后的参数字典。
    """
    safe: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            safe[key] = value[:MAX_PREVIEW_CHARS]
        elif isinstance(value, list):
            safe[key] = [
                item[:100] if isinstance(item, str) else item
                for item in value[:20]
            ]
        else:
            safe[key] = value
    return safe


def schedule_tool_invocation_record(
        *,
        tool_name: str,
        tool_call_id: str | None,
        args: dict[str, Any],
        status: str,
        thread_id: str | None = None,
        user_id: int | None = None,
        result_preview: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        duration_ms: int = 0,
) -> None:
    """在后台写入一条工具调用记录。

    Args:
        tool_name: 工具名。
        tool_call_id: 模型生成的工具调用 ID。
        args: 工具参数。
        status: 调用状态。
        thread_id: 会话 ID。
        user_id: 用户 ID。
        result_preview: 结果预览。
        error_code: 错误码。
        error_message: 错误说明。
        duration_ms: 总耗时，毫秒。
    """
    task = asyncio.create_task(
        _write_record(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            args=args,
            status=status,
            thread_id=thread_id,
            user_id=user_id,
            result_preview=result_preview,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        ),
        name=f"tool-invocation-{tool_name}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def shutdown_tool_invocation_recorder(timeout: float = 3.0) -> None:
    """等待未完成的工具调用记录写入任务。

    应用关闭时调用，避免进程退出导致记录丢失。

    Args:
        timeout: 最长等待秒数。
    """
    if not _background_tasks:
        return

    pending = list(_background_tasks)
    await asyncio.wait(pending, timeout=timeout)
    logger.info(f"工具调用记录已刷新，剩余任务 {len(_background_tasks)} 个")


async def _write_record(
        *,
        tool_name: str,
        tool_call_id: str | None,
        args: dict[str, Any],
        status: str,
        thread_id: str | None,
        user_id: int | None,
        result_preview: str | None,
        error_code: str | None,
        error_message: str | None,
        duration_ms: int,
) -> None:
    """写入工具调用记录。

    Args:
        tool_name: 工具名。
        tool_call_id: 工具调用 ID。
        args: 工具参数。
        status: 调用状态。
        thread_id: 会话 ID。
        user_id: 用户 ID。
        result_preview: 结果预览。
        error_code: 错误码。
        error_message: 错误说明。
        duration_ms: 总耗时，毫秒。
    """
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                session.add(
                    ToolInvocation(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        args=_safe_args(args),
                        status=status,
                        thread_id=thread_id,
                        user_id=user_id,
                        result_preview=(result_preview or "")[:MAX_PREVIEW_CHARS] or None,
                        error_code=error_code,
                        error_message=(error_message or "")[:MAX_PREVIEW_CHARS] or None,
                        duration_ms=duration_ms,
                    )
                )
    except Exception as exc:
        logger.warning(f"工具调用记录写入失败: tool={tool_name}, error={exc}")
