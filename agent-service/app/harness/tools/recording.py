"""工具调用记录中间件：把每次调用写入 tool_invocations。"""

import time
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain_core.messages import ToolMessage

from app.agents.tools.middleware import classify_tool_result
from app.core.logging import get_logger
from app.modules.tool_invocation.recorder import schedule_tool_invocation_record

logger = get_logger(__name__)


def _read_context_value(context: Any, name: str) -> Any:
    """从运行时上下文中读取字段，兼容 dict 与 dataclass。

    Args:
        context: 运行时上下文。
        name: 字段名。

    Returns:
        字段值；不存在时返回 None。
    """
    if context is None:
        return None
    if isinstance(context, dict):
        return context.get(name)
    return getattr(context, name, None)


def _safe_args(args: dict[str, Any]) -> dict[str, Any]:
    """把参数转换成可 JSON 序列化的形式，并截断长文本。

    Args:
        args: 原始工具参数。

    Returns:
        可序列化且已截断的参数。
    """
    safe: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            safe[key] = value[:200]
        elif isinstance(value, Decimal):
            safe[key] = float(value)
        elif isinstance(value, list):
            safe[key] = [
                item[:100] if isinstance(item, str) else item
                for item in value[:20]
            ]
        else:
            safe[key] = value
    return safe


@wrap_tool_call
async def record_tool_invocation(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
) -> Any:
    """记录每次工具调用的参数、状态、错误和耗时。

    Args:
        request: 工具调用请求。
        handler: 执行工具的回调。

    Returns:
        工具执行结果，原样返回。

    Raises:
        Exception: 工具抛出的异常会记录后继续向上抛出。
    """
    tool_call = request.tool_call or {}
    tool_name = str(tool_call.get("name") or "unknown")
    tool_call_id = tool_call.get("id")
    args = tool_call.get("args") or {}
    context = request.runtime.context
    user_id = _read_context_value(context, "user_id")
    thread_id = _read_context_value(context, "thread_id")

    start = time.perf_counter()
    status = "succeeded"
    error_code = None
    error_message = None
    result_preview = None

    try:
        result = await handler(request)

        # 1. ToolMessage.status 和业务结果都参与失败判断
        if isinstance(result, ToolMessage):
            result_preview = str(result.content)
            status, error_code, error_message = classify_tool_result(result)
        return result

    except Exception as exc:
        status = "failed"
        error_message = str(exc)[:500]
        error_code = "INTERNAL"
        raise

    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        schedule_tool_invocation_record(
            tool_name=tool_name,
            tool_call_id=str(tool_call_id) if tool_call_id else None,
            args=_safe_args(args),
            status=status,
            thread_id=str(thread_id) if thread_id else None,
            user_id=user_id if isinstance(user_id, int) else None,
            result_preview=result_preview,
            error_code=error_code,
            error_message=error_message,
            duration_ms=duration_ms,
        )


def build_recording_middleware() -> Any:
    """构建工具调用记录中间件。

    Returns:
        可直接传给 create_agent 的 middleware。
    """
    return record_tool_invocation
