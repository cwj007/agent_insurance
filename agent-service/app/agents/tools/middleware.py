"""工具调用中间件：次数上限、错误转换与重试。"""

from decimal import Decimal
from typing import Any

from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import ToolMessage

from app.agents.tools.result import (
    ToolResult,
    ToolTransientError,
    parse_tool_result,
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 单次运行内的工具调用上限，防止模型死循环
GLOBAL_RUN_TOOL_LIMIT = settings.tools.global_run_limit

# 单个工具在单次运行内的调用上限
PER_TOOL_RUN_LIMIT = settings.tools.per_tool_run_limit

# 重试配置：只重试瞬时异常，重试耗尽后交给 ToolErrorMiddleware
MAX_TOOL_RETRIES = settings.tools.max_retries
RETRY_INITIAL_DELAY = settings.tools.retry_initial_delay
RETRY_MAX_DELAY = settings.tools.retry_max_delay


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


def classify_tool_result(result: ToolMessage) -> tuple[str, str | None, str | None]:
    """判断工具调用结果的状态、错误码和错误信息。

    ToolMessage.status 只反映工具是否抛异常；工具正常返回的
    {"ok": false} 属于业务失败，同样需要按失败记录。

    Args:
        result: 工具执行返回的 ToolMessage。

    Returns:
        (status, error_code, error_message)；status 为 succeeded 或 failed。
    """
    payload = parse_tool_result(result.content)
    error = payload.get("error") if payload else None

    # 1. 工具抛异常时 ToolMessage.status=error；业务失败时 ok=false
    if result.status == "error" or (
            isinstance(payload, dict) and payload.get("ok") is False
    ):
        if isinstance(error, dict):
            return (
                "failed",
                error.get("code"),
                str(error.get("message") or "")[:500] or str(result.content)[:500],
            )
        return "failed", None, str(result.content)[:500]

    return "succeeded", None, None


def _on_tool_error(exc: Exception, request: ToolCallRequest) -> str:
    """把工具异常转换成模型可见的结构化错误。

    Args:
        exc: 工具抛出的异常。
        request: 工具调用请求。

    Returns:
        结构化错误 JSON 字符串，作为 ToolMessage 的内容返回给模型。
    """
    tool_name = str((request.tool_call or {}).get("name") or "unknown")
    args = (request.tool_call or {}).get("args") or {}
    logger.error(
        f"工具执行失败: tool={tool_name}, args={_safe_args(args)}, error={exc}",
        exc_info=True,
    )

    if isinstance(exc, ToolTransientError):
        return ToolResult.failure(
            "UPSTREAM_5XX",
            f"{tool_name} 暂时不可用",
            retryable=True,
            hint="不要编造工具结果；如果多次失败，请告知用户稍后再试",
        ).to_json()

    return ToolResult.failure(
        "INTERNAL",
        f"{tool_name} 执行失败",
        hint="不要编造工具结果；请换一种方式或告知用户暂时无法查询",
    ).to_json()


def build_tool_middlewares(
        per_tool_limits: dict[str, int] | None = None,
) -> list[Any]:
    """构建通用工具中间件列表。

    顺序约定：次数限制在最外层，其次是错误转换，重试在最内层；
    重试耗尽后的异常会被转换为模型可见的错误。
    调用记录属于 harness（运行时）职责，由 harness 在外层额外拼入。

    Args:
        per_tool_limits: 单工具调用上限，键为工具名；为空时只保留全局上限。

    Returns:
        可直接传给 create_agent 的 middleware 列表。
    """
    limits = per_tool_limits or {}
    return [
        # 1. 次数限制：全局和单工具，超过后阻止执行并提示模型
        ToolCallLimitMiddleware(
            run_limit=GLOBAL_RUN_TOOL_LIMIT,
            exit_behavior="continue",
        ),
        *[
            ToolCallLimitMiddleware(
                tool_name=tool_name,
                run_limit=run_limit,
                exit_behavior="continue",
            )
            for tool_name, run_limit in limits.items()
        ],
        # 2. 错误转换：异常转成结构化 ToolMessage，不让会话中断
        ToolErrorMiddleware(on_error=_on_tool_error),
        # 3. 最内层：只重试瞬时异常，耗尽后抛给外层错误处理
        ToolRetryMiddleware(
            max_retries=MAX_TOOL_RETRIES,
            retry_on=(ToolTransientError,),
            on_failure="error",
            initial_delay=RETRY_INITIAL_DELAY,
            max_delay=RETRY_MAX_DELAY,
        ),
    ]
