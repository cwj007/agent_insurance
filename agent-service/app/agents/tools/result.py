"""工具统一返回协议与错误类型。"""

import json
from typing import Any, Literal

from pydantic import Field

from app.common.schemas import BaseSchema

# 工具错误码，与 docs §4.3 保持一致
ToolErrorCode = Literal[
    "INVALID_ARGS",  # 参数非法，模型可修正
    "PRODUCT_REQUIRED",  # 缺少产品 ID
    "NOT_FOUND",  # 查询无结果
    "TIMEOUT",  # 超时
    "UPSTREAM_5XX",  # 上游服务异常
    "RATE_LIMIT",  # 限流
    "PERMISSION",  # 无权限
    "CONTRACT",  # 返回结构异常
    "INTERNAL",  # 其他内部错误
]


def parse_tool_result(content: Any) -> dict[str, Any] | None:
    """解析工具返回的 JSON 对象。

    Args:
        content: 工具返回的原始内容，通常是 JSON 字符串。

    Returns:
        解析后的字典；内容不是 JSON 对象时返回 None。
    """
    if not isinstance(content, str):
        return None

    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None

    return payload if isinstance(payload, dict) else None


class ToolError(BaseSchema):
    """工具错误信息。"""

    code: ToolErrorCode = Field(description="错误码")
    message: str = Field(description="给模型看的错误说明，必须可操作")
    retryable: bool = Field(default=False, description="是否可重试")
    hint: str | None = Field(default=None, description="下一步建议")


class ToolResult(BaseSchema):
    """工具统一返回结构。

    所有工具都返回本结构的 JSON 字符串：成功时包含 data，
    失败时包含结构化的 error，模型据此决定修正参数、换工具或告知用户。
    """

    ok: bool = Field(description="是否成功")
    data: Any = Field(default=None, description="业务数据；失败时为 null")
    error: ToolError | None = Field(default=None, description="错误信息；成功时为 null")

    @classmethod
    def success(cls, data: Any) -> "ToolResult":
        """构造成功结果。

        Args:
            data: 业务数据。

        Returns:
            成功的 ToolResult。
        """
        return cls(ok=True, data=data)

    @classmethod
    def failure(
            cls,
            code: ToolErrorCode,
            message: str,
            *,
            retryable: bool = False,
            hint: str | None = None,
    ) -> "ToolResult":
        """构造失败结果。

        Args:
            code: 错误码。
            message: 给模型看的错误说明。
            retryable: 是否可重试。
            hint: 下一步建议。

        Returns:
            失败的 ToolResult。
        """
        return cls(
            ok=False,
            error=ToolError(
                code=code,
                message=message,
                retryable=retryable,
                hint=hint,
            ),
        )

    def to_json(self) -> str:
        """序列化为 JSON 字符串，作为工具返回值。

        Returns:
            JSON 字符串；值为 None 的字段会被省略。
        """
        return self.model_dump_json(exclude_none=True)


class ToolTransientError(Exception):
    """工具可重试的瞬时异常，交由 ToolRetryMiddleware 重试。"""
