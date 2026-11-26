"""用户情节记忆响应结构。"""

from datetime import datetime

from pydantic import Field

from app.common.schemas import BaseSchema


class UserMemoryResponse(BaseSchema):
    """一条用户情节记忆的响应。"""

    id: int = Field(description="记忆 ID")
    kind: str = Field(description="类型：fact / preference / event / decision")
    content: str = Field(description="记忆内容")
    importance: float = Field(description="重要度，0~1")
    thread_id: str | None = Field(default=None, description="来源会话 ID")
    created_at: datetime = Field(description="记忆时间")
