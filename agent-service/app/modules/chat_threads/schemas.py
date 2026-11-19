from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator, BaseModel

from app.common.schemas import BaseSchema


class ChatThreadCreate(BaseSchema):
    """创建会话请求"""

    title: str = Field(default="新会话", min_length=1, max_length=200)

    # 校验标题是否为空
    @field_validator("title", mode="before") # 校验前执行
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return value.strip()


class ChatThreadResponse(BaseSchema):
    """会话响应"""

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


from typing import Literal


class ChatMessageResponse(BaseSchema):
    """历史消息"""

    role: Literal["user", "assistant"]
    content: str


class ChatHistoryResponse(BaseModel):
    """会话历史响应"""

    thread_id: UUID
    messages: list[ChatMessageResponse]
