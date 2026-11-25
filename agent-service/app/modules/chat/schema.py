"""聊天模块请求响应模型。"""

from uuid import UUID

from pydantic import Field

from app.common.schemas import BaseSchema


class ChatRequest(BaseSchema):
    """聊天请求模型。"""

    thread_id: UUID
    message: str = Field(description="用户消息", min_length=1) # 用户消息，不能为空