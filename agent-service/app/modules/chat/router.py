"""聊天模块路由。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Header
from fastapi.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.chat.schema import ChatRequest
from app.modules.chat.service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["与AI聊天对话"])


def get_chat_service(
        http_request: Request,  # 使用 fastapi 的 request 对象
        session: AsyncSession = Depends(get_session),  # 2. 缺少 session，需要依赖注入
):
    """获取 ChatService 实例。

    Args:
        http_request: FastAPI 请求对象，用于从 app.state 中获取 agent。
        session: 数据库异步会话，由 FastAPI 依赖注入。

    Returns:
        绑定当前请求会话与 agent 的 ChatService 实例。
    """
    agent = http_request.app.state.agent  # 3. 缺少 agent，需要用 fastapi 中的 state.agent
    return ChatService(session, agent)


# SSE 流式对话接口
@router.post("", response_class=EventSourceResponse)
async def chat_stream(
        request: ChatRequest,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        service: ChatService = Depends(get_chat_service),  # 1. 需要先调用 service
):
    """实现 Agent 流式对话。

    Args:
        request: 聊天请求体，包含会话 ID 和用户消息。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Yields:
        流式返回的 SSE 事件，文本片段事件为 message，结束事件为 done。
    """
    # 调用 ChatService 实例的 chat_stream 方法
    async for sse in service.chat_stream(user_id, request):
        yield sse # 这里是返回给前端的 ServerSentEvent
