"""聊天模块路由"""
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
    """获取 ChatService 实例"""

    agent = http_request.app.state.agent  # 3. 缺少 agent，需要用 fastapi 中的 state.agent
    return ChatService(session, agent)


# SSE 流式对话接口
@router.post("", response_class=EventSourceResponse)
async def chat_stream(
        request: ChatRequest,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        service: ChatService = Depends(get_chat_service),  # 1. 需要先调用 service
):
    """实现 Agent 流式对话"""

    # 调用 ChatService 实例的 chat_stream 方法
    async for sse in service.chat_stream(user_id, request):
        yield sse # 这里是返回给前端的 ServerSentEvent
