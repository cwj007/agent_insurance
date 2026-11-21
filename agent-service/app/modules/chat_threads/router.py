"""会话管理模块路由"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.chat_threads.schemas import ChatThreadResponse, ChatThreadCreate, ChatHistoryResponse
from app.modules.chat_threads.service import ChatThreadService

router = APIRouter(prefix="/api/v1/chat-threads", tags=['会话管理的接口'])


# 1. 抽取公共函数，生成 service
def get_chat_thread_service(
        request: Request, # 注入 agent
        session: AsyncSession = Depends(get_session)
):
    """获取数据库会话服务"""

    # 在初始化时，就获取 agent
    agent = request.app.state.agent

    return ChatThreadService(session, agent)


@router.post("", response_model=ChatThreadResponse, summary="添加会话")
async def create_chat_thread(
        request: ChatThreadCreate,
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """添加会话"""

    return await service.add(user_id, request.title)


@router.get("", response_model=list[ChatThreadResponse], summary="根据用户ID查询会话列表")
async def list_by_threads(
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """根据用户ID查询会话列表"""

    return await service.list_by_user(user_id)


@router.patch("/{thread_id}", response_model=ChatThreadResponse, summary="重命名会话")
async def rename_chat_thread(
        thread_id: UUID,  # 路径参数
        request: ChatThreadCreate,
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """重命名会话"""

    return await service.rename(user_id, thread_id, request.title)


@router.get("/{thread_id}/messages", response_model=ChatHistoryResponse, summary="查询会话历史消息")
async def get_chat_history(
        thread_id: UUID,  # 路径参数
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """查询当前用户的指定会话的历史消息"""

    return await service.get_chat_history(user_id, thread_id)