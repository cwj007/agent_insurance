from typing import Annotated

from fastapi import APIRouter, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.chat_threads.schemas import ChatThreadResponse, ChatThreadCreate
from app.modules.chat_threads.service import ChatThreadService

router = APIRouter(prefix="/api/v1/chat-threads", tags=['会话管理的接口'])


# 1. 抽取公共函数，生成 service
def get_chat_thread_service(session: AsyncSession = Depends(get_session)):
    """获取数据库会话服务"""

    return ChatThreadService(session)


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
