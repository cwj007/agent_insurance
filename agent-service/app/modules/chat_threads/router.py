"""会话管理模块路由。"""

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
        request: Request,  # 注入 agent
        session: AsyncSession = Depends(get_session)
):
    """获取数据库会话服务。

    Args:
        request: FastAPI 请求对象，用于从 app.state 中获取 agent。
        session: 数据库异步会话，由 FastAPI 依赖注入。

    Returns:
        绑定当前请求会话与 agent 的 ChatThreadService 实例。
    """
    # 在初始化时，就获取 agent
    agent = request.app.state.agent

    return ChatThreadService(session, agent)


@router.post("", response_model=ChatThreadResponse, summary="添加会话")
async def create_chat_thread(
        request: ChatThreadCreate,
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """添加会话。

    Args:
        request: 创建会话请求体，包含会话标题。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 会话服务实例，由 FastAPI 依赖注入。

    Returns:
        新建的会话信息。
    """
    return await service.add(user_id, request.title)


@router.get("", response_model=list[ChatThreadResponse], summary="根据用户ID查询会话列表")
async def list_by_threads(
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """根据用户 ID 查询会话列表。

    Args:
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 会话服务实例，由 FastAPI 依赖注入。

    Returns:
        该用户的会话列表。
    """
    return await service.list_by_user(user_id)


@router.delete("/{thread_id}", status_code=204, summary="删除会话")
async def remove(
        thread_id: str,
        user_id: Annotated[int, Header(alias="x-user-id")],
        service: ChatThreadService = Depends(get_chat_thread_service),
):
    """删除会话。

    Args:
        thread_id: 会话 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 会话服务实例，由 FastAPI 依赖注入。
    """
    await service.remove(user_id, thread_id)


@router.patch("/{thread_id}", response_model=ChatThreadResponse, summary="重命名会话")
async def rename_chat_thread(
        thread_id: UUID,  # 路径参数
        request: ChatThreadCreate,
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """重命名会话。

    Args:
        thread_id: 会话 ID（路径参数）。
        request: 重命名请求体，包含新的会话标题。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 会话服务实例，由 FastAPI 依赖注入。

    Returns:
        更新后的会话信息。
    """
    return await service.rename(user_id, thread_id, request.title)


@router.get("/{thread_id}/messages", response_model=ChatHistoryResponse, summary="查询会话历史消息")
async def get_chat_history(
        thread_id: UUID,  # 路径参数
        user_id: Annotated[int | None, Header(alias="x-user-id")],  # 不区分大小写
        service: ChatThreadService = Depends(get_chat_thread_service)

):
    """查询当前用户的指定会话的历史消息。

    Args:
        thread_id: 会话 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 会话服务实例，由 FastAPI 依赖注入。

    Returns:
        指定会话的历史消息列表。
    """
    return await service.get_chat_history(user_id, thread_id)
