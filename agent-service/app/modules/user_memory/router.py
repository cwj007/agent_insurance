"""用户情节记忆路由。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.user_memory.schemas import UserMemoryResponse
from app.modules.user_memory.service import UserMemoryService

router = APIRouter(prefix="/api/v1/user-memories", tags=["用户记忆"])


def get_user_memory_service(
        session: AsyncSession = Depends(get_session),
) -> UserMemoryService:
    """获取用户记忆服务实例。

    Args:
        session: 数据库异步会话，由 FastAPI 依赖注入。

    Returns:
        绑定当前请求会话的 UserMemoryService 实例。
    """
    return UserMemoryService(session)


@router.get("", response_model=list[UserMemoryResponse], summary="查询用户记忆")
async def list_user_memories(
        user_id: Annotated[int, Header(alias="x-user-id")],
        limit: int = Query(default=50, ge=1, le=200),
        service: UserMemoryService = Depends(get_user_memory_service),
):
    """查询当前用户最近的情节记忆。

    Args:
        user_id: 请求头 x-user-id 中的用户 ID。
        limit: 最多返回条数。
        service: 用户记忆服务实例。

    Returns:
        按时间倒序排列的记忆列表。
    """
    return await service.list_for_user(user_id, limit)


@router.delete("", status_code=204, summary="清空用户记忆")
async def delete_user_memories(
        user_id: Annotated[int, Header(alias="x-user-id")],
        service: UserMemoryService = Depends(get_user_memory_service),
) -> None:
    """删除当前用户的全部情节记忆。

    Args:
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 用户记忆服务实例。
    """
    await service.delete_all(user_id)
