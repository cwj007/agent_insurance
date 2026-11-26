"""用户画像模块路由。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.user_profile.schemas import (
    ProfileCorrectionRequest,
    UserProfileResponse,
)
from app.modules.user_profile.service import UserProfileService

router = APIRouter(prefix="/api/v1/user-profile", tags=["用户画像"])


def get_user_profile_service(
        session: AsyncSession = Depends(get_session),
) -> UserProfileService:
    """获取用户画像服务实例。

    Args:
        session: 数据库异步会话，由 FastAPI 依赖注入。

    Returns:
        绑定当前请求会话的 UserProfileService 实例。
    """
    return UserProfileService(session)


@router.get("", response_model=UserProfileResponse, summary="查询用户画像")
async def get_user_profile(
        user_id: Annotated[int, Header(alias="x-user-id")],
        service: UserProfileService = Depends(get_user_profile_service),
) -> UserProfileResponse:
    """查询当前用户的画像。

    Args:
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 用户画像服务实例。

    Returns:
        当前用户的画像。
    """
    return await service.get_response(user_id)


@router.patch("", response_model=UserProfileResponse, summary="更正用户画像")
async def correct_user_profile(
        request: ProfileCorrectionRequest,
        user_id: Annotated[int, Header(alias="x-user-id")],
        service: UserProfileService = Depends(get_user_profile_service),
) -> UserProfileResponse:
    """由用户手动更正画像字段。

    Args:
        request: 更正请求体，包含字段名到新值的映射。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 用户画像服务实例。

    Returns:
        更新后的画像。
    """
    return await service.apply_correction(user_id, request.fields)


@router.delete("", status_code=204, summary="删除用户画像")
async def delete_user_profile(
        user_id: Annotated[int, Header(alias="x-user-id")],
        service: UserProfileService = Depends(get_user_profile_service),
) -> None:
    """删除当前用户的画像及其变更记录。

    Args:
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 用户画像服务实例。
    """
    await service.delete(user_id)
