"""用户画像模块数据库 CRUD 操作。"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.user_profile.models import UserProfile, UserProfileEvent


class UserProfileRepository:
    """用户画像仓库。"""

    def __init__(self, session: AsyncSession):
        """初始化用户画像仓库。

        Args:
            session: 数据库异步会话。
        """
        self.session = session

    async def get(
            self,
            user_id: int,
            *,
            for_update: bool = False,
    ) -> UserProfile | None:
        """按用户 ID 查询画像。

        Args:
            user_id: 用户 ID。
            for_update: 是否加行级锁，合并写入时使用，避免并发覆盖。

        Returns:
            画像记录；不存在时返回 None。
        """
        stmt = select(UserProfile).where(UserProfile.user_id == user_id)
        if for_update:
            stmt = stmt.with_for_update()
        return await self.session.scalar(stmt)

    async def add_event(self, event: UserProfileEvent) -> None:
        """新增一条画像变更事件。

        Args:
            event: 待新增的事件对象。
        """
        self.session.add(event)

    async def list_events(
            self,
            user_id: int,
            limit: int = 20,
    ) -> list[UserProfileEvent]:
        """按用户 ID 查询最近的画像变更事件。

        Args:
            user_id: 用户 ID。
            limit: 最多返回的事件数量。

        Returns:
            按时间倒序排列的事件列表。
        """
        stmt = (
            select(UserProfileEvent)
            .where(UserProfileEvent.user_id == user_id)
            .order_by(UserProfileEvent.created_at.desc())
            .limit(limit)
        )
        result = await self.session.scalars(stmt)
        return list(result.all())

    async def delete_events(self, user_id: int) -> None:
        """删除用户的所有画像变更事件。

        Args:
            user_id: 用户 ID。
        """
        await self.session.execute(
            delete(UserProfileEvent).where(UserProfileEvent.user_id == user_id)
        )
