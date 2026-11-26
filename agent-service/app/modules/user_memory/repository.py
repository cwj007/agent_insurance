"""用户情节记忆数据库 CRUD 操作。"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.user_memory.models import UserMemory


class UserMemoryRepository:
    """用户情节记忆仓库。"""

    def __init__(self, session: AsyncSession):
        """初始化记忆仓库。

        Args:
            session: 数据库异步会话。
        """
        self.session = session

    async def add(self, memory: UserMemory) -> UserMemory:
        """新增一条记忆。

        Args:
            memory: 待新增的记忆对象。

        Returns:
            新增的记忆对象。
        """
        self.session.add(memory)
        await self.session.flush()
        return memory

    async def list_recent(self, user_id: int, limit: int = 20) -> list[UserMemory]:
        """查询用户最近的记忆。

        Args:
            user_id: 用户 ID。
            limit: 最多返回条数。

        Returns:
            按时间倒序排列的记忆列表。
        """
        stmt = (
            select(UserMemory)
            .where(UserMemory.user_id == user_id)
            .order_by(UserMemory.created_at.desc())
            .limit(limit)
        )
        result = await self.session.scalars(stmt)
        return list(result.all())

    async def list_recent_contents(
            self,
            user_id: int,
            limit: int = 50,
    ) -> set[str]:
        """查询用户最近记忆的内容集合，用于去重。

        Args:
            user_id: 用户 ID。
            limit: 最多检查条数。

        Returns:
            最近记忆的内容集合。
        """
        memories = await self.list_recent(user_id, limit)
        return {memory.content for memory in memories}

    async def delete_by_user(self, user_id: int) -> None:
        """删除用户全部记忆。

        Args:
            user_id: 用户 ID。
        """
        await self.session.execute(
            delete(UserMemory).where(UserMemory.user_id == user_id)
        )
