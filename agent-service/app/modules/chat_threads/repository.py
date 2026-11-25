"""会话管理模块数据库 CRUD 操作。"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_threads.models import ChatThread


# 只做增删改查
class ChatThreadRepository:
    """会话仓库。"""

    def __init__(self, session: AsyncSession):
        """初始化会话仓库。

        Args:
            session: 数据库异步会话。
        """
        self.session = session

    async def add(self, chat_thread: ChatThread):
        """添加会话。

        Args:
            chat_thread: 待新增的会话对象。
        """
        self.session.add(chat_thread)

    async def delete(self, chat_thread: ChatThread):
        """删除会话。

        Args:
            chat_thread: 待删除的会话对象。
        """
        # 删除会话
        await self.session.delete(chat_thread)

    async def list_by_user(self, user_id: int):
        """根据用户 ID 查询会话列表。

        Args:
            user_id: 用户 ID。

        Returns:
            该用户的会话列表，按更新时间和创建时间倒序排列。
        """
        stmt = (
            select(ChatThread)
            .where(ChatThread.user_id == user_id)
            .order_by(ChatThread.updated_at.desc(), ChatThread.created_at.desc())
        )
        result = await self.session.scalars(stmt)
        return result.all()

    async def required_owner(self, user_id: int, thread_id: str) -> ChatThread | None:
        """查询属于该用户的会话，不存在时返回 None。

        Args:
            user_id: 用户 ID。
            thread_id: 会话 ID。

        Returns:
            匹配的会话对象；不存在或不属于该用户时返回 None。
        """
        return await self.session.scalar(
            select(ChatThread)
            .where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
        )
