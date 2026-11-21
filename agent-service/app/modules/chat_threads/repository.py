"""会话管理模块数据库 CRUD 操作"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_threads.models import ChatThread


# 只做增删改查
class ChatThreadRepository:
    """会话仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, chat_thread: ChatThread):
        """添加会话"""

        self.session.add(chat_thread)

    async def list_by_user(self, user_id: int):
        """根据用户ID查询会话列表"""

        stmt = (
            select(ChatThread)
            .where(ChatThread.user_id == user_id)
            .order_by(ChatThread.updated_at.desc(), ChatThread.created_at.desc())
        )
        result = await self.session.scalars(stmt)
        return result.all()

    async def required_owner(self, user_id: int, thread_id: UUID) -> ChatThread | None:
        """查询属于该用户的会话，不存在时返回 None"""

        return await self.session.scalar(
            select(ChatThread)
            .where(ChatThread.id == thread_id, ChatThread.user_id == user_id)
        )