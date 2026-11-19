from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_threads.models import ChatThread

# 只做增删改查
class ChatThreadRepository:
    """会话仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, chat_thread: ChatThread) -> None:
        """添加会话"""

        self.session.add(chat_thread)

    async def list_by_user_id(self, user_id: int):
        """根据用户ID查询会话列表"""

        return await self.session.scalars(
            select(ChatThread)
            .where(ChatThread.user_id == user_id)
        ).all()