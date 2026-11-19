from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_threads.models import ChatThread

# 只做增删改查
class ChatThreadRepository:
    """会话仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, chat_thread: ChatThread) -> None:
        """添加会话"""
        await self.session.add(chat_thread)