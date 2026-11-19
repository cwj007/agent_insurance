from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_threads.models import ChatThread
from app.modules.chat_threads.repository import ChatThreadRepository


class ChatThreadService:
    """会话服务"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = ChatThreadRepository(session)

    async def add(self, user_id: int, title: str) -> None:
        """添加会话"""

        # 开启事务, 完成后自动 commit，出现问题，rollback
        async with self.session.begin():
            # 创建对象
            chat_thread = ChatThread(user_id=user_id, title=title)
            # 添加到数据库
            await self.repository.add(chat_thread)
            return chat_thread
