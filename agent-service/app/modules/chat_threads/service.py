"""会话管理模块服务"""

from uuid import UUID

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_threads.exceptions import ChatThreadNotFound
from app.modules.chat_threads.models import ChatThread
from app.modules.chat_threads.repository import ChatThreadRepository
from app.modules.chat_threads.schemas import ChatHistoryResponse, ChatMessageResponse


class ChatThreadService:
    """会话服务"""

    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.session = session
        self.agent = agent
        self.repository = ChatThreadRepository(session)

    async def add(self, user_id: int, title: str) -> ChatThread:
        """新增会话"""

        # 用上下文管理，开启事务，运行完成，自动commit，出现异常，自动rollback
        async with self.session.begin():
            # 创建对象，对象 映射 到数据库 一条数据
            chat_thread = ChatThread(user_id=user_id, title=title)
            # 新增对象，就是插入一条数据
            await self.repository.add(chat_thread)

            return chat_thread

    async def list_by_user(self, user_id: int) -> list[ChatThread]:
        """根据用户ID查询会话列表"""

        return await self.repository.list_by_user(user_id)

    async def rename(self, user_id: int, thread_id: UUID, title: str) -> ChatThread:
        """根据用户 ID 重命名会话"""

        async with self.session.begin():
            # 1. 查询会话，校验用户
            chat_thread = await self.repository.required_owner(user_id, thread_id)

            # 校验会话是否存在
            if not chat_thread:
                raise ChatThreadNotFound()

            # 2. 修改标题, 这时候会触发数据库更新
            chat_thread.title = title

            # 3. 刷新数据
            await self.session.flush()
            await self.session.refresh(chat_thread)

            # 4. 返回更新后的会话对象
            return chat_thread

    # agent 注入
    async def get_chat_history(self, user_id: int, thread_id: str):  # 使用 agent 获取
        """根据会话ID查询会话历史消息"""

        # 1. 查询会话，校验用户
        chat_thread = await self.repository.required_owner(user_id, thread_id)

        # 2. 校验会话是否存在
        if not chat_thread:
            raise ChatThreadNotFound()

        # 3. 使用 agent 获取对话记录
        config = {"configurable": {"thread_id": thread_id}}

        state_snapshot = await self.agent.aget_state(config)

        # 4. 封装历史消息
        messages = []
        history = state_snapshot.values

        for message in history.get('messages', []):
            # 过滤出 HumanMessage 和 AIMessage 类型的消息
            if isinstance(message, HumanMessage):
                role = 'user'
            elif isinstance(message, AIMessage):
                role = 'assistant'
            else:
                continue

            if message.text:
                messages.append(ChatMessageResponse(role=role, content=message.text)) # text 内部方法，可以是列表也可以是字符串

        # 封装好的历史消息
        return ChatHistoryResponse(thread_id=thread_id, messages=messages)