"""聊天模块服务"""
from fastapi.sse import ServerSentEvent
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat.schema import ChatRequest
from app.modules.chat_threads.exceptions import ChatThreadNotFound
from app.modules.chat_threads.repository import ChatThreadRepository


class ChatService:
    """聊天模块服务"""

    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.agent = agent
        self.repository = ChatThreadRepository(session)

    async def chat_stream(self, user_id: int, request: ChatRequest, ):
        """实现 Agent 流式对话"""

        # 校验：判断当前 thread_id 是否属于当前用户
        chat_thread = await self.repository.required_owner(user_id, request.thread_id)
        if not chat_thread:
            raise ChatThreadNotFound()

        # 1. 组织消息
        _input = {"messages": [{"role": "user", "content": request.message}]}

        # 配置 checkpoint
        config = {"configurable": {"thread_id": request.thread_id}}

        # 2. 流式输出模型 token
        stream = await self.agent.astream_events(
            _input,
            config,
            version="v3"
        )

        # 3. 返回结果
        async for message in stream.messages:
            async for chunk in message.text:
                yield ServerSentEvent(data=chunk, event="message") # 这里返回给的是后端接口

        yield ServerSentEvent(data="end", event="done")
