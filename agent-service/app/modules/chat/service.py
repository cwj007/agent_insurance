"""聊天模块服务。

对话执行交给 runner 在后台完成，SSE 只订阅运行事件；
连接断开不会中断运行，重连时按事件序号继续读取。
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import UUID

from fastapi.sse import ServerSentEvent
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.harness import (
    cancel_run as cancel_agent_run,
    replay_run as replay_agent_run,
    retry_run as retry_agent_run,
    start_run,
    subscribe_run_events,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.modules.agent_run.exceptions import AgentRunNotFound
from app.modules.agent_run.repository import AgentRunRepository
from app.modules.agent_run.schemas import AgentRunResponse
from app.modules.chat.schema import ChatRequest
from app.modules.chat_threads.exceptions import ChatThreadNotFound
from app.modules.chat_threads.repository import ChatThreadRepository
from app.modules.run_event.repository import RunEventRepository

logger = get_logger(__name__)

# 每次 SSE 推送的字符数；越小越接近逐字输出
STREAM_CHUNK_SIZE = settings.chat.stream_chunk_size

# 每个分片之间的间隔，以及单轮回答的总延迟上限
STREAM_CHUNK_DELAY_SECONDS = settings.chat.stream_chunk_delay_seconds
STREAM_MAX_TOTAL_DELAY_SECONDS = settings.chat.stream_max_total_delay_seconds


def _chunk_text(text: str, size: int = STREAM_CHUNK_SIZE) -> Iterator[str]:
    """把文本切成小块，保持 SSE 的分片推送体验。

    Args:
        text: 已经过输出校验的最终回答。
        size: 每块的字符数。

    Yields:
        文本片段。
    """
    for index in range(0, len(text), size):
        yield text[index:index + size]


def _json_event(payload: dict[str, Any]) -> str:
    """把 SSE 事件载荷序列化为 JSON。

    Args:
        payload: 事件载荷。

    Returns:
        JSON 字符串。
    """
    return json.dumps(payload, ensure_ascii=False, default=str)


async def _stream_answer_events(answer: str) -> AsyncIterator[ServerSentEvent]:
    """把回答切成小块，按固定节奏推送。

    Args:
        answer: 待推送的回答文本。

    Yields:
        message 类型的 SSE 事件。
    """
    chunk_size = STREAM_CHUNK_SIZE
    if len(answer) > 1200:
        # 长回答适当加大分片，避免产生过多 SSE 事件
        chunk_size = max(STREAM_CHUNK_SIZE, len(answer) // 600)

    chunks = list(_chunk_text(answer, chunk_size))
    chunk_delay = (
        min(
            STREAM_CHUNK_DELAY_SECONDS,
            STREAM_MAX_TOTAL_DELAY_SECONDS / len(chunks),
        )
        if chunks
        else 0
    )
    for chunk in chunks:
        yield ServerSentEvent(data=chunk, event="message")
        await asyncio.sleep(chunk_delay)


def _parse_after(after: str | None) -> tuple[str | None, int]:
    """解析 Last-Event-ID，格式为 run_id:seq。

    Args:
        after: 客户端最后收到的事件 ID。

    Returns:
        (run_id, seq)；无法解析时 run_id 为 None，seq 为 -1。
    """
    if not after:
        return None, -1

    run_id, _, seq_text = after.partition(":")
    if not run_id:
        return None, -1

    try:
        return run_id, int(seq_text) if seq_text else -1
    except ValueError:
        return run_id, -1


class ChatService:
    """聊天服务。"""

    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        """初始化聊天服务。

        Args:
            session: 数据库异步会话。
            agent: 编译后的 LangGraph agent 实例。
        """
        self.session = session
        self.agent = agent
        self.repository = ChatThreadRepository(session)
        self.run_event_repository = RunEventRepository(session)
        self.run_repository = AgentRunRepository(session)

    async def _require_thread(self, user_id: int | None, thread_id: UUID) -> None:
        """校验会话存在且属于当前用户。

        Args:
            user_id: 用户 ID。
            thread_id: 会话 ID。

        Raises:
            ChatThreadNotFound: 会话不存在或不属于当前用户时抛出。
        """
        chat_thread = await self.repository.required_owner(user_id, thread_id)
        if not chat_thread:
            raise ChatThreadNotFound()

    async def replay_events(
            self,
            user_id: int,
            thread_id: UUID,
            after: str | None = None,
    ) -> list[dict[str, Any]]:
        """按 Last-Event-ID 回放运行事件（JSON 形式）。

        Args:
            user_id: 当前用户 ID。
            thread_id: 会话 ID。
            after: 客户端最后收到的事件 ID，格式为 run_id:seq；
                为空时返回该会话最近一次运行的全部事件。

        Returns:
            事件列表，按 seq 升序排列。

        Raises:
            ChatThreadNotFound: 会话不存在或不属于当前用户时抛出。
        """
        await self._require_thread(user_id, thread_id)

        run_id, after_seq = _parse_after(after)
        if run_id is None:
            run_id = await self.run_event_repository.latest_run_id(str(thread_id))
        if run_id is None:
            return []

        events = await self.run_event_repository.list_after(run_id, after_seq)
        return [
            {
                "id": f"{run_id}:{event.seq}",
                "run_id": run_id,
                "seq": event.seq,
                "type": event.type,
                "payload": event.payload,
                "created_at": (
                    event.created_at.isoformat() if event.created_at else None
                ),
            }
            for event in events
        ]

    async def _stream_run(
            self,
            run_id: str,
            after_seq: int = -1,
    ) -> AsyncIterator[ServerSentEvent]:
        """订阅运行事件并转换成 SSE。

        Args:
            run_id: 运行 ID。
            after_seq: 只订阅序号大于该值的事件。

        Yields:
            SSE 事件：run_started、工具事件、message、replace、handoff、done。
        """
        streamed_answer: str | None = None

        async for event in subscribe_run_events(run_id, after_seq):
            event_id = f"{run_id}:{event.seq}"

            if event.type == "run_started":
                yield ServerSentEvent(
                    data=_json_event({"run_id": run_id}),
                    event="run_started",
                    id=event_id,
                )

            elif event.type in (
                    "tool_call_started",
                    "tool_call_completed",
                    "tool_call_failed",
            ):
                yield ServerSentEvent(
                    data=_json_event(event.payload),
                    event=event.type,
                    id=event_id,
                )

            # 工作流节点完成：先把原始回答推给前端，output_guard 仍在后台执行
            elif event.type == "answer_streamed":
                answer = str(event.payload.get("answer") or "")
                if answer:
                    streamed_answer = answer
                    async for sse in _stream_answer_events(answer):
                        yield sse

            elif event.type == "answer_completed":
                answer = str(event.payload.get("answer") or "")
                if streamed_answer is None:
                    # 没有捕获到工作流节点输出（老运行或异常路径），直接推最终回答
                    async for sse in _stream_answer_events(answer):
                        yield sse

                elif answer.strip() != streamed_answer.strip():
                    # output_guard 替换了原始回答，通知前端覆盖已展示内容
                    yield ServerSentEvent(
                        data=_json_event({"content": answer}),
                        event="replace",
                        id=event_id,
                    )

            elif event.type == "handoff":
                yield ServerSentEvent(
                    data="handoff",
                    event="handoff",
                    id=event_id,
                )

            elif event.type in ("run_completed", "run_failed", "run_cancelled"):
                yield ServerSentEvent(
                    data="end",
                    event="done",
                    id=event_id,
                )

    async def chat_stream(
            self,
            user_id: int | None,
            request: ChatRequest,
    ) -> AsyncIterator[ServerSentEvent]:
        """创建一次运行并订阅其事件。

        Args:
            user_id: 当前用户 ID。
            request: 聊天请求体，包含会话 ID 和用户消息。

        Yields:
            SSE 事件流。

        Raises:
            ChatThreadNotFound: 会话不存在或不属于当前用户时抛出。
        """
        await self._require_thread(user_id, request.thread_id)

        # 1. 创建后台运行，HTTP 连接断开不会中断执行
        run_id = await start_run(
            self.agent,
            user_id=user_id,
            thread_id=str(request.thread_id),
            message=request.message,
        )

        # 2. 订阅运行事件并推送
        async for sse in self._stream_run(run_id):
            yield sse

    async def stream_run_events(
            self,
            user_id: int | None,
            run_id: str,
            after: str | None = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """订阅指定运行的事件，用于断线重连。

        Args:
            user_id: 当前用户 ID。
            run_id: 运行 ID。
            after: 最后收到的事件 ID，格式为 run_id:seq。

        Yields:
            SSE 事件流。

        Raises:
            AgentRunNotFound: 运行不存在或不属于当前用户时抛出。
        """
        if user_id is None:
            raise AgentRunNotFound()

        run = await self.run_repository.get_for_user(run_id, user_id)
        if run is None:
            raise AgentRunNotFound()

        _, after_seq = _parse_after(after)
        async for sse in self._stream_run(run_id, after_seq):
            yield sse

    async def get_run(self, user_id: int | None, run_id: str) -> AgentRunResponse:
        """查询运行状态。

        Args:
            user_id: 当前用户 ID。
            run_id: 运行 ID。

        Returns:
            运行详情。

        Raises:
            AgentRunNotFound: 运行不存在或不属于当前用户时抛出。
        """
        if user_id is None:
            raise AgentRunNotFound()

        run = await self.run_repository.get_for_user(run_id, user_id)
        if run is None:
            raise AgentRunNotFound()

        return AgentRunResponse.model_validate(run)

    async def cancel_run(self, user_id: int | None, run_id: str) -> bool:
        """取消运行。

        Args:
            user_id: 当前用户 ID。
            run_id: 运行 ID。

        Returns:
            是否成功取消活动任务。

        Raises:
            AgentRunNotFound: 运行不存在或不属于当前用户时抛出。
        """
        if user_id is None:
            raise AgentRunNotFound()

        run = await self.run_repository.get_for_user(run_id, user_id)
        if run is None:
            raise AgentRunNotFound()

        return await cancel_agent_run(run_id)

    async def replay_run(self, user_id: int | None, run_id: str) -> str:
        """重放一次已结束的运行。

        Args:
            user_id: 当前用户 ID。
            run_id: 原运行 ID。

        Returns:
            新运行 ID。

        Raises:
            AgentRunNotFound: 运行不存在或不属于当前用户时抛出。
            AgentRunConflict: 运行尚未结束或状态不允许重放时抛出。
        """
        if user_id is None:
            raise AgentRunNotFound()

        return await replay_agent_run(self.agent, run_id, user_id)

    async def retry_run(self, user_id: int | None, run_id: str) -> str:
        """重试失败或取消的运行。

        Args:
            user_id: 当前用户 ID。
            run_id: 原运行 ID。

        Returns:
            新运行 ID。

        Raises:
            AgentRunNotFound: 运行不存在或不属于当前用户时抛出。
            AgentRunConflict: 运行状态不是失败或取消时抛出。
        """
        if user_id is None:
            raise AgentRunNotFound()

        return await retry_agent_run(self.agent, run_id, user_id)
