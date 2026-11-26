"""运行事件写入与订阅。"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from app.harness.state import (
    EVENT_POLL_INTERVAL_SECONDS,
    EVENT_SUBSCRIBE_TIMEOUT_SECONDS,
    TERMINAL_STATUSES,
)
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.agent_run.repository import AgentRunRepository
from app.modules.run_event.models import RunEvent
from app.modules.run_event.repository import RunEventRepository

logger = get_logger(__name__)


async def _append_event(
        run_id: str,
        thread_id: str,
        seq: int,
        event_type: str,
        payload: dict[str, Any],
) -> None:
    """写入一条运行事件。

    Args:
        run_id: 运行 ID。
        thread_id: 会话 ID。
        seq: 运行内事件序号。
        event_type: 事件类型。
        payload: 事件载荷。
    """
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                session.add(
                    RunEvent(
                        run_id=run_id,
                        thread_id=thread_id,
                        seq=seq,
                        type=event_type,
                        payload=payload,
                    )
                )
    except Exception as exc:
        logger.warning(f"运行事件写入失败: run_id={run_id}, type={event_type}, error={exc}")


async def _update_run_status(
        run_id: str,
        status: str,
        *,
        answer: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        finished: bool = False,
) -> None:
    """更新运行主记录状态。

    Args:
        run_id: 运行 ID。
        status: 新状态。
        answer: 最终回答。
        error_code: 失败错误码。
        error_message: 失败说明。
        finished: 是否已结束。
    """
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                await AgentRunRepository(session).update_status(
                    run_id,
                    status,
                    answer=answer,
                    error_code=error_code,
                    error_message=error_message,
                    finished=finished,
                )
    except Exception as exc:
        logger.warning(f"运行状态更新失败: run_id={run_id}, error={exc}")


async def subscribe_run_events(
        run_id: str,
        after_seq: int = -1,
        *,
        poll_interval: float = EVENT_POLL_INTERVAL_SECONDS,
        timeout: float = EVENT_SUBSCRIBE_TIMEOUT_SECONDS,
) -> AsyncIterator[RunEvent]:
    """按序号轮询订阅运行事件，直到运行结束。

    Args:
        run_id: 运行 ID。
        after_seq: 只返回序号大于该值的事件。
        poll_interval: 轮询间隔秒数。
        timeout: 最长订阅秒数，避免僵死运行导致无限等待。

    Yields:
        按 seq 升序排列的事件。
    """
    last_seq = after_seq
    deadline = time.monotonic() + timeout

    while True:
        async with AsyncSessionFactory() as session:
            events = await RunEventRepository(session).list_after(run_id, last_seq)
            run = await AgentRunRepository(session).get(run_id)

        for event in events:
            last_seq = event.seq
            yield event

        if run is not None and run.status in TERMINAL_STATUSES and not events:
            break

        if time.monotonic() > deadline:
            logger.warning(f"运行事件订阅超时: run_id={run_id}")
            break

        await asyncio.sleep(poll_interval)
