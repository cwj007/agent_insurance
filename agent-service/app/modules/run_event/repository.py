"""运行事件模块数据库 CRUD 操作。"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.run_event.models import RunEvent


class RunEventRepository:
    """运行事件仓库。"""

    def __init__(self, session: AsyncSession):
        """初始化运行事件仓库。

        Args:
            session: 数据库异步会话。
        """
        self.session = session

    async def add(
            self,
            *,
            run_id: str,
            thread_id: str,
            seq: int,
            event_type: str,
            payload: dict[str, Any],
    ) -> RunEvent:
        """新增一条运行事件。

        Args:
            run_id: 运行 ID。
            thread_id: 会话 ID。
            seq: 运行内事件序号。
            event_type: 事件类型。
            payload: 事件载荷。

        Returns:
            新建的事件对象。
        """
        event = RunEvent(
            run_id=run_id,
            thread_id=thread_id,
            seq=seq,
            type=event_type,
            payload=payload,
        )
        self.session.add(event)
        await self.session.flush()
        return event

    async def list_after(self, run_id: str, after_seq: int = -1) -> list[RunEvent]:
        """查询指定运行中序号大于 after_seq 的事件。

        Args:
            run_id: 运行 ID。
            after_seq: 只返回序号大于该值的事件；-1 表示返回全部。

        Returns:
            按序号升序排列的事件列表。
        """
        stmt = (
            select(RunEvent)
            .where(RunEvent.run_id == run_id, RunEvent.seq > after_seq)
            .order_by(RunEvent.seq.asc())
        )
        result = await self.session.scalars(stmt)
        return list(result.all())

    async def latest_run_id(self, thread_id: str) -> str | None:
        """查询会话最近一次运行的 ID。

        Args:
            thread_id: 会话 ID。

        Returns:
            最近一次运行的 run_id；没有事件时返回 None。
        """
        stmt = (
            select(RunEvent.run_id)
            .where(RunEvent.thread_id == thread_id)
            .order_by(RunEvent.id.desc())
            .limit(1)
        )
        return await self.session.scalar(stmt)
