"""Agent 运行管理数据库 CRUD 操作。"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.agent_run.models import AgentRun


class AgentRunRepository:
    """Agent 运行仓库。"""

    def __init__(self, session: AsyncSession):
        """初始化运行仓库。

        Args:
            session: 数据库异步会话。
        """
        self.session = session

    async def add(self, run: AgentRun) -> AgentRun:
        """新增一次运行记录。

        Args:
            run: 待新增的运行对象。

        Returns:
            新增的运行对象。
        """
        self.session.add(run)
        await self.session.flush()
        return run

    async def get(self, run_id: str) -> AgentRun | None:
        """按运行 ID 查询运行记录。

        Args:
            run_id: 运行 ID。

        Returns:
            运行记录；不存在时返回 None。
        """
        return await self.session.scalar(
            select(AgentRun).where(AgentRun.run_id == run_id)
        )

    async def get_for_user(self, run_id: str, user_id: int) -> AgentRun | None:
        """查询属于指定用户的运行记录。

        Args:
            run_id: 运行 ID。
            user_id: 用户 ID。

        Returns:
            匹配的运行记录；不存在或不属于该用户时返回 None。
        """
        return await self.session.scalar(
            select(AgentRun).where(
                AgentRun.run_id == run_id,
                AgentRun.user_id == user_id,
            )
        )

    async def update_status(
            self,
            run_id: str,
            status: str,
            *,
            answer: str | None = None,
            error_code: str | None = None,
            error_message: str | None = None,
            finished: bool = False,
    ) -> None:
        """更新运行状态。

        Args:
            run_id: 运行 ID。
            status: 新状态。
            answer: 最终回答，仅成功时传入。
            error_code: 失败错误码。
            error_message: 失败说明。
            finished: 是否已结束，True 时写入 ended_at。
        """
        values: dict = {
            "status": status,
            "answer": answer,
            "error_code": error_code,
            "error_message": error_message,
        }
        if finished:
            values["ended_at"] = datetime.now(UTC)

        await self.session.execute(
            update(AgentRun)
            .where(AgentRun.run_id == run_id)
            .values(**values)
        )

    async def update_heartbeat(self, run_id: str, worker_id: str) -> None:
        """更新运行心跳。

        Args:
            run_id: 运行 ID。
            worker_id: 当前 worker 标识。
        """
        await self.session.execute(
            update(AgentRun)
            .where(
                AgentRun.run_id == run_id,
                AgentRun.worker_id == worker_id,
            )
            .values(heartbeat_at=datetime.now(UTC))
        )

    async def request_cancel(self, run_id: str) -> bool:
        """写入取消请求标记。

        Args:
            run_id: 运行 ID。

        Returns:
            运行处于可取消状态时返回 True。
        """
        result = await self.session.execute(
            update(AgentRun)
            .where(
                AgentRun.run_id == run_id,
                AgentRun.status.in_(("pending", "running")),
            )
            .values(cancel_requested=True)
        )
        return (result.rowcount or 0) > 0

    async def is_cancel_requested(self, run_id: str) -> bool:
        """查询运行是否收到取消请求。

        Args:
            run_id: 运行 ID。

        Returns:
            收到取消请求时返回 True。
        """
        value = await self.session.scalar(
            select(AgentRun.cancel_requested).where(
                AgentRun.run_id == run_id
            )
        )
        return bool(value)

    async def mark_stale_runs_failed(self, stale_seconds: float = 60.0) -> int:
        """把心跳超时的运行中记录标记为失败。

        多 worker 部署时只处理心跳过期的运行，
        避免误伤其他 worker 正在执行的任务。

        Args:
            stale_seconds: 心跳超过该秒数视为 worker 失联。

        Returns:
            被标记的记录数量。
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=max(stale_seconds, 1.0))
        result = await self.session.execute(
            update(AgentRun)
            .where(
                AgentRun.status.in_(("pending", "running")),
                (AgentRun.heartbeat_at.is_(None))
                | (AgentRun.heartbeat_at < cutoff),
            )
            .values(
                status="failed",
                error_code="WORKER_LOST",
                error_message="worker 心跳超时，运行已中断",
                ended_at=datetime.now(UTC),
            )
        )
        return result.rowcount or 0
