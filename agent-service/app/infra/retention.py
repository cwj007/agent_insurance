"""过期数据清理：运行事件、工具调用记录和运行主记录。"""

import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete

from app.core.config import settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.agent_run.models import AgentRun
from app.modules.run_event.models import RunEvent
from app.modules.tool_invocation.models import ToolInvocation
from app.modules.user_memory.models import UserMemory

logger = get_logger(__name__)


async def cleanup_expired_records() -> dict[str, int]:
    """删除超过保留期的数据。

    Returns:
        各表删除的行数。
    """
    now = datetime.now(UTC)
    counts: dict[str, int] = {}

    async with AsyncSessionFactory() as session:
        async with session.begin():
            # 1. 运行事件
            cutoff = now - timedelta(days=settings.retention.run_event_days)
            result = await session.execute(
                delete(RunEvent).where(RunEvent.created_at < cutoff)
            )
            counts["run_events"] = result.rowcount or 0

            # 2. 工具调用记录
            cutoff = now - timedelta(days=settings.retention.tool_invocation_days)
            result = await session.execute(
                delete(ToolInvocation).where(ToolInvocation.created_at < cutoff)
            )
            counts["tool_invocations"] = result.rowcount or 0

            # 3. 运行主记录，只清理已结束的运行
            cutoff = now - timedelta(days=settings.retention.agent_run_days)
            result = await session.execute(
                delete(AgentRun).where(
                    AgentRun.ended_at.is_not(None),
                    AgentRun.ended_at < cutoff,
                )
            )
            counts["agent_runs"] = result.rowcount or 0

            # 4. 用户情节记忆
            cutoff = now - timedelta(days=settings.retention.user_memory_days)
            result = await session.execute(
                delete(UserMemory).where(UserMemory.created_at < cutoff)
            )
            counts["user_memories"] = result.rowcount or 0

    if any(counts.values()):
        logger.info(f"过期数据清理完成: {counts}")
    return counts


async def retention_worker(stop_event: asyncio.Event) -> None:
    """定时执行过期数据清理，直到收到停止信号。

    Args:
        stop_event: 停止信号。
    """
    interval_seconds = max(settings.retention.cleanup_interval_hours, 1) * 3600

    while not stop_event.is_set():
        try:
            await cleanup_expired_records()
        except Exception as exc:
            logger.warning(f"过期数据清理失败: {exc}")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass


def start_retention_worker() -> tuple[asyncio.Event, asyncio.Task]:
    """启动过期数据清理任务。

    Returns:
        (停止信号, 后台任务)。
    """
    stop_event = asyncio.Event()
    task = asyncio.create_task(
        retention_worker(stop_event),
        name="retention-worker",
    )
    logger.info("过期数据清理任务已启动~☑️")
    return stop_event, task


async def stop_retention_worker(
        stop_event: asyncio.Event,
        task: asyncio.Task,
        timeout: float = 5.0,
) -> None:
    """停止过期数据清理任务。

    Args:
        stop_event: 停止信号。
        task: 后台任务。
        timeout: 最长等待秒数。
    """
    stop_event.set()
    await asyncio.wait([task], timeout=timeout)
    logger.info("过期数据清理任务已停止")
