"""运行心跳、跨 worker 取消轮询与失联清理。"""

import asyncio
import time

from app.harness.state import WORKER_ID
from app.core.config import settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.agent_run.repository import AgentRunRepository

logger = get_logger(__name__)

# 失联运行清理任务
_stale_reaper_task: asyncio.Task | None = None


async def _update_run_heartbeat(run_id: str) -> None:
    """更新运行心跳。

    Args:
        run_id: 运行 ID。
    """
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                await AgentRunRepository(session).update_heartbeat(
                    run_id,
                    WORKER_ID,
                )
    except Exception as exc:
        logger.warning(f"运行心跳更新失败: run_id={run_id}, error={exc}")


async def _is_cancel_requested(run_id: str) -> bool:
    """查询运行是否收到跨 worker 取消请求。

    Args:
        run_id: 运行 ID。

    Returns:
        收到取消请求时返回 True。
    """
    try:
        async with AsyncSessionFactory() as session:
            return await AgentRunRepository(session).is_cancel_requested(run_id)
    except Exception as exc:
        logger.warning(f"查询取消请求失败: run_id={run_id}, error={exc}")
        return False


async def _run_heartbeat_loop(run_id: str, owner_task: asyncio.Task) -> None:
    """定时更新心跳并响应跨 worker 取消请求。

    Args:
        run_id: 运行 ID。
        owner_task: 执行该运行的任务，收到取消请求时取消它。
    """
    heartbeat_interval = max(settings.agent_run.heartbeat_interval_seconds, 1.0)
    cancel_interval = max(settings.agent_run.cancel_poll_interval_seconds, 0.5)
    next_heartbeat_at = 0.0

    while True:
        now = time.monotonic()
        if now >= next_heartbeat_at:
            await _update_run_heartbeat(run_id)
            next_heartbeat_at = time.monotonic() + heartbeat_interval

        if await _is_cancel_requested(run_id):
            logger.info(f"收到跨 worker 取消请求: run_id={run_id}")
            owner_task.cancel()
            return

        await asyncio.sleep(cancel_interval)


async def _stale_run_reaper_loop(interval_seconds: float) -> None:
    """定时把心跳超时的运行标记为失败。

    Args:
        interval_seconds: 扫描间隔秒数。
    """
    while True:
        try:
            async with AsyncSessionFactory() as session:
                async with session.begin():
                    count = await AgentRunRepository(session).mark_stale_runs_failed(
                        settings.agent_run.stale_run_seconds
                    )
            if count:
                logger.warning(f"已逐出 {count} 条心跳超时的运行")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"失联运行清理失败: {exc}")

        await asyncio.sleep(interval_seconds)


def start_stale_run_reaper() -> asyncio.Task:
    """启动失联运行清理任务。

    Returns:
        后台任务。
    """
    global _stale_reaper_task
    interval = max(settings.agent_run.stale_run_seconds / 2, 10.0)
    _stale_reaper_task = asyncio.create_task(
        _stale_run_reaper_loop(interval),
        name="agent-run-stale-reaper",
    )
    logger.info(f"失联运行清理任务已启动，间隔 {interval:.0f}s~☑️")
    return _stale_reaper_task


async def stop_stale_run_reaper(timeout: float = 5.0) -> None:
    """停止失联运行清理任务。

    Args:
        timeout: 最长等待秒数。
    """
    global _stale_reaper_task
    task = _stale_reaper_task
    _stale_reaper_task = None
    if task is None:
        return

    task.cancel()
    await asyncio.wait([task], timeout=timeout)
    logger.info("失联运行清理任务已停止")
