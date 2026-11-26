"""Agent 运行管理数据库模型。"""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base
from app.core.logging import get_logger
from app.infra.database import engine

logger = get_logger(__name__)


class AgentRun(Base):
    """一次 Agent 运行的主记录。

    运行与 SSE 连接解耦：HTTP 请求只负责创建运行和订阅事件，
    实际执行在后台任务中完成，断线后仍可继续跑完。
    """

    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="运行 ID",
    )
    thread_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="会话 ID",
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger,
        index=True,
        nullable=True,
        comment="用户 ID",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        index=True,
        comment="状态：pending / running / completed / failed / cancelled",
    )
    input_message: Mapped[str] = mapped_column(
        Text,
        comment="触发本次运行的用户消息",
    )
    answer: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="最终回答",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="失败错误码",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败说明",
    )
    replay_of_run_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="重放来源运行 ID",
    )
    worker_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="执行该运行的 worker 标识",
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="worker 最近心跳时间",
    )
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        comment="是否请求取消",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="开始时间",
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="结束时间",
    )


async def init_agent_run_tables() -> None:
    """创建 Agent 运行主表，并补齐多 worker 需要的列。

    使用 checkfirst 保证幂等，只建本模块的表；
    已有表不会由 create_all 补列，因此额外执行轻量 ALTER。
    """
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[AgentRun.__table__],
        )
        # 兼容已经存在的 agent_runs 表
        await conn.execute(text(
            "ALTER TABLE agent_runs "
            "ADD COLUMN IF NOT EXISTS worker_id VARCHAR(64)"
        ))
        await conn.execute(text(
            "ALTER TABLE agent_runs "
            "ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ"
        ))
        await conn.execute(text(
            "ALTER TABLE agent_runs "
            "ADD COLUMN IF NOT EXISTS cancel_requested "
            "BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_agent_runs_worker_id "
            "ON agent_runs (worker_id)"
        ))
    logger.info("Agent 运行表初始化完成~☑️")
