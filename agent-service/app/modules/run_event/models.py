"""运行事件模块数据库模型。"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base
from app.core.logging import get_logger
from app.infra.database import engine

logger = get_logger(__name__)


class RunEvent(Base):
    """一次对话运行的关键事件，用于断线重连回放和审计。"""

    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_run_events_run_seq"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    run_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="运行 ID",
    )
    thread_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        comment="会话 ID",
    )
    seq: Mapped[int] = mapped_column(
        Integer,
        comment="运行内事件序号，从 0 开始",
    )
    type: Mapped[str] = mapped_column(
        String(50),
        comment="事件类型",
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="事件载荷",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="事件时间",
    )


async def init_run_event_tables() -> None:
    """创建运行事件表。

    使用 checkfirst 保证幂等，只建本模块的表。
    """
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[RunEvent.__table__],
        )
    logger.info("运行事件表初始化完成~☑️")
