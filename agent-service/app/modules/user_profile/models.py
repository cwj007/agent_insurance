"""用户画像模块数据库模型。"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base
from app.core.logging import get_logger
from app.infra.database import engine

logger = get_logger(__name__)


class UserProfile(Base):
    """用户画像主表。

    每个用户一行，profile 字段保存画像字段的 JSONB 快照，
    version 用于乐观并发与变更追溯。
    """

    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        comment="用户 ID",
    )
    profile: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="画像内容，结构为 {'fields': {字段名: 字段对象}}",
    )
    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        server_default="1",
        comment="画像版本号，每次变更递增",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="最近更新时间",
    )


class UserProfileEvent(Base):
    """用户画像变更事件，用于审计和追溯。"""

    __tablename__ = "user_profile_events"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        index=True,
        comment="用户 ID",
    )
    patch: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="本次变更内容，包含变更前后的字段值",
    )
    reason: Mapped[str] = mapped_column(
        String(32),
        comment="变更原因：extraction / correction / deletion",
    )
    source_thread_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="来源会话 ID，抽取变更时记录",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="事件时间",
    )


async def init_user_profile_tables() -> None:
    """创建用户画像相关表。

    使用 checkfirst 保证幂等，只建本模块的两张表，不影响其他业务表。
    """
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[UserProfile.__table__, UserProfileEvent.__table__],
        )
    logger.info("用户画像表初始化完成~☑️")
