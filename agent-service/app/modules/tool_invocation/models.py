"""工具调用记录模块数据库模型。"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base
from app.core.logging import get_logger
from app.infra.database import engine

logger = get_logger(__name__)


class ToolInvocation(Base):
    """工具调用记录，用于审计和失败定位。"""

    __tablename__ = "tool_invocations"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    thread_id: Mapped[str | None] = mapped_column(
        String(64),
        index=True,
        nullable=True,
        comment="会话 ID",
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger,
        index=True,
        nullable=True,
        comment="用户 ID",
    )
    tool_call_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="模型生成的工具调用 ID",
    )
    tool_name: Mapped[str] = mapped_column(
        String(100),
        index=True,
        comment="工具名",
    )
    args: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        comment="脱敏并截断后的工具参数",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        index=True,
        comment="调用状态：succeeded / failed / blocked",
    )
    result_preview: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="结果预览，截断保存",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="错误码",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="错误说明",
    )
    duration_ms: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="总耗时，毫秒",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="调用时间",
    )


async def init_tool_invocation_tables() -> None:
    """创建工具调用记录表。

    使用 checkfirst 保证幂等，只建本模块的表。
    """
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[ToolInvocation.__table__],
        )
    logger.info("工具调用记录表初始化完成~☑️")
