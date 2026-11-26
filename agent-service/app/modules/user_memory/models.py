"""用户情节记忆数据库模型。"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base
from app.core.logging import get_logger
from app.infra.database import engine

logger = get_logger(__name__)


class UserMemory(Base):
    """用户情节记忆：跨会话的关键事实、事件和决定。"""

    __tablename__ = "user_memories"

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
    thread_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="来源会话 ID",
    )
    kind: Mapped[str] = mapped_column(
        String(32),
        comment="类型：fact / preference / event / decision",
    )
    content: Mapped[str] = mapped_column(
        Text,
        comment="一句话记忆内容",
    )
    importance: Mapped[float] = mapped_column(
        Float,
        default=0.5,
        comment="重要度，0~1",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        comment="记忆时间",
    )


async def init_user_memory_tables() -> None:
    """创建用户情节记忆表。

    使用 checkfirst 保证幂等，只建本模块的表。
    """
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[UserMemory.__table__],
        )
    logger.info("用户情节记忆表初始化完成~☑️")
