"""通用数据库模型基类。"""

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime
from sqlalchemy import DateTime, func


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类，所有数据库表模型都必须继承。"""

    def __repr__(self) -> str:
        """生成对象的调试字符串，隐藏以下划线开头的私有属性。

        Returns:
            形如 `类名({属性字典})` 的字符串，日期时间统一格式化为 `%Y-%m-%d %H:%M:%S`。
        """
        cls_name = self.__class__.__name__
        attrs = {}
        for key, val in vars(self).items():
            if key.startswith("_"):
                continue
            if isinstance(val, datetime):
                attrs[key] = val.strftime("%Y-%m-%d %H:%M:%S")
            else:
                attrs[key] = val
        return f"{cls_name}({attrs})"


class CreateAtMixin:
    """通用创建时间混入类，让数据模型复用创建时间字段。"""

    # func.now() 会让数据库在插入时自动生成服务器时间（如 PostgreSQL 的 NOW()）
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )


class UpdateAtMixin:
    """通用更新时间混入类，让数据模型复用更新时间字段。"""

    # onupdate=func.now() 会在每次数据行被 UPDATE 时，由 SQLAlchemy 自动更新该字段
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()  # 每次更新时自动更新当前时间
    )
