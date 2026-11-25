"""通用请求响应模型基类。"""

from typing import TypeVar, Generic

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Pydantic 请求响应模型基类，允许直接从 ORM 对象转换。"""

    model_config = ConfigDict(from_attributes=True)

# 定义一个通用 泛型
T = TypeVar("T")

class PageResult(BaseSchema, Generic[T]):
    """通用分页结果。"""

    items: list[T]
    page: int
    page_size: int
    total: int