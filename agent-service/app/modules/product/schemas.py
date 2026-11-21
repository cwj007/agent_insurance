"""保险产品模块响应格式"""

from decimal import Decimal
from pydantic import BaseModel, field_validator, ConfigDict

from app.common.schemas import BaseSchema


class ProductResponse(BaseSchema):
    """保险产品响应结果"""

    # 通常情况下，普通对象是无法直接转换成 BaseModel 类对象
    # 1. 使用 ConfigDict 配置模型，允许从 ORM 对象直接创建/序列化 Pydantic 模型
    # 2. 使用 Pydantic 提供的 model_validate 方法可以将普通对象转换为 Pydantic 模型
    # 3. 在 BaseSchema 已经配置了 from_attributes=True，所以这里不需要再配置
    # model_config = ConfigDict(from_attributes=True) # 允许从 ORM 对象直接创建/序列化 Pydantic 模型

    id: int
    name: str
    clause_name: str
    category: str
    insurer: str
    image_url: str
    description: str
    min_premium: Decimal | None
    max_premium: Decimal | None
    target_group: str | None
    highlights: list[str]
    status: str

    @field_validator("highlights", mode="before")
    @classmethod
    def normalize_highlights(cls, value: list[str] | None) -> list[str] | None:
        return value or []