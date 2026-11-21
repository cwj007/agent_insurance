"""智能体工具函数"""

from decimal import Decimal

from langchain_core.tools import tool

from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from app.modules.product.schemas import ProductResponse
from app.modules.product.service import ProductService


@tool(parse_docstring=True)
async def query_candidate_products(
        categories: list[str],
        premium_min: Decimal | None = None,
        limit_per_category: int = 5,
) -> list[str]:
    """
    根据保险种和保费条件查询可用于推荐的候选保险产品。当用户查询具体保险产品或需要保险产品推荐时使用。

    Args:
        categories: 产品分类列表，可选值为 [medical, critical_illness, life, accident]。
        premium_min: 最低保费上限，可选参数，只返回最低保费小于该值的产品。
        limit_per_category: 每个险种最多返回的产品数量，可选参数，默认 5 个。
    """

    async with AsyncSessionFactory() as session:
        service = ProductService(session)
        products: list[Product] = await service.list_candidates(
            categories=categories,
            premium_min=premium_min,
            limit_per_category=limit_per_category,
        )

        return [
            ProductResponse.model_validate(product).model_dump_json()
            for product in products
        ]
