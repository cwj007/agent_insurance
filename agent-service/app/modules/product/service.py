"""保险产品模块数据库服务"""

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.product.models import Product
from app.modules.product.repository import ProductRepository


class ProductService:
    """产品服务"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = ProductRepository(session)

    async def list_products(self, category: str | None = None) -> list[Product]:
        """列出产品"""

        products = await self.repository.query_by_category(category)
        return products

    async def list_candidates(
            self,
            categories: list[str],
            premium_min: Decimal | None,
            limit_per_category: int,
    ) -> list[Product]:
        """列出推荐候选产品"""

        return await self.repository.find_candidates(
            categories=categories,
            premium_min=premium_min,
            limit_per_category=limit_per_category,
        )