"""保险产品模块服务。"""

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.product.models import Product
from app.modules.product.repository import ProductRepository


class ProductService:
    """产品服务。"""

    def __init__(self, session: AsyncSession):
        """初始化产品服务。

        Args:
            session: 数据库异步会话。
        """
        self.session = session
        self.repository = ProductRepository(session)

    async def list_products(self, category: str | None = None) -> list[Product]:
        """列出产品。

        Args:
            category: 产品分类编码；None 表示查询全部上架产品。

        Returns:
            产品列表。
        """
        products = await self.repository.query_by_category(category)
        return products

    async def find_by_name(self, name: str) -> list[Product]:
        """按名称查询上架产品。

        Args:
            name: 产品名称。

        Returns:
            匹配到的产品列表；名称为空时返回空列表。
        """
        name = name.strip()
        if not name:
            return []

        return await self.repository.find_by_name(name)

    async def list_candidates(
            self,
            categories: list[str],
            premium_min: Decimal | None,
            limit_per_category: int,
    ) -> list[Product]:
        """列出推荐候选产品。

        Args:
            categories: 产品分类编码列表。
            premium_min: 起投保费上限，None 表示不限制。
            limit_per_category: 每个分类返回的产品数量上限。

        Returns:
            候选产品列表。
        """
        return await self.repository.find_candidates(
            categories=categories,
            premium_min=premium_min,
            limit_per_category=limit_per_category,
        )