from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.product.models import Product


class ProductRepository:
    """产品仓库"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def query_by_category(self, category: str | None):
        """根据分类查询产品"""

        # 1. 动态条件
        # 1.1 产品必须是上架的
        conditions = [Product.status == 'active']
        # 1.2 如果 category 不为空，添加分类查询条件
        if category:
            conditions.append(Product.category == category)

        # 2. 执行查询
        result = await self.session.scalars(
            select(Product)
            .where(*conditions)
            .order_by(Product.id.asc())
        )

        return result.all()

    async def find_candidates(
            self,
            categories: list[str],
            premium_min: Decimal | None,
            limit_per_category: int,
    ) -> list[Product]:
        """查询推荐候选产品"""

        # 每个险种查 limit_per_category 条，所以先定义list，收集结果
        candidate_products: list[Product] = []

        # 利用 dict 去重，并保留调用方传入的险种顺序，然后遍历逐个查询
        for category in dict.fromkeys(categories):
            conditions = [
                Product.status == "active",
                Product.category == category,
            ]
            # 2.2 如果 premium_min 不为空，添加保费下限查询条件
            if premium_min is not None:
                conditions.append(Product.min_premium <= premium_min)

            products = await self.session.scalars(
                select(Product)
                .where(*conditions)
                .order_by(
                    Product.min_premium.asc().nullslast(),
                    Product.id.asc(),
                )
                .limit(limit_per_category)
            )
            candidate_products.extend(products.all())

        return candidate_products