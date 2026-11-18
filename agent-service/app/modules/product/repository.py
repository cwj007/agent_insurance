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
