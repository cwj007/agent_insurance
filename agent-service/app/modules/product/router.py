from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.product.schemas import ProductResponse
from app.infra.database import get_session
from app.modules.product.service import ProductService

router = APIRouter(prefix="/api/v1/products", tags=["products"])

# 1. 抽取公共函数，生成 service
def get_product_service(session: AsyncSession = Depends(get_session)):
    return ProductService(session)

# 获取产品列表
@router.get(
    "",
    response_model=list[ProductResponse],
    summary="列出保险产品",
    description="""根据分类筛选保险产品列表。\n- **category**: 可选，不传则返回全部分类，返回结果按创建时间倒序排列。"""
)
async def list_products(
        category: str | None = Query(None, description="产品分类编码，不传则返回全部"),
        service: ProductService = Depends(get_product_service),
) -> list[ProductResponse]:
    """列出产品"""

    # 执行查询
    return await service.list_products(category)


# 查询推荐候选产品
@router.get(
    "/candidates",
    response_model=list[ProductResponse],
    summary='列出推荐候选产品',
    description="""根据用户输入的关键词，列出推荐的保险产品。 \n- **categories**: 产品分类编码列表，必填。\n- **premium_min**: 保险金额下限，可选。\n- **limit_per_category**: 每个分类返回数量，默认5个。"""
)
async def list_candidates(
        categories: list[str] = Query(min_length=1),
        premium_min: Decimal | None = None,
        limit_per_category: int = 5,
        service: ProductService = Depends(get_product_service)
) -> list[ProductResponse]:
    # 2.查询结果
    return await service.list_candidates(
        categories=categories,
        premium_min=premium_min,
        limit_per_category=limit_per_category,
    )
