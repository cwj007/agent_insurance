"""智能体工具函数。"""

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
    """根据险种和保费条件查询可用于推荐的候选保险产品。

    当用户查询具体保险产品或需要保险产品推荐时使用。

    Args:
        categories: 产品分类列表，可选值为 [medical, critical_illness, life, accident]。
        premium_min: 起投保费上限，可选参数，只返回最低保费不超过该值的产品。
        limit_per_category: 每个险种最多返回的产品数量，可选参数，默认 5 个。

    Returns:
        产品 JSON 字符串列表，每个元素为一个产品的序列化结果。
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

@tool(parse_docstring=True)
async def retrieve_insurance_clause(
        query: str,
        k: int,
        product_id: int,
):
    """查询指定保险产品的条款内容。

    Args:
        query: 需要从保险条款中查询的问题。
        k: 返回的文档数量。
        product_id: 保险产品 ID。

    Returns:
        检索到的文档列表，每个元素为一个包含 section_path 和 content 字段的字符串。
    """

    # 导入 Retriever
    from app.rag import retriever

    # 检索
    chunks = await retriever.retrieve(query, k, product_id)

    # 手动将数据转换为字符串
    results = [
        f"章节: {chunk.section_path}\n 内容: {chunk.content}"
        for chunk in chunks
    ]

    # 返回结果
    return results