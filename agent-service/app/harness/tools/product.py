"""产品查询工具。"""

import asyncio
from decimal import Decimal
from typing import Any

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.agents.tools.result import ToolResult, ToolTransientError
from app.harness.tools.constants import TOOL_TIMEOUT_SECONDS
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from app.modules.product.service import ProductService

logger = get_logger(__name__)

# 产品查询约束
PRODUCT_CATEGORIES: frozenset[str] = frozenset(
    {"medical", "critical_illness", "life", "accident"}
)
DEFAULT_LIMIT_PER_CATEGORY = 3
MAX_LIMIT_PER_CATEGORY = 5
MAX_PRODUCTS_TOTAL = 20


def _product_summary(product: Product) -> dict[str, Any]:
    """把产品 ORM 对象裁剪成推荐所需的精简字段。

    Args:
        product: 产品 ORM 对象。

    Returns:
        精简后的产品字典。
    """
    return {
        "id": product.id,
        "name": product.name,
        "category": product.category,
        "min_premium": (
            float(product.min_premium)
            if product.min_premium is not None
            else None
        ),
        "target_group": product.target_group,
        "highlights": list(product.highlights or [])[:5],
    }


@tool(parse_docstring=True)
async def query_candidate_products(
        categories: list[str],
        premium_min: Decimal | None = None,
        limit_per_category: int = DEFAULT_LIMIT_PER_CATEGORY,
) -> str:
    """根据险种和保费条件查询可用于推荐的候选保险产品。

    当用户需要保险产品推荐、或需要先确定产品 ID 再查询条款时使用。

    Args:
        categories: 产品分类列表，只能取 medical、critical_illness、life、accident。
        premium_min: 可选，只返回最低保费不超过该值的产品（单位：元）。
        limit_per_category: 每个险种最多返回的产品数量，范围 1~5，默认 3。

    Returns:
        JSON 字符串。成功时为 {"ok": true, "data": [...]}，
        失败时为 {"ok": false, "error": {"code": ..., "message": ..., "hint": ...}}。
        产品字段包含 id、name、category、min_premium、target_group、highlights。
    """
    # 1. 参数校验：分类必须合法，数量必须收敛在可控范围
    invalid_categories = [
        category for category in categories if category not in PRODUCT_CATEGORIES
    ]
    if not categories or invalid_categories:
        return ToolResult.failure(
            "INVALID_ARGS",
            f"categories 包含非法值: {invalid_categories or categories}",
            hint="categories 只能取 medical、critical_illness、life、accident",
        ).to_json()

    if premium_min is not None and premium_min < 0:
        return ToolResult.failure(
            "INVALID_ARGS",
            "premium_min 不能为负数",
            hint="请传入大于等于 0 的保费，或省略该参数",
        ).to_json()

    limit = min(
        max(int(limit_per_category), 1),
        MAX_LIMIT_PER_CATEGORY,
    )

    # 2. 查询候选产品；超时和上游异常交给重试中间件处理
    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            async with AsyncSessionFactory() as session:
                service = ProductService(session)
                products: list[Product] = await service.list_candidates(
                    categories=list(categories),
                    premium_min=premium_min,
                    limit_per_category=limit,
                )
    except TimeoutError as exc:
        logger.warning(f"查询候选产品超时，可重试: {exc}")
        raise ToolTransientError("产品查询服务超时") from exc
    except (SQLAlchemyError, OSError) as exc:
        logger.warning(f"查询候选产品失败，可重试: {exc}")
        raise ToolTransientError("产品查询服务暂时不可用") from exc

    # 3. 裁剪结果，避免完整 JSON 撑爆上下文
    data = [_product_summary(product) for product in products[:MAX_PRODUCTS_TOTAL]]
    if not data:
        return ToolResult.failure(
            "NOT_FOUND",
            "没有符合条件的候选产品",
            hint="可以放宽保费条件或更换险种后重试",
        ).to_json()

    return ToolResult.success(data).to_json()
