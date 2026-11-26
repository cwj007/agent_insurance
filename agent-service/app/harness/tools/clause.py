"""保险条款检索工具。"""

import asyncio

from langchain_core.tools import tool
from sqlalchemy.exc import SQLAlchemyError

from app.agents.tools.result import ToolResult, ToolTransientError
from app.harness.tools.constants import TOOL_TIMEOUT_SECONDS
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.product.service import ProductService

logger = get_logger(__name__)

# 条款检索约束
DEFAULT_CLAUSE_CHUNKS = 3
MAX_CLAUSE_CHUNKS = 8
MAX_CLAUSE_CHARS_PER_CHUNK = 1200
MAX_CLAUSE_CHARS_TOTAL = 6000
MAX_QUERY_CHARS = 500


@tool(parse_docstring=True)
async def retrieve_insurance_clause(
        query: str,
        k: int = DEFAULT_CLAUSE_CHUNKS,
        product_id: int | None = None,
        product_name: str | None = None,
) -> str:
    """查询指定保险产品的条款内容。

    可以用 product_id 或 product_name 定位产品；两者都没有时先向用户确认产品。

    Args:
        query: 需要从保险条款中查询的问题，建议具体到保障责任或赔付条件。
        k: 返回的条款片段数量，范围 1~8，默认 3。
        product_id: 保险产品 ID；已知时优先传入。
        product_name: 产品名称；没有产品 ID 时传入，由工具按名称解析产品。

    Returns:
        JSON 字符串。成功时为 {"ok": true, "data": [{"section": ..., "content": ...}]}，
        失败时为 {"ok": false, "error": {...}}。
        条款内容过长时会被截断并注明。
    """
    # 1. 参数校验
    query = query.strip()
    if not query:
        return ToolResult.failure(
            "INVALID_ARGS",
            "query 不能为空",
            hint="传入具体要查询的保障责任或赔付条件",
        ).to_json()

    product_name = (product_name or "").strip() or None

    # 2. 优先用产品名称解析，避免 Agent 带着上一轮的错误 ID 继续查
    if product_name:
        try:
            async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
                async with AsyncSessionFactory() as session:
                    products = await ProductService(session).find_by_name(product_name)
        except TimeoutError as exc:
            logger.warning(f"按名称查询产品超时，可重试: {exc}")
            raise ToolTransientError("产品查询服务超时") from exc
        except (SQLAlchemyError, OSError) as exc:
            logger.warning(f"按名称查询产品失败，可重试: {exc}")
            raise ToolTransientError("产品查询服务暂时不可用") from exc

        if len(products) == 1:
            product_id = products[0].id
        elif not products and product_id is None:
            return ToolResult.failure(
                "NOT_FOUND",
                f"未找到产品: {product_name}",
                hint="请确认产品名称，或先调用 query_candidate_products 获取候选产品",
            ).to_json()
        elif len(products) > 1 and product_id is None:
            candidates = "、".join(
                f"{product.name}(ID={product.id})" for product in products[:5]
            )
            return ToolResult.failure(
                "NOT_FOUND",
                f"产品名称「{product_name}」匹配到多个结果",
                hint=f"请向用户确认具体产品，候选：{candidates}",
            ).to_json()

    if product_id is None:
        return ToolResult.failure(
            "PRODUCT_REQUIRED",
            "缺少产品 ID 或产品名称，无法定位条款",
            hint="请向用户确认产品名称，或先调用 query_candidate_products 获取产品 ID",
        ).to_json()

    k = min(max(int(k), 1), MAX_CLAUSE_CHUNKS)
    query = query[:MAX_QUERY_CHARS]

    # 3. 检索条款；超时和上游异常交给重试中间件处理
    try:
        from app.rag import retriever

        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            chunks = await retriever.retrieve(query, k, product_id)
    except TimeoutError as exc:
        logger.warning(f"条款检索超时，可重试: {exc}")
        raise ToolTransientError("条款检索服务超时") from exc
    except Exception as exc:
        logger.warning(f"条款检索失败，可重试: {exc}")
        raise ToolTransientError("条款检索服务暂时不可用") from exc

    # 4. 裁剪条款内容，限制单条和总长度
    data: list[dict[str, str]] = []
    total_chars = 0
    for chunk in chunks:
        content = chunk.content.strip()
        if not content:
            continue
        if len(content) > MAX_CLAUSE_CHARS_PER_CHUNK:
            content = content[:MAX_CLAUSE_CHARS_PER_CHUNK] + "\n...(内容过长已截断)"

        section = " > ".join(chunk.section_path) if chunk.section_path else "未标注章节"
        item_chars = len(section) + len(content)
        if total_chars + item_chars > MAX_CLAUSE_CHARS_TOTAL:
            break
        data.append({"section": section, "content": content})
        total_chars += item_chars

    if not data:
        return ToolResult.failure(
            "NOT_FOUND",
            "没有检索到相关条款",
            hint="换一种更具体的问法，或先确认产品 ID 是否正确",
        ).to_json()

    return ToolResult.success(data).to_json()
