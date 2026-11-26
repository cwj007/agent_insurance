"""RAG 模块检索器。"""

from langchain_milvus import Milvus
from pymilvus import Function, FunctionType

from app.core.config import settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.rag.models import ParentChunk
from app.rag.repository import ParentChunkRepository

logger = get_logger(__name__)

# 重排前的子块候选数量；候选越多召回越好，但重排延迟越高
DEFAULT_FETCH_K = 10

# 释义、疾病附表等章节容易在“流程/规则”类问题中抢占排名，统一降到后面
REFERENCE_SECTION_HINTS: tuple[str, ...] = ("释义", "附表", "病种", "疾病定义")

# 多取的父块数量；降权引用章节后仍要凑够 k 条
REFERENCE_OVERFETCH = 5

# 免责类问法本身已经很明确，扩展“理赔”等词反而会把检索带偏
EXCLUSION_QUERY_HINTS: tuple[str, ...] = (
    "不赔", "不能赔", "不能理赔", "不理赔", "免责", "责任免除", "除外",
    "拒赔", "不承担", "不保", "不予赔付",
)

# 口语问法到条款术语的扩展规则，帮助业务章节进入召回候选
QUERY_EXPANSION_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("理赔", "报销", "赔付", "索赔"),
        ("保险金申请", "索赔材料"),
    ),
    (
        ("等待期", "观察期", "多久生效"),
        ("等待期", "保险责任开始"),
    ),
    (
        ("保险期间", "保障到多少岁", "保多久", "保障期限", "多少岁"),
        ("保险期间", "保障期限", "保险年龄"),
    ),
    (
        ("续保",),
        ("续保", "保证续保"),
    ),
    (
        ("退保", "解除合同"),
        ("退保", "解除合同", "现金价值"),
    ),
    (
        ("免赔额",),
        ("免赔额", "起付线"),
    ),
    (
        ("宽限期", "忘交保费", "交保费"),
        ("宽限期", "效力中止", "复效"),
    ),
    (
        ("犹豫期",),
        ("犹豫期", "全额退还"),
    ),
    (
        ("如实告知", "健康告知", "既往病史"),
        ("如实告知", "健康告知", "告知义务"),
    ),
)


def _expand_query(query: str) -> str:
    """按关键词把口语问法扩展为条款术语。

    Args:
        query: 用户原始问题。

    Returns:
        扩展后的查询；没有命中规则时原样返回。
    """
    # 免责类问法只补责任免除术语，避免“理赔”扩展把检索带向保险金申请
    if any(hint in query for hint in EXCLUSION_QUERY_HINTS):
        terms = ("责任免除", "免责条款", "不承担保险责任")
        additions = [term for term in terms if term not in query]
        return query if not additions else f"{query} {' '.join(additions)}"

    additions: list[str] = []
    for keywords, terms in QUERY_EXPANSION_RULES:
        if any(keyword in query for keyword in keywords):
            additions.extend(term for term in terms if term not in query)

    if not additions:
        return query
    return f"{query} {' '.join(dict.fromkeys(additions))}"


def _is_reference_section(chunk: ParentChunk) -> bool:
    """判断父块是否位于释义或疾病附表章节。

    Args:
        chunk: 父知识块。

    Returns:
        章节路径包含引用类章节关键词时返回 True。
    """
    return any(
        hint in section
        for section in (chunk.section_path or [])
        for hint in REFERENCE_SECTION_HINTS
    )


def _prioritize_chunks(chunks: list[ParentChunk], k: int) -> list[ParentChunk]:
    """把业务章节排在释义/附表之前，再截取前 k 条。

    只降权不删除：业务章节不足 k 条时，引用章节仍会回填。

    Args:
        chunks: 按相关度排序的父块列表。
        k: 返回的最大数量。

    Returns:
        重排后的父块列表。
    """
    primary = [chunk for chunk in chunks if not _is_reference_section(chunk)]
    reference = [chunk for chunk in chunks if _is_reference_section(chunk)]
    return (primary + reference)[:k]


def create_cross_encoder_ranker(queries: str):
    """创建交叉编码 ranker

    Args:
        queries: 查询条件列表。

    Returns:
        交叉编码 ranker 实例。
    """

    return Function(
        name="dashscope_semantic_ranker",  # ranker名词，唯一即可
        input_field_names=["text"],  # 原始文档字段
        function_type=FunctionType.RERANK,  # ranker类型，这里是固定值
        params={
            "reranker": "model",  # rerank类型，基于模型rerank，也就是cross-encoder
            "provider": "ali",  # rerank模型提供者
            "model_name": 'gte-rerank-v2',  # rerank模型名称
            "queries": [queries],  # 查询条件
            "max_client_batch_size": 5,  # 单批次中处理的文档最大数量
        },
    )


class ParentChunkRetriever:
    """RAG 模块检索器。"""

    def __init__(self, vector_store: Milvus):
        self._vector_store = vector_store

    async def retrieve(
            self,
            query: str,
            k: int,
            product_id: int,
            fetch_k: int = DEFAULT_FETCH_K,
            *,
            use_rerank: bool | None = None,
    ) -> list[ParentChunk]:
        """混合检索集成重排模型。

        Args:
            query: 用户提问。
            k: 返回的文档数量。
            product_id: 产品 ID。
            fetch_k: 重排前的子块候选数量。
            use_rerank: 是否启用重排；为空时读取 RAG_ENABLE_RERANK 配置。

        Returns:
            检索到的文档列表。
        """
        if use_rerank is None:
            use_rerank = settings.rag.enable_rerank

        # 多取一些父块，降权引用章节后仍能凑够 k 条
        search_k = k + REFERENCE_OVERFETCH
        search_kwargs = {
            "query": _expand_query(query),
            "k": search_k,
            "fetch_k": max(fetch_k, search_k),
            "expr": f"product_id == {product_id}",
        }

        # 1.在指定产品内混合检索并重排，得到子块
        if use_rerank:
            try:
                child_chunks = await self._vector_store.asimilarity_search(
                    **search_kwargs,
                    reranker=create_cross_encoder_ranker(query),
                )
            except Exception as exc:
                # 重排服务故障时降级为混合检索，避免整个条款查询不可用
                logger.warning(f"重排检索失败，降级为无重排混合检索: {exc}")
                child_chunks = await self._vector_store.asimilarity_search(
                    **search_kwargs,
                )
        else:
            child_chunks = await self._vector_store.asimilarity_search(
                **search_kwargs,
            )

        # 2.提取parent_id，并按照召回顺序去重
        parent_ids = list(
            dict.fromkeys(  # 用dict.fromkeys对parent_id滤重
                child.metadata["parent_id"] for child in child_chunks
            )
        )
        if not parent_ids:
            return []

        # 3.根据parent_id批量查询父块
        async with AsyncSessionFactory() as session:
            repository = ParentChunkRepository(session)
            parent_chunks = await repository.query_by_ids(parent_ids)

        # 4.按照子块召回顺序组装父文档
        # 4.1.先组装dict[parent_id, parent_chunk]
        parent_chunk_map = {
            str(parent.id): parent
            for parent in parent_chunks
        }
        # 4.2.按照召回的parent_ids顺序遍历，再到dict中拿到parent
        ordered = [
            parent_chunk_map[parent_id]  # 推导式的结果
            for parent_id in parent_ids  # 推导式数据来源
        ]
        # 4.3.业务章节优先，释义/附表随后，避免附表挤占 top-k
        return _prioritize_chunks(ordered, k)