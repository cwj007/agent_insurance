"""RAG 模块检索器。"""

from langchain_milvus import Milvus
from pymilvus import Function, FunctionType

from app.infra.database import AsyncSessionFactory
from app.rag.models import ParentChunk
from app.rag.repository import ParentChunkRepository


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

    async def retrieve(self, query: str, k: int, product_id: int) -> list[ParentChunk]:
        """混合检索集成重排模型。

        Args:
            query: 用户提问。
            k: 返回的文档数量。
            product_id: 产品 ID。

        Returns:
            检索到的文档列表。
        """

        # 1.在指定产品内混合检索并重排，得到子块
        child_chunks = await self._vector_store.asimilarity_search(
            query=query,
            k=k,
            fetch_k=10,
            expr=f"product_id == {product_id}",
            reranker=create_cross_encoder_ranker(query),
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
        return [
            parent_chunk_map[parent_id]  # 推导式的结果
            for parent_id in parent_ids  # 推导式数据来源
        ]