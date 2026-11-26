"""RAG 模块。"""

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import ParentChunkRetriever

logger = get_logger(__name__)

# 1.初始化向量模型
_embeddings = OllamaEmbeddings(
    model=settings.rag.embedding_model,
    dimensions=settings.rag.embedding_dimensions,
    base_url=settings.llm.ollama_base_url,
    client_kwargs={
        "trust_env": False,  # 关键：忽略系统代理/环境变量
        "timeout": 60.0,
    },
)

# 2.初始化VectorStore
_vector_store = Milvus(
    embedding_function=_embeddings,
    collection_name=settings.rag.collection_name,
    builtin_function=BM25BuiltInFunction(
        analyzer_params={"type": "chinese"}
    ),
    vector_field=["dense", "sparse"],
    connection_args={"uri": settings.rag.milvus_url},
    auto_id=True,
)

# 3.创建模块级Pipeline和Retriever
pipeline = RAGPipeline(_vector_store)
retriever = ParentChunkRetriever(_vector_store)


def close_rag() -> None:
    """关闭RAG组件"""
    pipeline.close()
    _vector_store.client.close()
    logger.info("RAG 模块资源已关闭")


__all__ = ['pipeline', 'retriever', 'close_rag']
