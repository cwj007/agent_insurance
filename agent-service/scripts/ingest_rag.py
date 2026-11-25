"""RAG 知识库构建脚本。

查询数据库中全部上架保险产品，逐个解析条款 PDF，并将切分后的父子块
分别写入 Milvus 向量库和关系数据库。

用法：
    uv run python scripts/ingest_rag.py
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录加入模块搜索路径，支持直接运行 `python scripts/ingest_rag.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import OllamaEmbeddings

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.product.models import Product
from app.modules.product.service import ProductService
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)

# 嵌入模型配置
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BASE_URL = "http://127.0.0.1:11434"

# Milvus Collection 名称
COLLECTION_NAME = "insurance_collection"


def build_embeddings() -> OllamaEmbeddings:
    """初始化 Ollama 嵌入模型。

    Returns:
        配置好的 Ollama 嵌入模型实例。
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        base_url=EMBEDDING_BASE_URL,
    )


def build_vector_store(embeddings: OllamaEmbeddings) -> Milvus:
    """初始化 Milvus 向量库。

    Collection 不存在时会自动创建，包含稠密向量字段和 BM25 稀疏向量字段。

    Args:
        embeddings: 稠密向量嵌入模型。

    Returns:
        Milvus 向量库实例。
    """
    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        builtin_function=BM25BuiltInFunction(
            analyzer_params={"type": "chinese"},
        ),
        vector_field=["dense", "sparse"],
        connection_args={"uri": settings.rag.milvus_url},
        auto_id=True,
    )


async def load_products() -> list[Product]:
    """查询全部上架的保险产品。

    Returns:
        处于上架状态的产品列表，按产品 ID 升序排列。
    """
    async with AsyncSessionFactory() as session:
        service = ProductService(session)
        return await service.list_products()


async def main() -> None:
    """构建全部保险产品的 RAG 知识库。"""
    configure_logging(settings.logging.level)

    # 1. 初始化嵌入模型和 Milvus 向量库
    embeddings = build_embeddings()
    vector_store = build_vector_store(embeddings)

    # 2. 初始化知识库构建管道
    pipeline = RAGPipeline(vector_store)

    try:
        # 3. 查询全部上架产品
        products = await load_products()
        logger.info("待处理产品查询完成", product_count=len(products))

        # 4. 逐个产品解析条款并构建知识库
        for index, product in enumerate(products, start=1):
            logger.info(
                "开始构建产品知识库",
                index=index,
                total=len(products),
                product_id=product.id,
                product_name=product.name,
            )
            await pipeline.ingest(product)
            logger.info(
                "产品知识库构建完成",
                product_id=product.id,
                product_name=product.name,
            )

        logger.info("RAG知识库构建完成", product_count=len(products))
    finally:
        # 5. 释放 MinerU、Milvus 和数据库连接
        pipeline.close()
        vector_store.client.close()
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
