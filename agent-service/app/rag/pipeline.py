"""知识库构建管道。"""

from typing import Any
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from mineru import MinerU

from app.core.config import APP_ROOT, settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from app.rag.models import ParentChunk
from app.rag.repository import ParentChunkRepository

logger = get_logger(__name__)

MARKDOWN_OPTIMIZATION_PROMPT = """
下面这份Markdown文档是从保险条款PDF解析得来，由于PDF中的各个小节是以表格形式存在，所以解析时出现错乱。你分析内容，帮我转为格式正确的Markdown，特别是标题编号要正确。
- 标题等级要从1级标题开始，逐层增加，目录和文档名不计入标题等级。
- 输出结果中不要包含正文开始之前的部分。
- 输出结果只包含Markdown正文，不要解释或代码围栏。
- 不要修改保险条款。
""".strip()


class RagPipeline:
    """知识库构建管道。"""

    def __init__(self, vector_store: Milvus):
        """初始化知识库构建管道。

        Args:
            vector_store: Milvus 向量库实例，用于存储和检索子块。
        """
        self._vector_store = vector_store
        # 1. 初始化 MinerU 客户端
        self._mineru = MinerU(token=settings.rag.mineru_token)
        # 2. 初始化模型
        self._llm = init_chat_model(
            model_provider='deepseek',
            model=settings.llm.deepseek_chat_model,
            api_key=settings.llm.deepseek_api_key,
            extra_body={'thinking': {'type': 'disabled'}},
            max_tokens=100000
        )
        # 3. 初始化文档切分器
        # 3.1. Markdown 文档切分器
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=True,  # 文章中不携带标题
        )

        # 3.2. 递归切分器，切子块
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", ";", "，", ","],
            keep_separator="end"
        )

    async def ingest(self, product: Product):
        """将单个保险产品的条款文档构建为知识库数据。

        解析 PDF 条款为 Markdown，切分为父子块，并将子块写入向量库。

        Args:
            product: 待入库的保险产品，包含条款文件名、产品 ID 等信息。
        """
        # 1. 加载和解析文档
        file_path = str(APP_ROOT / "data/raw/kb" / product.clause_name)
        markdown = self._parse_document(file_path)

        # 2. 切分父子块
        parent_chunks, child_chunks = self._split_documents(markdown, product)

        # 3. 子块写入向量数据库
        self._save_child_chunks(child_chunks, product)

        # 4. 父块写入关系数据库
        await self._save_parent_chunks(parent_chunks, product)

    def _parse_document(self, file_path: str) -> str:
        """使用 MinerU 解析条款 PDF，并由大模型修复 Markdown 格式。

        Args:
            file_path: 条款 PDF 文件路径。

        Returns:
            格式修复后的 Markdown 正文。
        """
        result = self._mineru.extract(file_path)

        response = self._llm.invoke([
            {"role": "system", "content": MARKDOWN_OPTIMIZATION_PROMPT},
            {"role": "user", "content": result.markdown},
        ])

        return response.content

    def _split_documents(
            self,
            markdown: str | list[str | dict[Any, Any]],
            product: Product,
    ) -> tuple[list[ParentChunk], list[Document]]:
        """切分文档为父块和子块。

        Args:
            markdown: 清洗后的解析文档内容。
            product: 产品信息。

        Returns:
            父块列表与子块文档列表组成的元组，顺序为 (parent_chunks, child_chunks)。
        """
        # 2. 切分文档
        sections = self._markdown_splitter.split_text(markdown)

        # 3. 处理父子块
        parent_chunks: list[ParentChunk] = []
        child_chunks: list[Document] = []

        # 3.1. 遍历父块小节
        for section in sections:
            # 3.1.1. 生成章节路径，section.metadata 是各级标题的字典，values 即标题列表
            section_path = list(section.metadata.values())

            # 3.2. 组织 parent_chunk
            parent = ParentChunk(
                id=uuid4(),
                product_id=product.id,
                content=section.page_content,
                section_path=section_path,
                clause_name=product.clause_name
            )
            # 收集父块
            parent_chunks.append(parent)

            # 3.3. 利用递归切分器切分父块得到子块，并重新构造 metadata
            section.metadata = {
                "product_id": product.id,
                "parent_id": str(parent.id)
            }
            # 这里会切分出多个 list[Document]
            children = self._child_splitter.split_documents([section])

            # 凭借 section_path，放到子块的 page_content 中
            section_path_header = "\n".join(section_path) + "\n"
            # 将所有子块进行拼接
            for child in children:
                child.page_content = section_path_header + child.page_content
            # 所以使用 extend
            child_chunks.extend(children)

        return parent_chunks, child_chunks

    def _save_child_chunks(self, child_chunks: list[Document], product: Product):
        """将子块写入向量数据库。

        写入前先删除该产品已有的向量数据，避免重复导入产生脏数据。

        Args:
            child_chunks: 待写入的子块文档列表。
            product: 所属产品，用于定位并清理旧向量数据。
        """

        # Collection 存在时，先删除，在写入
        if self._vector_store.client.has_collection(
                self._vector_store.collection_name
        ):
            # 先删除
            self._vector_store.delete(expr=f"product_id=={product.id}")

        # 在存储
        self._vector_store.add_documents(child_chunks)

    async def _save_parent_chunks(self, parent_chunks: list[ParentChunk], product: Product):
        """将父块写入关系数据库。

        Args:
            parent_chunks: 待写入的父块列表。
            product: 所属产品，用于定位并清理旧数据。
        """

        async with AsyncSessionFactory() as session:
            async with session.begin():
                repository = ParentChunkRepository(session)

                # 1. 先删除
                await repository.delete_by_product(product.id)
                # 2. 在添加
                await repository.add_all(parent_chunks)

    def close(self):
        """关闭mineru连接"""

        self._mineru.close()