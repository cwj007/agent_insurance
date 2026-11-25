"""RAG 模块父块数据库 CRUD 操作。"""
from typing import Any, Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.models import ParentChunk


class ParentChunkRepository:
    """父块仓库。"""

    def __init__(self, session: AsyncSession):
        """初始化父块仓库。

        Args:
            session: 数据库异步会话。
        """
        self.session = session

    async def add_all(self, parent_chunks: list[ParentChunk]):
        """批量添加父块。

        Args:
            parent_chunks: 待批量新增的父块列表。
        """
        self.session.add_all(parent_chunks)

    async def delete_by_product(self, product_id: int):
        """根据产品 ID 删除所有父块。

        Args:
            product_id: 产品 ID。
        """
        await self.session.execute(
            delete(ParentChunk)
            .where(ParentChunk.product_id == product_id)
        )

    async def query_by_ids(self, ids: list[str]) -> Sequence[ParentChunk]:
        """查询指定 ID 的父块

        Args:
            ids: 父块 ID 列表。

        Returns:
            父块列表。
        """

        result = await self.session.scalars(
            select(ParentChunk)
            .where(ParentChunk.id.in_(ids))
        )

        return result.all()
