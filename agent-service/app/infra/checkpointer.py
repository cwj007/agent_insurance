"""LangChain PostgreSQL 会话记忆存储器。"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 1. 创建 checkpointer 专用连接池
checkpoint_pool = AsyncConnectionPool(
    # checkpoint 的协议地址
    conninfo=settings.db.checkpoint_url,
    min_size=1,  # 最小连接数大小
    max_size=5,  # 最大连接数大小
    kwargs={
        "autocommit": True,  # 自动提交事务
        "prepare_threshold": 0,  # 不做预准备 SQL
        "row_factory": dict_row,  # 查询结果以 dict 格式返回
    },
    open=False  # 不自动创建连接
)


async def init_checkpointer() -> AsyncPostgresSaver:
    """初始化会话记忆存储器。

    打开专用连接池、创建 AsyncPostgresSaver 实例并自动建表。

    Returns:
        初始化完成的 AsyncPostgresSaver 实例。
    """
    # 1. 初始化连接池
    await checkpoint_pool.open()  # 初始化链接
    await checkpoint_pool.wait()  # 等待连接池初始化完成

    # 2. 创建 checkpointer 实例
    checkpointer = AsyncPostgresSaver(checkpoint_pool)

    # 3. 自动创建表
    await checkpointer.setup()
    logger.info("Checkpointer 初始化成功~")

    return checkpointer


async def close_checkpointer() -> None:
    """关闭会话记忆存储器。

    关闭 checkpointer 专用连接池，释放全部数据库连接。
    """
    await checkpoint_pool.close()
    logger.info("Checkpointer 连接池已关闭~")
