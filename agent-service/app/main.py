"""应用入口：创建 FastAPI 应用，管理生命周期、全局异常与路由注册。"""

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from app.harness.bootstrap import build_insurance_agent_graph

# 以当前文件为基准计算目录，避免依赖启动时的工作目录
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# 保证无论从哪个工作目录启动，都能导入 app 包
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.exceptions import ApplicationError
from app.core.logging import configure_logging, get_logger

# 日志配置
configure_logging(settings.logging.level)
# 获取日志记录器
logger = get_logger(__name__)


# 初始化生命周期钩子
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子，用于初始化和关闭数据库连接池。

    Args:
        app: FastAPI 应用实例，用于挂载 agent 等共享状态。

    Yields:
        应用运行期间的控制权。
    """
    logger.info(f"【{settings.app.name}】项目启动中 ...")

    from app.infra.database import AsyncSessionFactory, check_database, close_database
    from app.infra.checkpointer import init_checkpointer, close_checkpointer
    from app.infra.retention import (
        cleanup_expired_records,
        start_retention_worker,
        stop_retention_worker,
    )
    from app.harness import (
        shutdown_runner,
        start_stale_run_reaper,
        stop_stale_run_reaper,
    )
    from app.modules.agent_run.models import init_agent_run_tables
    from app.modules.agent_run.repository import AgentRunRepository
    from app.modules.user_profile.models import init_user_profile_tables
    from app.modules.user_memory.models import init_user_memory_tables
    from app.modules.tool_invocation.models import init_tool_invocation_tables
    from app.modules.tool_invocation.recorder import shutdown_tool_invocation_recorder
    from app.modules.run_event.models import init_run_event_tables
    from app.rag import close_rag

    retention_stop: asyncio.Event | None = None
    retention_task: asyncio.Task | None = None

    try:
        # 1. 导入数据库管理函数，完成初始化
        await check_database()
        # 2. 初始化用户画像、情节记忆、工具调用记录、运行事件、运行主表
        await init_user_profile_tables()
        await init_user_memory_tables()
        await init_tool_invocation_tables()
        await init_run_event_tables()
        await init_agent_run_tables()

        # 3. 只逐出心跳超时的运行，避免误伤其他 worker 正在执行的任务
        async with AsyncSessionFactory() as session:
            async with session.begin():
                stale_count = await AgentRunRepository(session).mark_stale_runs_failed(
                    settings.agent_run.stale_run_seconds
                )
        if stale_count:
            logger.warning(f"已逐出 {stale_count} 条心跳超时的运行")

        # 3.1 启动周期性失联运行清理，worker 崩溃后由其他实例接管
        start_stale_run_reaper()

        # 4. 启动时清理一次过期数据，并启动定时清理任务
        await cleanup_expired_records()
        retention_stop, retention_task = start_retention_worker()

        # 5. 初始化会话记忆存储器
        checkpoint = await init_checkpointer()
        # 6. 初始化保险顾问智能体: 使用 FastAPI 的 state
        app.state.agent = await build_insurance_agent_graph(checkpoint)  # 一切请求都共享 agent

        yield
    finally:
        logger.info("应用关闭中 ...")
        # 7. 停止过期数据清理任务
        if retention_stop is not None and retention_task is not None:
            await stop_retention_worker(retention_stop, retention_task)
        # 8. 先停止失联运行清理，再取消运行中的 Agent 任务
        await stop_stale_run_reaper()
        await shutdown_runner()
        # 9. 刷新工具调用记录，避免后台任务丢失
        await shutdown_tool_invocation_recorder()
        # 10. 关闭 RAG 模块资源
        close_rag()
        # 11. 关闭数据库连接池
        await close_database()
        # 12. 关闭会话记忆存储器
        await close_checkpointer()


# 初始化 FastAPI 应用
app = FastAPI(
    title=settings.app.name,  # 应用名称
    version=settings.app.version,  # 应用版本
    debug=settings.app.debug,  # 是否开启调试模式
    lifespan=lifespan,  # 生命周期钩子
)


# 全局异常处理
@app.exception_handler(ApplicationError)
def handle_exception(request: Request, exc: ApplicationError):
    """全局异常处理，将业务异常转换为统一的 JSON 响应。

    Args:
        request: FastAPI 请求对象，用于记录请求路径。
        exc: 捕获到的业务异常。

    Returns:
        包含业务状态码、错误编码和错误信息的 JSON 响应。
    """
    logger.error(f"处理异常, 请求路径: {request.url}, 异常编码: {exc.code}, 异常信息: {exc.message}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message
        },
    )


# 挂载路由
from app.modules.product.router import router as product_router
from app.modules.chat_threads.router import router as chat_thread_router
from app.modules.chat.router import router as chat_router
from app.modules.user_profile.router import router as user_profile_router
from app.modules.user_memory.router import router as user_memory_router

app.include_router(product_router)
app.include_router(chat_thread_router)
app.include_router(chat_router)
app.include_router(user_profile_router)
app.include_router(user_memory_router)


# 测试接口，健康检查
@app.get("/healthy", summary="健康检查")
async def healthy() -> dict[str, str]:
    """健康检查接口。

    Returns:
        固定返回 `{"status": "healthy"}`，用于探活检查。
    """
    logger.info("健康检查接口正常调用")
    return {"status": "healthy"}


# 启动项目
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=True,
        # 使用绝对路径，无论工作目录是项目根还是 app 目录，热重载都能生效
        reload_dirs=[str(APP_DIR)],
        reload_excludes=["*.pyc", "*/migrations/*"],
    )
