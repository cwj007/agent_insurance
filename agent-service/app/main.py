import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

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
    """应用生命周期钩子，用于初始化和关闭数据库连接池"""

    logger.info(f"【{settings.app.name}】项目启动中 ...")

    from app.infra.database import check_database, close_database
    from app.infra.checkpointer import init_checkpointer, close_checkpointer
    from app.agents.insurance_advisor import init_insurance_agent

    try:
        # 1. 导入数据库管理函数，完成初始化
        await check_database()
        # 2. 初始化会话记忆存储器
        checkpoint = await init_checkpointer()
        # 3. 初始化保险顾问智能体: 使用 FastAPI 的 state
        app.state.agent = await init_insurance_agent(checkpoint) # 一切请求都共享 agent
        yield
    finally:
        logger.info("应用关闭中 ...")
        # 4. 关闭数据库连接池
        await close_database()
        # 5. 关闭会话记忆存储器
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

app.include_router(product_router)
app.include_router(chat_thread_router)
app.include_router(chat_router)


# 测试接口，健康检查
@app.get("/healthy", summary="健康检查")
async def healthy() -> dict[str, str]:
    """健康检查接口"""

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
