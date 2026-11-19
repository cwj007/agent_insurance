from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

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

    logger.info("项目启动中，正在初始化 ...")
    from app.infra.database import check_database, close_database

    try:
        # 1. 导入数据库管理函数，完成初始化
        await check_database()

        yield
    finally:
        # 2. 关闭数据库连接池
        await close_database()


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

app.include_router(product_router)
app.include_router(chat_thread_router)


# 测试接口，健康检查
@app.get("/healthy", summary="健康检查")
async def healthy() -> dict[str, str]:
    """健康检查接口"""

    logger.info("健康检查接口正常调用")
    return {"status": "healthy"}


# 启动项目
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # ✅ 改成这个
        host=settings.app.host,
        port=settings.app.port,
        reload=True,
        reload_dirs=["app"],  # ✅ 对应改成 app
        reload_excludes=["*.pyc", "*/migrations/*"],
    )
