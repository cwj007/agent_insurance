"""工具共享常量。"""

from app.core.config import settings

# 单次工具调用的超时时间，避免数据库或 RAG 卡住拖死整轮运行
TOOL_TIMEOUT_SECONDS = settings.tools.timeout_seconds
