"""应用日志配置。"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """配置彩色控制台日志。

    同时配置 Python 标准 logging 与 structlog，并收敛第三方库的日志输出。

    Args:
        level: 日志级别字符串，如 INFO、DEBUG，默认 INFO。
    """
    log_level = _log_level(level)

    # 配置 Python 标准 logging
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    root_logger.addHandler(console_handler)

    # 配置 structlog
    processors: list[Any] = [
        # 合并 request_id、conversation_id 等上下文变量
        structlog.contextvars.merge_contextvars,
        # 添加 Logger 名称
        structlog.stdlib.add_logger_name,
        # 添加日志级别
        structlog.stdlib.add_log_level,
        # 添加时间
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        # 允许添加占位符参数
        structlog.stdlib.PositionalArgumentsFormatter(),
        # 渲染 stack_info=True
        structlog.processors.StackInfoRenderer(),
        # 渲染异常堆栈
        structlog.processors.format_exc_info,
        # 彩色控制台渲染
        structlog.dev.ConsoleRenderer(colors=True),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 控制第三方库日志数量
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO
        if log_level == logging.DEBUG
        else logging.WARNING
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """获取 structlog 日志器。

    Args:
        name: 日志器名称，通常传入 __name__。

    Returns:
        绑定名称的 structlog 日志器实例。
    """
    return structlog.get_logger(name)

def _log_level(level: str) -> int:
    """将字符串日志级别转换为 logging 常量。

    Args:
        level: 日志级别字符串，如 INFO、DEBUG。

    Returns:
        对应的 logging 级别常量；无法识别时返回 logging.INFO。
    """
    return getattr(logging, level.strip().upper(), logging.INFO)