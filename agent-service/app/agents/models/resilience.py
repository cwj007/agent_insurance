"""模型调用的重试、兜底与熔断。"""

import asyncio
import time
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 主模型总尝试次数（含第一次），以及每次重试前的退避时间
DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 0.3

# 主模型连续失败多少次后熔断，以及熔断冷却时间
DEFAULT_BREAKER_FAILURES = 3
DEFAULT_BREAKER_COOLDOWN_SECONDS = 60.0


def build_fallback_chat_model() -> BaseChatModel | None:
    """按配置创建 DashScope 兜底聊天模型。

    Returns:
        兜底模型；未配置 API Key 或初始化失败时返回 None。
    """
    api_key = settings.llm.dashscope_api_key
    model_name = settings.llm.dashscope_chat_model
    if not api_key or not model_name:
        logger.warning("未配置 DashScope 兜底模型，主模型故障时无备用模型")
        return None

    try:
        # 1. 复用项目已有的 DashScope 依赖，避免引入新的模型供应商
        from langchain_community.chat_models import ChatTongyi

        model = ChatTongyi(model=model_name, api_key=api_key, temperature=0)
        logger.info(f"DashScope 兜底模型初始化完成: {model_name}~☑️")
        return model
    except Exception as exc:
        logger.warning(f"DashScope 兜底模型初始化失败: {exc}")
        return None


def build_model_reliability_middlewares(
        fallback_model: BaseChatModel | None,
) -> list[Any]:
    """构建模型调用的重试与兜底中间件。

    Args:
        fallback_model: 主模型失败时使用的备用模型；为空时只做重试。

    Returns:
        中间件列表：兜底在外层，重试在内层，重试耗尽后由兜底接管。
    """
    from langchain.agents.middleware import (
        ModelFallbackMiddleware,
        ModelRetryMiddleware,
    )

    middlewares: list[Any] = []
    if fallback_model is not None:
        middlewares.append(ModelFallbackMiddleware(fallback_model))

    middlewares.append(
        ModelRetryMiddleware(
            max_retries=1,
            on_failure="error",
            initial_delay=0.5,
            max_delay=2.0,
            jitter=False,
        )
    )
    return middlewares


class CircuitBreaker:
    """连续失败熔断器。

    主模型连续失败达到阈值后，在冷却时间内直接跳过主模型，
    避免每个请求都先等一次必然失败的调用。

    Attributes:
        _failure_threshold: 触发熔断的连续失败次数。
        _cooldown_seconds: 熔断冷却时间。
        _failures: 当前连续失败次数。
        _opened_at: 熔断开始时间；未熔断时为 None。
    """

    def __init__(
            self,
            failure_threshold: int = DEFAULT_BREAKER_FAILURES,
            cooldown_seconds: float = DEFAULT_BREAKER_COOLDOWN_SECONDS,
    ):
        """初始化熔断器。

        Args:
            failure_threshold: 连续失败多少次后熔断。
            cooldown_seconds: 熔断后冷却多少秒再尝试。
        """
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        """判断当前是否允许调用主模型。

        Returns:
            未熔断或冷却已结束时返回 True。
        """
        if self._opened_at is None:
            return True

        if time.monotonic() - self._opened_at >= self._cooldown_seconds:
            self._failures = 0
            self._opened_at = None
            return True

        return False

    def record_success(self) -> None:
        """记录一次成功，重置连续失败计数。"""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """记录一次失败，达到阈值后打开熔断。"""
        self._failures += 1
        if self._failures < self._failure_threshold:
            return

        self._opened_at = time.monotonic()
        logger.warning(
            f"主模型连续失败 {self._failures} 次，"
            f"熔断 {self._cooldown_seconds:.0f}s"
        )


async def ainvoke_with_fallback(
        primary: Any,
        messages: list,
        fallback: Any | None = None,
        *,
        breaker: CircuitBreaker | None = None,
        attempts: int = DEFAULT_RETRY_ATTEMPTS,
        backoff: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        validate: Callable[[Any], bool] | None = None,
        label: str = "模型",
) -> Any:
    """带重试、模型兜底和熔断的异步调用。

    调用顺序：
    1. 主模型最多尝试 attempts 次，失败后短暂退避；
    2. 主模型不可用或重试仍失败时，切换兜底模型再试一次；
    3. 兜底模型也失败时抛出最后一条异常，由调用方做业务降级。

    Args:
        primary: 主模型调用链。
        messages: 传给模型的 messages。
        fallback: 可选的兜底模型调用链。
        breaker: 可选的主模型熔断器。
        attempts: 主模型总尝试次数，含第一次。
        backoff: 重试退避基数，按尝试次数线性增长。
        validate: 可选的结果校验函数；返回 False 视为本次失败。
        label: 日志中使用的调用名称。

    Returns:
        主模型或兜底模型返回的结果。

    Raises:
        Exception: 主模型和兜底模型都失败时抛出最后一次异常。
    """
    attempts = max(1, attempts)
    last_exc: Exception | None = None
    primary_attempted = False

    if breaker is None or breaker.allow():
        for attempt in range(1, attempts + 1):
            # 重试过程中可能刚好触发熔断，此时不再继续打主模型
            if breaker is not None and not breaker.allow():
                break

            primary_attempted = True
            validation_failed = False
            try:
                result = await primary.ainvoke(messages)
                if validate is not None and not validate(result):
                    # 结果校验失败不代表模型不可用，不计入熔断
                    validation_failed = True
                    raise ValueError("模型结果未通过校验")

                if breaker is not None:
                    breaker.record_success()
                logger.debug(f"{label} 主模型第 {attempt} 次调用成功")
                return result
            except Exception as exc:
                last_exc = exc
                if breaker is not None and not validation_failed:
                    breaker.record_failure()
                logger.warning(f"{label} 主模型第 {attempt}/{attempts} 次失败: {exc}")
                if attempt < attempts:
                    await asyncio.sleep(backoff * attempt)

    if not primary_attempted:
        logger.warning(f"{label} 主模型处于熔断状态，直接使用兜底模型")

    if fallback is not None:
        try:
            result = await fallback.ainvoke(messages)
            if validate is not None and not validate(result):
                raise ValueError("兜底模型结果未通过校验")

            logger.debug(f"{label} 兜底模型调用成功")
            return result
        except Exception as exc:
            last_exc = exc
            logger.error(f"{label} 兜底模型失败: {exc}")

    if last_exc is not None:
        raise last_exc

    raise RuntimeError(f"{label} 没有可用的模型")
