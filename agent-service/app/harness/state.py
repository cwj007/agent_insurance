"""运行时共享常量与进程内任务表。"""

import asyncio
import os
import socket
from uuid import uuid4

# 运行终态
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled"}
)

# 图执行失败或没有有效回答时的兜底
FALLBACK_ANSWER = "抱歉，当前服务繁忙，请稍后再试。"

# 这些节点产出经过安全校验的用户可见回答。
# 工作流节点不再提前流式，必须等 output_guard 通过后再推给前端，
# 避免未审核内容先发给客户端。
STREAMABLE_NODES: frozenset[str] = frozenset({
    "guard_reply",
    "output_guard",
})

# 兼容旧的提前流式行为；只有显式开启 STREAM_BEFORE_OUTPUT_GUARD 时才使用
EARLY_STREAMABLE_NODES: frozenset[str] = frozenset({
    "recommendation_node",
    "claim_node",
    "chitchat_node",
    "human_handoff_node",
    "fallback_node",
    "clarify_node",
})

# 事件订阅轮询间隔与默认超时
EVENT_POLL_INTERVAL_SECONDS = 0.15
EVENT_SUBSCRIBE_TIMEOUT_SECONDS = 600.0

# 工具参数推送时的截断长度
TOOL_ARG_PREVIEW_CHARS = 200

# 当前 worker 的唯一标识，用于心跳和跨 worker 取消
WORKER_ID = f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"[:64]

# 运行中的后台任务，按 run_id 索引
_active_tasks: dict[str, asyncio.Task] = {}

# 持久化强度：写操作使用 sync，保证工具副作用与 checkpoint 一致
try:
    from langgraph._internal._constants import CONFIG_KEY_DURABILITY
except ImportError:  # pragma: no cover - 兼容 LangGraph 内部常量变化
    CONFIG_KEY_DURABILITY = "__pregel_durability"
