"""运行器：创建运行、后台执行、流式事件与关闭清理。"""

import asyncio
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.state import CompiledStateGraph

from app.harness.events import _append_event, _update_run_status
from app.harness.heartbeat import _run_heartbeat_loop
from app.harness.profile_extractor import schedule_profile_extraction
from app.harness.state import (
    CONFIG_KEY_DURABILITY,
    EARLY_STREAMABLE_NODES,
    FALLBACK_ANSWER,
    STREAMABLE_NODES,
    TOOL_ARG_PREVIEW_CHARS,
    WORKER_ID,
    _active_tasks,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.agent_run.models import AgentRun
from app.modules.run_event.models import RunEvent
from app.modules.user_memory.service import UserMemoryService
from app.modules.user_profile.service import UserProfileService

logger = get_logger(__name__)


def _message_text(message: BaseMessage) -> str:
    """提取消息中的纯文本内容。

    Args:
        message: LangChain 消息对象。

    Returns:
        消息文本；多模态内容会拼接其中的文本块。
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _last_ai_text_or_none(state: dict) -> str | None:
    """提取最后一条有效 AI 消息文本。

    Args:
        state: Graph 状态或节点输出，需包含 messages。

    Returns:
        最后一条有效 AI 文本；没有时返回 None。
    """
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            text = _message_text(message).strip()
            if text:
                return text
    return None


def _last_ai_text(state: dict) -> str:
    """提取最终 AI 回答。

    Args:
        state: Graph 最终状态。

    Returns:
        最后一条有效 AI 消息；没有时返回兜底话术。
    """
    return _last_ai_text_or_none(state) or FALLBACK_ANSWER


def _safe_tool_args(args: Any) -> dict[str, Any]:
    """截断工具参数，避免长文本进入事件表。

    Args:
        args: 工具调用参数。

    Returns:
        可 JSON 序列化且已截断的参数。
    """
    if not isinstance(args, dict):
        return {}

    safe: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            safe[key] = value[:TOOL_ARG_PREVIEW_CHARS]
        elif isinstance(value, list):
            safe[key] = [
                item[:100] if isinstance(item, str) else item
                for item in value[:20]
            ]
        else:
            safe[key] = value
    return safe


async def start_run(
        agent: CompiledStateGraph,
        *,
        user_id: int | None,
        thread_id: str,
        message: str,
        replay_of_run_id: str | None = None,
        resume: bool = False,
) -> str:
    """创建一次运行并在后台执行。

    Args:
        agent: 已编译的主 Graph。
        user_id: 用户 ID。
        thread_id: 会话 ID。
        message: 用户消息。
        replay_of_run_id: 重放/重试来源的运行 ID。
        resume: 是否从上次失败的 checkpoint 续跑，不重新写入用户消息。

    Returns:
        新建的运行 ID。
    """
    run_id = uuid4().hex
    profile_text = ""
    memory_text = ""

    # 1. 创建运行记录与 run_started 事件；画像/记忆读取失败不影响运行
    async with AsyncSessionFactory() as session:
        async with session.begin():
            if user_id is not None:
                try:
                    profile_text = await UserProfileService(session).get_render_text(user_id)
                except Exception as exc:
                    logger.warning(f"读取用户画像失败，本次运行不注入画像: {exc}")

                try:
                    memory_text = await UserMemoryService(session).get_render_text(user_id)
                except Exception as exc:
                    logger.warning(f"读取用户记忆失败，本次运行不注入记忆: {exc}")

            session.add(
                AgentRun(
                    run_id=run_id,
                    thread_id=thread_id,
                    user_id=user_id,
                    status="running",
                    input_message=message,
                    replay_of_run_id=replay_of_run_id,
                    worker_id=WORKER_ID,
                    heartbeat_at=datetime.now(UTC),
                    cancel_requested=False,
                )
            )
            session.add(
                RunEvent(
                    run_id=run_id,
                    thread_id=thread_id,
                    seq=0,
                    type="run_started",
                    payload={
                        "run_id": run_id,
                        "replay_of_run_id": replay_of_run_id,
                        "resumed": resume,
                    },
                )
            )

    # 2. 后台执行，HTTP 连接断开不会终止运行
    task = asyncio.create_task(
        _execute_run(
            run_id=run_id,
            agent=agent,
            user_id=user_id,
            thread_id=thread_id,
            message=message,
            profile_text=profile_text,
            memory_text=memory_text,
            resume=resume,
        ),
        name=f"agent-run-{run_id}",
    )
    _active_tasks[run_id] = task
    task.add_done_callback(lambda _: _active_tasks.pop(run_id, None))

    logger.info(
        f"运行已创建: run_id={run_id}, thread_id={thread_id}, "
        f"resume={resume}, worker={WORKER_ID}"
    )
    return run_id


async def _execute_run(
        *,
        run_id: str,
        agent: CompiledStateGraph,
        user_id: int | None,
        thread_id: str,
        message: str,
        profile_text: str,
        memory_text: str,
        resume: bool = False,
) -> None:
    """执行一次运行并落库事件。

    Args:
        run_id: 运行 ID。
        agent: 主 Graph。
        user_id: 用户 ID。
        thread_id: 会话 ID。
        message: 用户消息。
        profile_text: 用户画像文本。
        memory_text: 用户情节记忆文本。
        resume: 是否从 checkpoint 续跑，不重复写入用户消息。
    """
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_profile_text": profile_text,
        "user_memory_text": memory_text,
    }
    # 写操作使用同步持久化，避免工具已执行但 checkpoint 未落盘；
    # 无 checkpointer 时不能设置，否则 LangGraph 内部同步写入路径会报错
    if getattr(agent, "checkpointer", None) is not None:
        configurable[CONFIG_KEY_DURABILITY] = settings.agent_run.durability
    config = {"configurable": configurable}

    seq = 1
    final_state: dict[str, Any] = {}
    tool_started_at: dict[str, float] = {}
    streamed_answer = ""

    # 续跑时不要再写入用户消息，否则 checkpoint 里会出现重复的 human 消息
    graph_input = (
        None
        if resume
        else {"messages": [{"role": "user", "content": message}]}
    )
    streamable_nodes = STREAMABLE_NODES
    if settings.agent_run.stream_before_output_guard:
        streamable_nodes = STREAMABLE_NODES | EARLY_STREAMABLE_NODES

    # 心跳同时负责跨 worker 取消：其他 worker 写 cancel_requested 后在这里响应
    owner_task = asyncio.current_task()
    heartbeat_task: asyncio.Task | None = None
    if owner_task is not None:
        heartbeat_task = asyncio.create_task(
            _run_heartbeat_loop(run_id, owner_task),
            name=f"agent-run-heartbeat-{run_id}",
        )

    try:
        async for event in agent.astream_events(
                graph_input,
                config,
                version="v2",
        ):
            kind = event.get("event")

            if kind == "on_tool_start":
                tool_run_id = str(event.get("run_id") or "")
                tool_started_at[tool_run_id] = time.monotonic()
                await _append_event(
                    run_id,
                    thread_id,
                    seq,
                    "tool_call_started",
                    {
                        "tool": event.get("name") or "unknown",
                        "args": _safe_tool_args(
                            (event.get("data") or {}).get("input")
                        ),
                        "tool_call_id": tool_run_id,
                    },
                )
                seq += 1

            elif kind == "on_tool_end":
                tool_run_id = str(event.get("run_id") or "")
                started_at = tool_started_at.pop(tool_run_id, None)
                await _append_event(
                    run_id,
                    thread_id,
                    seq,
                    "tool_call_completed",
                    {
                        "tool": event.get("name") or "unknown",
                        "tool_call_id": tool_run_id,
                        "duration_ms": (
                            int((time.monotonic() - started_at) * 1000)
                            if started_at is not None
                            else None
                        ),
                    },
                )
                seq += 1

            elif kind == "on_tool_error":
                error = (event.get("data") or {}).get("error")
                await _append_event(
                    run_id,
                    thread_id,
                    seq,
                    "tool_call_failed",
                    {
                        "tool": event.get("name") or "unknown",
                        "tool_call_id": str(event.get("run_id") or ""),
                        "error": str(error)[:200] if error else "工具执行失败",
                    },
                )
                seq += 1

            # 输出安全校验通过后再把最终回答落 answer_streamed；
            # guard_reply 是固定安全话术，同样通过 final_answer 输出
            elif (
                    kind == "on_chain_end"
                    and not streamed_answer
                    and event.get("name") in streamable_nodes
            ):
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict):
                    raw_answer = output.get("final_answer")
                    if not isinstance(raw_answer, str) or not raw_answer.strip():
                        raw_answer = _last_ai_text_or_none(output)
                    if isinstance(raw_answer, str) and raw_answer.strip():
                        streamed_answer = raw_answer
                        await _append_event(
                            run_id,
                            thread_id,
                            seq,
                            "answer_streamed",
                            {"answer": raw_answer},
                        )
                        seq += 1

            # 主 Graph 根事件：记录最终状态
            elif kind == "on_chain_end" and not event.get("parent_ids"):
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict):
                    final_state = output

    except asyncio.CancelledError:
        await _append_event(run_id, thread_id, seq, "run_cancelled", {})
        await _update_run_status(run_id, "cancelled", finished=True)
        raise

    except Exception as exc:
        logger.error(f"运行失败: run_id={run_id}, error={exc}")
        await _append_event(
            run_id,
            thread_id,
            seq,
            "answer_completed",
            {"answer": FALLBACK_ANSWER},
        )
        seq += 1
        await _append_event(
            run_id,
            thread_id,
            seq,
            "run_failed",
            {"error": str(exc)[:200]},
        )
        await _update_run_status(
            run_id,
            "failed",
            answer=FALLBACK_ANSWER,
            error_code="RUN_FAILED",
            error_message=str(exc)[:500],
            finished=True,
        )
        return

    finally:
        # 运行结束后停止心跳，避免后台任务泄漏
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

    # 3. 兜底读取最终状态
    if not final_state:
        try:
            snapshot = await agent.aget_state(config)
            final_state = snapshot.values or {}
        except Exception as exc:
            logger.warning(f"读取运行最终状态失败: run_id={run_id}, error={exc}")

    # 优先使用输出安全校验后的 final_answer，保证流式内容与最终回答一致
    answer = str(final_state.get("final_answer") or "") or _last_ai_text(final_state)
    handoff_requested = bool(final_state.get("handoff_requested"))
    workflow_failed = bool(final_state.get("workflow_failed"))
    workflow_error = str(final_state.get("workflow_error") or "").strip()
    input_blocked = bool(
        final_state.get("guard_stage") == "input"
        and final_state.get("guard_blocked")
    )

    # 4. 写入回答与结束事件，再更新运行状态
    await _append_event(
        run_id,
        thread_id,
        seq,
        "answer_completed",
        {"answer": answer},
    )
    seq += 1

    # 5. 工作流内部失败：按失败落库，让前端可以触发重试
    if workflow_failed:
        await _append_event(
            run_id,
            thread_id,
            seq,
            "run_failed",
            {
                "error": (workflow_error or "工作流执行失败")[:200],
                "error_code": "WORKFLOW_FAILED",
            },
        )
        await _update_run_status(
            run_id,
            "failed",
            answer=answer,
            error_code="WORKFLOW_FAILED",
            error_message=(workflow_error or "工作流执行失败")[:500],
            finished=True,
        )
        # 失败时仍尝试从用户消息抽取画像，但不把兜底话术当成助手回答
        if user_id is not None and not input_blocked:
            schedule_profile_extraction(user_id, thread_id, message, "")
        return

    if handoff_requested:
        await _append_event(run_id, thread_id, seq, "handoff", {})
        seq += 1

    await _append_event(
        run_id,
        thread_id,
        seq,
        "run_completed",
        {
            "run_id": run_id,
            "handoff_requested": handoff_requested,
        },
    )
    await _update_run_status(run_id, "completed", answer=answer, finished=True)

    # 6. 画像抽取在运行侧触发，连接断开也不影响；
    # 输入被安全校验拦截时跳过，避免从违规输入中抽取画像
    if (
            user_id is not None
            and not input_blocked
            and answer.strip()
            and answer != FALLBACK_ANSWER
    ):
        schedule_profile_extraction(user_id, thread_id, message, answer)


async def shutdown_runner(timeout: float = 5.0) -> None:
    """取消并等待运行中的后台任务。

    应用关闭时调用；被取消的运行会落库为 cancelled。

    Args:
        timeout: 最长等待秒数。
    """
    tasks = list(_active_tasks.values())
    if not tasks:
        return

    for task in tasks:
        task.cancel()

    await asyncio.wait(tasks, timeout=timeout)
    logger.info(f"运行任务已停止，剩余 {len(_active_tasks)} 个")
