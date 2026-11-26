"""运行恢复：取消、重试、重放与续跑判断。"""

from typing import Any
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.harness.runner import _message_text, start_run
from app.harness.state import WORKER_ID, _active_tasks
from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.agent_run.exceptions import AgentRunConflict, AgentRunNotFound
from app.modules.agent_run.repository import AgentRunRepository
from app.modules.chat_threads.models import ChatThread
from app.modules.chat_threads.repository import ChatThreadRepository

logger = get_logger(__name__)


async def _can_resume_run(agent: CompiledStateGraph, thread_id: str) -> bool:
    """判断失败运行能否从 checkpoint 续跑。

    Args:
        agent: 主 Graph。
        thread_id: 会话 ID。

    Returns:
        存在待执行节点时返回 True；没有 checkpointer 或读取失败时返回 False。
    """
    if getattr(agent, "checkpointer", None) is None:
        return False

    try:
        snapshot = await agent.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
    except Exception as exc:
        logger.warning(
            f"读取重试 checkpoint 失败: thread_id={thread_id}, error={exc}"
        )
        return False

    return bool(snapshot and snapshot.next)


async def cancel_run(run_id: str) -> bool:
    """取消正在执行的运行，支持跨 worker 请求。

    先写数据库取消标记，再取消本进程任务；
    其他 worker 上的运行会由心跳循环轮询到标记后自行取消。

    Args:
        run_id: 运行 ID。

    Returns:
        运行处于可取消状态或本进程存在活动任务时返回 True。
    """
    # 1. 写取消标记，远端 worker 会轮询到
    requested = False
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                requested = await AgentRunRepository(session).request_cancel(run_id)
    except Exception as exc:
        logger.warning(f"写入取消请求失败: run_id={run_id}, error={exc}")

    # 2. 本进程持有任务时立即取消
    task = _active_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
        logger.info(f"运行已请求取消（本 worker）: run_id={run_id}")
        return True

    if requested:
        logger.info(f"运行已请求取消（等待所属 worker 响应）: run_id={run_id}")
    return requested


async def _load_replay_context(
        agent: CompiledStateGraph,
        thread_id: str,
        input_message: str,
) -> dict[str, Any]:
    """读取原会话中目标用户消息之前的上下文。

    Args:
        agent: 主 Graph。
        thread_id: 原会话 ID。
        input_message: 原运行的用户消息。

    Returns:
        可直接写入新会话的状态片段；无法定位时返回空字典。
    """
    if getattr(agent, "checkpointer", None) is None:
        return {}

    config = {"configurable": {"thread_id": thread_id}}
    try:
        history = [
            snapshot
            async for snapshot in agent.aget_state_history(config)
        ]
    except Exception as exc:
        logger.warning(
            f"读取重放上下文失败: thread_id={thread_id}, error={exc}"
        )
        return {}

    # 从旧到新找第一条包含原用户消息的快照；
    # 不要求该消息是最后一条，因为 checkpoint 可能已经包含本轮回答
    matched = None
    target_index = -1
    target = input_message.strip()
    for snapshot in reversed(history):
        messages = snapshot.values.get("messages", [])
        for index, message in enumerate(messages):
            if (
                    isinstance(message, HumanMessage)
                    and _message_text(message).strip() == target
            ):
                matched = snapshot
                target_index = index
                break
        if matched is not None:
            break

    if matched is None:
        logger.warning(
            f"重放未找到原始用户消息对应的 checkpoint: thread_id={thread_id}"
        )
        return {}

    messages = list(matched.values.get("messages", []))
    context: dict[str, Any] = {}

    # 1. 只保留原用户消息之前的消息，本轮回答不进入新会话
    prior_messages = messages[:target_index]
    if prior_messages:
        context["messages"] = prior_messages

    # 2. 摘要和主流程意图也要带过去，避免新会话丢失背景
    summary = str(matched.values.get("summary") or "").strip()
    if summary:
        context["summary"] = summary

    active_workflow = matched.values.get("active_workflow")
    if active_workflow:
        context["active_workflow"] = active_workflow

    previous_workflow = matched.values.get("previous_workflow")
    if previous_workflow:
        context["previous_workflow"] = previous_workflow

    return context


async def replay_run(
        agent: CompiledStateGraph,
        run_id: str,
        user_id: int,
) -> str:
    """基于原运行的用户消息，在新会话中创建一次运行。

    原会话保持不变；新会话会复制原运行之前的上下文，
    因此重放不会在原会话里追加重复问答。

    Args:
        agent: 主 Graph。
        run_id: 原运行 ID。
        user_id: 当前用户 ID。

    Returns:
        新运行 ID。

    Raises:
        AgentRunNotFound: 运行不存在或不属于当前用户时抛出。
        AgentRunConflict: 运行尚未结束或原运行无法重放时抛出。
    """
    async with AsyncSessionFactory() as session:
        run = await AgentRunRepository(session).get_for_user(run_id, user_id)

    if run is None:
        raise AgentRunNotFound()
    if run.status in ("pending", "running"):
        raise AgentRunConflict()

    # 1. 为新会话生成独立 thread_id，避免污染原会话
    new_thread_id = str(uuid4())
    if run.user_id is not None:
        try:
            async with AsyncSessionFactory() as session:
                async with session.begin():
                    original = await ChatThreadRepository(session).get(
                        run.thread_id
                    )
                    title = (
                        f"{original.title if original else '会话'}（重放）"
                    )[:200]
                    session.add(
                        ChatThread(
                            id=UUID(new_thread_id),
                            user_id=run.user_id,
                            title=title,
                        )
                    )
        except Exception as exc:
            logger.warning(
                f"创建重放会话失败，仍按独立 thread 执行: {exc}"
            )

    # 2. 复制原运行之前的历史上下文；失败时退化为只有本轮消息
    try:
        context = await _load_replay_context(
            agent,
            run.thread_id,
            run.input_message,
        )
        if context:
            await agent.aupdate_state(
                {"configurable": {"thread_id": new_thread_id}},
                context,
            )
    except Exception as exc:
        logger.warning(f"复制重放上下文失败，按空上下文执行: {exc}")

    logger.info(
        f"重放运行: source_run_id={run_id}, new_thread_id={new_thread_id}, "
        f"worker={WORKER_ID}"
    )
    return await start_run(
        agent,
        user_id=run.user_id,
        thread_id=new_thread_id,
        message=run.input_message,
        replay_of_run_id=run_id,
    )


async def retry_run(
        agent: CompiledStateGraph,
        run_id: str,
        user_id: int,
) -> str:
    """重试失败或取消的运行。

    Args:
        agent: 主 Graph。
        run_id: 原运行 ID。
        user_id: 当前用户 ID。

    Returns:
        新运行 ID。

    Raises:
        AgentRunNotFound: 运行不存在或不属于当前用户时抛出。
        AgentRunConflict: 运行状态不是失败或取消时抛出。
    """
    async with AsyncSessionFactory() as session:
        run = await AgentRunRepository(session).get_for_user(run_id, user_id)

    if run is None:
        raise AgentRunNotFound()
    if run.status not in ("failed", "cancelled"):
        raise AgentRunConflict()

    # 优先从失败节点续跑，避免把同一条用户消息重复写入 checkpoint；
    # 没有待执行节点时再回退到重新发送消息
    resume = await _can_resume_run(agent, run.thread_id)
    logger.info(f"重试运行: run_id={run_id}, resume={resume}")
    return await start_run(
        agent,
        user_id=run.user_id,
        thread_id=run.thread_id,
        message=run.input_message,
        replay_of_run_id=run_id,
        resume=resume,
    )
