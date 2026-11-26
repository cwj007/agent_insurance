"""聊天模块路由。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.modules.agent_run.schemas import AgentRunResponse, RunIdResponse
from app.modules.chat.schema import ChatRequest
from app.modules.chat.service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["与AI聊天对话"])


def get_chat_service(
        http_request: Request,  # 使用 fastapi 的 request 对象
        session: AsyncSession = Depends(get_session),  # 2. 缺少 session，需要依赖注入
):
    """获取 ChatService 实例。

    Args:
        http_request: FastAPI 请求对象，用于从 app.state 中获取 agent。
        session: 数据库异步会话，由 FastAPI 依赖注入。

    Returns:
        绑定当前请求会话与 agent 的 ChatService 实例。
    """
    agent = http_request.app.state.agent  # 3. 缺少 agent，需要用 fastapi 中的 state.agent
    return ChatService(session, agent)


# SSE 流式对话接口
@router.post("", response_class=EventSourceResponse)
async def chat_stream(
        request: ChatRequest,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        service: ChatService = Depends(get_chat_service),  # 1. 需要先调用 service
):
    """实现 Agent 流式对话。

    Args:
        request: 聊天请求体，包含会话 ID 和用户消息。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Yields:
        流式返回的 SSE 事件，文本片段事件为 message，结束事件为 done。
    """
    # 调用 ChatService 实例的 chat_stream 方法
    async for sse in service.chat_stream(user_id, request):
        yield sse # 这里是返回给前端的 ServerSentEvent


@router.get(
    "/runs/{run_id}",
    response_model=AgentRunResponse,
    summary="查询运行状态",
)
async def get_agent_run(
        run_id: str,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        service: ChatService = Depends(get_chat_service),
):
    """查询一次 Agent 运行的状态与结果。

    Args:
        run_id: 运行 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Returns:
        运行详情。
    """
    return await service.get_run(user_id, run_id)


@router.get(
    "/runs/{run_id}/events",
    response_class=EventSourceResponse,
    summary="订阅运行事件",
)
async def stream_agent_run_events(
        run_id: str,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        after: str | None = Query(
            default=None,
            description="最后收到的事件 ID，格式为 run_id:seq",
        ),
        last_event_id: Annotated[
            str | None,
            Header(alias="Last-Event-ID"),
        ] = None,
        service: ChatService = Depends(get_chat_service),
):
    """订阅指定运行的事件，用于断线重连。

    Args:
        run_id: 运行 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        after: 查询参数形式的最后事件 ID。
        last_event_id: 请求头 Last-Event-ID。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Yields:
        包括已发生事件在内的 SSE 事件流，运行结束后关闭。
    """
    async for sse in service.stream_run_events(
            user_id,
            run_id,
            last_event_id or after,
    ):
        yield sse


@router.post("/runs/{run_id}/cancel", summary="取消运行")
async def cancel_agent_run(
        run_id: str,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        service: ChatService = Depends(get_chat_service),
):
    """取消正在执行的运行。

    Args:
        run_id: 运行 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Returns:
        是否找到并取消了活动任务。
    """
    cancelled = await service.cancel_run(user_id, run_id)
    return {"cancelled": cancelled}


@router.post(
    "/runs/{run_id}/replay",
    response_model=RunIdResponse,
    summary="重放运行",
)
async def replay_agent_run(
        run_id: str,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        service: ChatService = Depends(get_chat_service),
):
    """基于原运行的用户消息重新执行一次。

    Args:
        run_id: 原运行 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Returns:
        新运行 ID。
    """
    new_run_id = await service.replay_run(user_id, run_id)
    return RunIdResponse(run_id=new_run_id)


@router.post(
    "/runs/{run_id}/retry",
    response_model=RunIdResponse,
    summary="重试运行",
)
async def retry_agent_run(
        run_id: str,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        service: ChatService = Depends(get_chat_service),
):
    """重试失败或取消的运行。

    Args:
        run_id: 原运行 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Returns:
        新运行 ID。
    """
    new_run_id = await service.retry_run(user_id, run_id)
    return RunIdResponse(run_id=new_run_id)


@router.get("/{thread_id}/events", summary="回放运行事件")
async def replay_run_events(
        thread_id: UUID,
        user_id: Annotated[int | None, Header(alias="x-user-id")],
        after: str | None = Query(
            default=None,
            description="最后收到的事件 ID，格式为 run_id:seq",
        ),
        service: ChatService = Depends(get_chat_service),
):
    """按 Last-Event-ID 回放运行事件，供断线重连使用。

    Args:
        thread_id: 会话 ID（路径参数）。
        user_id: 请求头 x-user-id 中的用户 ID。
        after: 客户端最后收到的事件 ID，格式为 run_id:seq。
        service: 聊天服务实例，由 FastAPI 依赖注入。

    Returns:
        按序号升序排列的事件列表。
    """
    return await service.replay_events(user_id, thread_id, after)
