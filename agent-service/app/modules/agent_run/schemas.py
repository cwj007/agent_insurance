"""Agent 运行管理响应结构。"""

from datetime import datetime

from pydantic import Field

from app.common.schemas import BaseSchema


class AgentRunResponse(BaseSchema):
    """一次 Agent 运行的响应。"""

    run_id: str = Field(description="运行 ID")
    thread_id: str = Field(description="会话 ID")
    user_id: int | None = Field(default=None, description="用户 ID")
    status: str = Field(description="状态")
    input_message: str = Field(description="触发运行的用户消息")
    answer: str | None = Field(default=None, description="最终回答")
    error_code: str | None = Field(default=None, description="失败错误码")
    error_message: str | None = Field(default=None, description="失败说明")
    replay_of_run_id: str | None = Field(
        default=None,
        description="重放来源运行 ID",
    )
    cancel_requested: bool = Field(
        default=False,
        description="是否已请求取消",
    )
    started_at: datetime = Field(description="开始时间")
    ended_at: datetime | None = Field(default=None, description="结束时间")


class RunIdResponse(BaseSchema):
    """创建、重放或重试运行后的响应。"""

    run_id: str = Field(description="运行 ID")
