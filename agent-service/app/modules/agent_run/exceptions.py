"""Agent 运行管理异常。"""

from app.core.exceptions import ApplicationError


class AgentRunNotFound(ApplicationError):
    """运行不存在或不属于当前用户。"""

    status_code: int = 400
    code: str = "AGENT_RUN_NOT_FOUND"
    message: str = "当前运行不存在或不属于当前用户"


class AgentRunConflict(ApplicationError):
    """运行状态不允许当前操作。"""

    status_code: int = 400
    code: str = "AGENT_RUN_CONFLICT"
    message: str = "当前运行状态不允许该操作"
