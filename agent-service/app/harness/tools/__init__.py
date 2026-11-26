"""Harness 工具装配：具体业务工具 + 调用记录中间件。"""

from typing import Any

from app.agents.tools.contracts import WorkflowTools
from app.agents.tools.middleware import PER_TOOL_RUN_LIMIT, build_tool_middlewares
from app.harness.tools.clause import retrieve_insurance_clause
from app.harness.tools.product import query_candidate_products
from app.harness.tools.recording import build_recording_middleware


def build_workflow_tools() -> WorkflowTools:
    """构建各工作流可用的具体工具集合。

    Returns:
        推荐与理赔工作流的工具集合。
    """
    return WorkflowTools(
        recommendation=[query_candidate_products, retrieve_insurance_clause],
        claim=[retrieve_insurance_clause],
    )


def build_agent_tool_middlewares() -> list[Any]:
    """构建完整工具中间件列表：记录在最外层，通用中间件在内层。

    Returns:
        可直接传给 create_agent 的 middleware 列表。
    """
    return [
        build_recording_middleware(),
        *build_tool_middlewares({
            "query_candidate_products": PER_TOOL_RUN_LIMIT,
            "retrieve_insurance_clause": PER_TOOL_RUN_LIMIT,
        }),
    ]


__all__ = [
    "build_agent_tool_middlewares",
    "build_workflow_tools",
    "query_candidate_products",
    "retrieve_insurance_clause",
]
