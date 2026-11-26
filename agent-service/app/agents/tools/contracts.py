"""工作流工具契约：由 harness 注入具体工具，agents 不直接依赖业务模块。"""

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool


@dataclass(slots=True)
class WorkflowTools:
    """各工作流可用的工具集合。

    Attributes:
        recommendation: 推荐工作流可用的工具。
        claim: 理赔工作流可用的工具。
    """

    recommendation: list[BaseTool] = field(default_factory=list)
    claim: list[BaseTool] = field(default_factory=list)
