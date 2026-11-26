"""安全合规校验的数据结构。"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# 安全合规风险类别
GuardCategory = Literal[
    "none",  # 无风险
    "prompt_injection",  # 提示词注入、越狱、套取系统提示
    "illegal",  # 骗保、伪造材料、洗钱等违法请求
    "harmful",  # 暴力、色情、歧视、侮辱等有害内容
    "privacy",  # 他人隐私、保单、个人信息泄露
    "compliance",  # 保险销售合规风险（承诺收益、保证赔付等）
    "medical",  # 医疗诊断、处方等超范围建议
    "other",  # 其他无法归类的问题
]

# 校验发生在输入侧还是输出侧
GuardStage = Literal["input", "output"]


class GuardDecision(BaseModel):
    """一轮安全合规校验结果。

    只描述是否放行和风险类别，不生成用户可见内容；
    拦截话术由代码中的固定模板提供，避免校验模型本身被诱导。

    Attributes:
        allowed: 是否允许放行。
        category: 风险类别，放行时为 none。
        reason: 一句话理由，仅用于日志和排查。
    """

    allowed: bool = Field(description="是否允许放行")
    category: GuardCategory = Field(description="风险类别；放行时必须为 none")
    reason: str = Field(
        max_length=100,
        description="一句话判断理由，仅用于日志和排查",
    )

    @model_validator(mode="after")
    def _check_category(self) -> "GuardDecision":
        """校验放行标记与风险类别是否自洽。

        Returns:
            校验通过的实例自身。

        Raises:
            ValueError: allowed 与 category 不一致时抛出。
        """
        if self.allowed and self.category != "none":
            raise ValueError("允许放行时 category 必须为 none")
        if not self.allowed and self.category == "none":
            raise ValueError("拦截时 category 不能为 none")
        return self
