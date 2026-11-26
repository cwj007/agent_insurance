"""用户画像模块数据结构。"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.common.schemas import BaseSchema

# 画像字段分组：组名 -> 字段名，顺序决定渲染顺序
PROFILE_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "basic": ("age", "gender", "city_tier", "occupation", "social_insurance"),
    "family": ("marital_status", "children_count", "parents_supported"),
    "health": ("health_conditions", "smoking", "drinking"),
    "finance": ("annual_income_range", "budget"),
    "needs": (
        "target_categories",
        "concerns",
        "recommended_products",
        "plan_summary",
    ),
    "preferences": ("price_sensitivity", "brand_preference"),
}

# 分组的中文展示名
PROFILE_GROUP_LABELS: dict[str, str] = {
    "basic": "基本信息",
    "family": "家庭情况",
    "health": "健康情况",
    "finance": "财务状况",
    "needs": "保障需求",
    "preferences": "偏好",
}

# 字段的中文展示名
PROFILE_FIELD_LABELS: dict[str, str] = {
    "age": "年龄",
    "gender": "性别",
    "city_tier": "城市级别",
    "occupation": "职业",
    "social_insurance": "社保",
    "marital_status": "婚姻状况",
    "children_count": "子女数量",
    "parents_supported": "赡养父母",
    "health_conditions": "健康状况",
    "smoking": "吸烟",
    "drinking": "饮酒",
    "annual_income_range": "年收入范围",
    "budget": "保费预算",
    "target_categories": "意向险种",
    "concerns": "关注点",
    "recommended_products": "已推荐产品",
    "plan_summary": "方案概述",
    "price_sensitivity": "价格敏感度",
    "brand_preference": "品牌偏好",
}

# 所有合法字段名
ALL_PROFILE_FIELDS: frozenset[str] = frozenset(PROFILE_FIELD_LABELS)

# 敏感字段：只接受用户显式陈述，禁止模型推断
SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {"health_conditions", "smoking", "drinking"}
)

# 枚举值的中文展示
PROFILE_VALUE_LABELS: dict[str, dict[str, str]] = {
    "gender": {"male": "男", "female": "女", "other": "其他"},
    "marital_status": {
        "single": "未婚",
        "married": "已婚",
        "divorced": "离异",
        "widowed": "丧偶",
    },
    "target_categories": {
        "medical": "医疗险",
        "critical_illness": "重疾险",
        "life": "寿险",
        "accident": "意外险",
    },
}

# 字段来源：用户明确陈述 / 模型推断 / 用户手动更正
FieldSource = Literal["stated", "inferred", "corrected"]

# 画像字段值的类型
ProfileValue = bool | str | int | float | list[str]


class ExtractedField(BaseModel):
    """从单轮对话中抽取到的一个画像字段。

    Attributes:
        value: 字段值。
        source: 来源；stated 表示用户明确说出，inferred 表示模型合理推断。
        confidence: 置信度，0~1。
    """

    value: ProfileValue = Field(description="字段值")
    source: Literal["stated", "inferred"] = Field(
        description="stated 表示用户明确说出，inferred 表示模型推断",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="置信度，0~1",
    )


class ExtractedEpisode(BaseModel):
    """一条值得跨会话记住的情节记忆。

    Attributes:
        kind: 记忆类型。
        content: 一句话内容。
        importance: 重要度，0~1。
    """

    kind: Literal["fact", "preference", "event", "decision"] = Field(
        description="记忆类型",
    )
    content: str = Field(description="一句话内容，具体、可复用")
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="重要度，0~1",
    )


class ExtractedProfile(BaseModel):
    """单轮对话的画像抽取结果。

    只包含本轮新出现或发生变化的信息，未提及的字段保持为 None。

    Attributes:
        age: 年龄，整数。
        gender: 性别，male/female/other。
        city_tier: 城市级别，一线/二线/三线/四线及以下。
        occupation: 职业。
        social_insurance: 是否有社保。
        marital_status: 婚姻状况，single/married/divorced/widowed。
        children_count: 子女数量，整数。
        parents_supported: 是否赡养父母。
        health_conditions: 健康状况或既往病史列表。
        smoking: 是否吸烟。
        drinking: 是否饮酒。
        annual_income_range: 年收入范围，带单位。
        budget: 年保费预算，带单位。
        target_categories: 意向险种列表，取值 medical/critical_illness/life/accident。
        concerns: 关注点列表。
        recommended_products: 已推荐产品列表，每项含险种和产品名。
        plan_summary: 当前推荐方案概述。
        price_sensitivity: 价格敏感度，高/中/低。
        brand_preference: 品牌偏好。
        episodes: 本轮值得跨会话记住的关键情节，1~3 条。
    """

    age: ExtractedField | None = Field(default=None, description="年龄，整数")
    gender: ExtractedField | None = Field(
        default=None,
        description="性别，male/female/other",
    )
    city_tier: ExtractedField | None = Field(
        default=None,
        description="城市级别，一线/二线/三线/四线及以下",
    )
    occupation: ExtractedField | None = Field(default=None, description="职业")
    social_insurance: ExtractedField | None = Field(
        default=None,
        description="是否有社保，布尔值",
    )
    marital_status: ExtractedField | None = Field(
        default=None,
        description="婚姻状况，single/married/divorced/widowed",
    )
    children_count: ExtractedField | None = Field(
        default=None,
        description="子女数量，整数",
    )
    parents_supported: ExtractedField | None = Field(
        default=None,
        description="是否赡养父母，布尔值",
    )
    health_conditions: ExtractedField | None = Field(
        default=None,
        description="健康状况或既往病史列表",
    )
    smoking: ExtractedField | None = Field(
        default=None,
        description="是否吸烟，布尔值",
    )
    drinking: ExtractedField | None = Field(
        default=None,
        description="是否饮酒，布尔值",
    )
    annual_income_range: ExtractedField | None = Field(
        default=None,
        description="年收入范围，带单位，例如 20-30 万元",
    )
    budget: ExtractedField | None = Field(
        default=None,
        description="年保费预算，带单位，例如 3000 元/年",
    )
    target_categories: ExtractedField | None = Field(
        default=None,
        description="意向险种列表，取值 medical/critical_illness/life/accident",
    )
    concerns: ExtractedField | None = Field(
        default=None,
        description="关注点列表，例如 重疾保障、保费价格",
    )
    recommended_products: ExtractedField | None = Field(
        default=None,
        description=(
            "已推荐产品列表，每项格式为「险种：产品名」，"
            "例如 医疗险：众安尊享e生百万医疗险2026版"
        ),
    )
    plan_summary: ExtractedField | None = Field(
        default=None,
        description="当前推荐方案概述，一句话说明组合和理由",
    )
    price_sensitivity: ExtractedField | None = Field(
        default=None,
        description="价格敏感度，高/中/低",
    )
    brand_preference: ExtractedField | None = Field(
        default=None,
        description="品牌偏好",
    )
    episodes: list[ExtractedEpisode] = Field(
        default_factory=list,
        description="本轮值得跨会话记住的关键情节，1~3 条",
    )


class ProfileField(BaseModel):
    """存储中的画像字段，带来源与时间等元信息。

    Attributes:
        value: 字段值。
        source: 字段来源。
        confidence: 置信度，0~1。
        updated_at: 最近一次确认或更新的时间。
        needs_confirmation: 是否存在冲突、需要向用户确认。
        previous_value: 冲突时的旧值，便于追溯。
    """

    value: ProfileValue = Field(description="字段值")
    source: FieldSource = Field(description="字段来源")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="置信度")
    updated_at: datetime = Field(description="最近一次确认或更新时间")
    needs_confirmation: bool = Field(
        default=False,
        description="是否存在冲突、需要向用户确认",
    )
    previous_value: ProfileValue | None = Field(
        default=None,
        description="冲突时的旧值",
    )


class UserProfileResponse(BaseSchema):
    """用户画像响应。

    Attributes:
        user_id: 用户 ID。
        version: 画像版本号；从未生成过画像时为 0。
        fields: 字段名到画像字段的映射。
        updated_at: 最近更新时间；从未生成过画像时为 None。
    """

    user_id: int = Field(description="用户 ID")
    version: int = Field(default=0, description="画像版本号")
    fields: dict[str, ProfileField] = Field(
        default_factory=dict,
        description="字段名到画像字段的映射",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="最近更新时间",
    )


class ProfileCorrectionRequest(BaseSchema):
    """用户手动更正画像的请求体。

    Attributes:
        fields: 字段名到新值的映射；值为 null 表示删除该字段。
    """

    fields: dict[str, Any] = Field(
        description="字段名到新值的映射；值为 null 表示删除该字段",
    )
