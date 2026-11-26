"""用户画像服务。"""

import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.modules.user_memory.repository import UserMemoryRepository
from app.modules.user_profile.exceptions import (
    ProfileFieldInvalid,
    ProfileValueInvalid,
)
from app.modules.user_profile.models import UserProfile, UserProfileEvent
from app.modules.user_profile.repository import UserProfileRepository
from app.modules.user_profile.schemas import (
    ALL_PROFILE_FIELDS,
    PROFILE_FIELD_GROUPS,
    PROFILE_FIELD_LABELS,
    PROFILE_GROUP_LABELS,
    PROFILE_VALUE_LABELS,
    SENSITIVE_FIELDS,
    ProfileField,
    UserProfileResponse,
)

logger = get_logger(__name__)

# 字段展示时追加的单位
_FIELD_UNITS: dict[str, str] = {
    "age": "岁",
    "children_count": "个",
}

# 字段来源的展示文案
_SOURCE_LABELS: dict[str, str] = {
    "stated": "用户自述",
    "inferred": "模型推断",
    "corrected": "用户更正",
}

# 需要按列表处理的字段
_LIST_FIELDS: frozenset[str] = frozenset(
    {"health_conditions", "concerns", "target_categories", "recommended_products"}
)

# 布尔字段的文本映射
_BOOL_TRUE = frozenset({"true", "是", "有", "yes", "1", "已", "需要"})
_BOOL_FALSE = frozenset({"false", "否", "无", "no", "0", "没有", "不需要"})


def render_profile_text(fields: dict[str, Any]) -> str:
    """把画像字段渲染成可注入 system prompt 的文本。

    Args:
        fields: 字段名到画像字段字典的映射。

    Returns:
        渲染后的画像文本；没有任何有效字段时返回空字符串。
    """
    group_blocks: list[str] = []

    for group, names in PROFILE_FIELD_GROUPS.items():
        lines: list[str] = []
        for name in names:
            raw = fields.get(name)
            if not raw:
                continue
            try:
                field = ProfileField.model_validate(raw)
            except Exception:
                logger.warning(f"用户画像字段解析失败，已跳过: field={name}")
                continue

            value_text = _format_value(name, field.value)
            if not value_text:
                continue

            source_label = _SOURCE_LABELS.get(field.source, field.source)
            date_text = field.updated_at.strftime("%Y-%m-%d")
            extra = "；待确认" if field.needs_confirmation else ""
            lines.append(
                f"- {PROFILE_FIELD_LABELS[name]}：{value_text}"
                f"（{source_label}，{date_text}{extra}）"
            )

        if lines:
            group_blocks.append(
                f"### {PROFILE_GROUP_LABELS[group]}\n" + "\n".join(lines)
            )

    if not group_blocks:
        return ""

    return "\n".join([
        "## 用户档案（来自历史对话，仅供个性化参考）",
        *group_blocks,
        "使用要求：不要主动复述用户的敏感信息；"
        "与用户当前说法冲突时，以当前说法为准；标注「待确认」的信息需要先向用户确认。",
    ])


def _format_value(name: str, value: Any) -> str:
    """把单个画像字段值格式化为中文展示文本。

    Args:
        name: 字段名。
        value: 字段值。

    Returns:
        格式化后的文本；空值时返回空字符串。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, list):
        labels = PROFILE_VALUE_LABELS.get(name, {})
        parts = [labels.get(str(item), str(item)) for item in value]
        return "、".join(part for part in parts if part)

    text = str(value).strip()
    if not text:
        return ""

    labels = PROFILE_VALUE_LABELS.get(name)
    if labels:
        text = labels.get(text, text)

    unit = _FIELD_UNITS.get(name, "")
    if unit and text.isdigit():
        return f"{text}{unit}"
    return text


def _normalize_value(name: str, value: Any) -> Any:
    """校验并规范化画像字段值。

    Args:
        name: 字段名。
        value: 待规范化的原始值。

    Returns:
        规范化后的值；值无效时返回 None。
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    if name == "age":
        return _to_int(value, minimum=0, maximum=120)
    if name == "children_count":
        return _to_int(value, minimum=0, maximum=20)
    if name in ("parents_supported", "social_insurance"):
        return _to_bool(value)
    if name in _LIST_FIELDS:
        items = value if isinstance(value, list) else re.split(r"[、,，;；/]", str(value))
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return cleaned or None
    if isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _to_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    """把值转换为区间内的整数。

    Args:
        value: 待转换的值。
        minimum: 允许的最小值。
        maximum: 允许的最大值。

    Returns:
        转换成功时返回整数；超出区间或无法转换时返回 None。
    """
    if isinstance(value, bool):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    if minimum <= number <= maximum:
        return number
    return None


def _to_bool(value: Any) -> bool | None:
    """把值转换为布尔值。

    Args:
        value: 待转换的值。

    Returns:
        转换成功时返回布尔值；无法识别时返回 None。
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    return None


def _merge_field(
        name: str,
        old: dict[str, Any] | None,
        value: Any,
        source: str,
        confidence: float,
        now: datetime,
) -> dict[str, Any] | None:
    """按合并规则决定单个字段是否更新。

    Args:
        name: 字段名。
        old: 已有字段字典；没有时为 None。
        value: 新字段值。
        source: 新值来源，stated 或 inferred。
        confidence: 新值置信度。
        now: 本次合并时间。

    Returns:
        合并后的字段字典；不应更新时返回 None。
    """
    # 模型推断的置信度封顶，避免过度自信
    if source == "inferred":
        confidence = min(confidence, 0.7)

    new_field = {
        "value": value,
        "source": source,
        "confidence": round(confidence, 2),
        "updated_at": now.isoformat(),
        "needs_confirmation": False,
        "previous_value": None,
    }

    # 1. 用户手动更正过的字段，自动抽取不再覆盖
    if old and old.get("source") == "corrected":
        return None

    # 2. 已推荐产品是系统累积信息：新旧列表合并去重，而不是覆盖
    if name == "recommended_products":
        old_value = (old or {}).get("value")
        old_items = old_value if isinstance(old_value, list) else []
        new_items = value if isinstance(value, list) else [value]
        merged_items = list(dict.fromkeys([*old_items, *new_items]))[:10]
        if old_value == merged_items:
            return None
        new_field["value"] = merged_items
        new_field["source"] = "inferred"
        new_field["needs_confirmation"] = False
        return new_field

    if not old:
        return new_field

    old_source = old.get("source")

    # 3. 方案概述每轮更新即可，不做冲突确认
    if name == "plan_summary":
        if old.get("value") == value:
            return None
        return new_field

    # 4. 已有用户自述信息时，模型推断不能覆盖
    if source == "inferred" and old_source == "stated":
        return None

    # 5. 值没有变化：通常保留原记录，不刷新「用户自述时间」；
    # 但如果来源从推断升级为用户自述，或者用户再次确认了该值，
    # 则更新来源并清除待确认标记
    if old.get("value") == value:
        if source == "stated" and (
                old_source == "inferred" or old.get("needs_confirmation")
        ):
            return new_field
        return None

    # 6. 值冲突：新值生效，但标记待确认并保留旧值
    new_field["needs_confirmation"] = True
    new_field["previous_value"] = old.get("value")
    return new_field


class UserProfileService:
    """用户画像服务。"""

    def __init__(self, session: AsyncSession):
        """初始化用户画像服务。

        Args:
            session: 数据库异步会话。
        """
        self.session = session
        self.repository = UserProfileRepository(session)

    async def get_render_text(self, user_id: int) -> str:
        """读取并渲染用户画像文本，供 system prompt 注入。

        Args:
            user_id: 用户 ID。

        Returns:
            渲染后的画像文本；用户没有画像时返回空字符串。
        """
        record = await self.repository.get(user_id)
        if record is None:
            return ""
        return render_profile_text((record.profile or {}).get("fields", {}))

    async def get_response(self, user_id: int) -> UserProfileResponse:
        """查询用户画像响应。

        Args:
            user_id: 用户 ID。

        Returns:
            画像响应；用户没有画像时返回空结构。
        """
        record = await self.repository.get(user_id)
        if record is None:
            return UserProfileResponse(user_id=user_id)

        fields: dict[str, ProfileField] = {}
        raw_fields = (record.profile or {}).get("fields", {}) or {}
        for name, raw in raw_fields.items():
            if name not in ALL_PROFILE_FIELDS:
                continue
            try:
                fields[name] = ProfileField.model_validate(raw)
            except Exception:
                logger.warning(f"用户画像字段解析失败，已跳过: field={name}")

        return UserProfileResponse(
            user_id=user_id,
            version=record.version or 0,
            fields=fields,
            updated_at=record.updated_at,
        )

    async def merge_extraction(
            self,
            user_id: int,
            extracted: dict[str, dict[str, Any]],
            *,
            thread_id: str | None = None,
    ) -> bool:
        """合并一轮对话抽取出的画像字段。

        Args:
            user_id: 用户 ID。
            extracted: 字段名到抽取结果字典的映射。
            thread_id: 来源会话 ID，便于追溯。

        Returns:
            画像是否发生变更。
        """
        async with self.session.begin():
            record = await self.repository.get(user_id, for_update=True)
            if record is None:
                record = UserProfile(
                    user_id=user_id,
                    profile={"fields": {}},
                    version=0,
                )
                self.session.add(record)
                await self.session.flush()

            fields = dict((record.profile or {}).get("fields", {}))
            now = datetime.now(UTC)
            patch: dict[str, Any] = {}

            for name, data in extracted.items():
                if name not in ALL_PROFILE_FIELDS or not isinstance(data, dict):
                    continue

                source = data.get("source") or "inferred"
                if source not in ("stated", "inferred"):
                    source = "inferred"

                # 敏感字段只接受用户显式陈述，禁止模型推断
                if name in SENSITIVE_FIELDS and source != "stated":
                    continue

                value = _normalize_value(name, data.get("value"))
                if value is None:
                    continue

                merged = _merge_field(
                    name,
                    fields.get(name),
                    value,
                    source,
                    float(data.get("confidence", 0.8)),
                    now,
                )
                if merged is None:
                    continue

                patch[name] = {"before": fields.get(name), "after": merged}
                fields[name] = merged

            if not patch:
                return False

            record.profile = {"fields": fields}
            record.version = (record.version or 0) + 1
            await self.repository.add_event(
                UserProfileEvent(
                    user_id=user_id,
                    patch=patch,
                    reason="extraction",
                    source_thread_id=thread_id,
                )
            )

        logger.info(f"用户画像已更新: user_id={user_id}, fields={list(patch)}~☑️")
        return True

    async def apply_correction(
            self,
            user_id: int,
            values: dict[str, Any],
    ) -> UserProfileResponse:
        """应用用户手动更正。

        Args:
            user_id: 用户 ID。
            values: 字段名到新值的映射；值为 None 表示删除该字段。

        Returns:
            更新后的画像。

        Raises:
            ProfileFieldInvalid: 包含白名单外的字段名时抛出。
            ProfileValueInvalid: 字段值格式不合法时抛出。
        """
        invalid_fields = [name for name in values if name not in ALL_PROFILE_FIELDS]
        if invalid_fields:
            logger.warning(f"用户画像更正包含不支持字段: {invalid_fields}")
            raise ProfileFieldInvalid()

        async with self.session.begin():
            record = await self.repository.get(user_id, for_update=True)
            if record is None:
                record = UserProfile(
                    user_id=user_id,
                    profile={"fields": {}},
                    version=0,
                )
                self.session.add(record)
                await self.session.flush()

            fields = dict((record.profile or {}).get("fields", {}))
            now = datetime.now(UTC)
            patch: dict[str, Any] = {}

            for name, raw in values.items():
                if raw is None:
                    if name in fields:
                        patch[name] = {"before": fields.pop(name), "after": None}
                    continue

                value = _normalize_value(name, raw)
                if value is None:
                    logger.warning(f"用户画像更正值无效: field={name}")
                    raise ProfileValueInvalid()

                field = {
                    "value": value,
                    "source": "corrected",
                    "confidence": 1.0,
                    "updated_at": now.isoformat(),
                    "needs_confirmation": False,
                    "previous_value": None,
                }
                patch[name] = {"before": fields.get(name), "after": field}
                fields[name] = field

            if patch:
                record.profile = {"fields": fields}
                record.version = (record.version or 0) + 1
                await self.repository.add_event(
                    UserProfileEvent(
                        user_id=user_id,
                        patch=patch,
                        reason="correction",
                    )
                )

        return await self.get_response(user_id)

    async def delete(self, user_id: int) -> None:
        """删除用户画像、变更事件和情节记忆。

        Args:
            user_id: 用户 ID。
        """
        async with self.session.begin():
            record = await self.repository.get(user_id, for_update=True)
            if record is not None:
                await self.session.delete(record)
            await self.repository.delete_events(user_id)
            await UserMemoryRepository(self.session).delete_by_user(user_id)

        logger.info(f"用户画像及记忆已删除: user_id={user_id}")
