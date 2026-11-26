"""用户画像抽取器：从单轮对话中抽取用户信息并异步落库。"""

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.infra.database import AsyncSessionFactory
from app.modules.user_memory.service import UserMemoryService
from app.modules.user_profile.schemas import ExtractedProfile
from app.modules.user_profile.service import UserProfileService

logger = get_logger(__name__)

# 单次抽取超时时间，避免后台任务长期占用资源
EXTRACTION_TIMEOUT_SECONDS = 30.0

# 助手回复截断长度，仅用于理解上下文
ASSISTANT_TEXT_LIMIT = 1000

SYSTEM_PROMPT = """你是保险顾问系统的用户画像抽取器。
请从「用户本轮消息」中提取用户个人信息；推荐产品和方案概述允许从助手回复中提取。

字段约定：
- age：年龄，整数
- gender：性别，取值 male / female / other
- city_tier：城市级别，取值 一线 / 二线 / 三线 / 四线及以下
- occupation：职业
- social_insurance：是否有社保，布尔值
- marital_status：婚姻状况，取值 single / married / divorced / widowed
- children_count：子女数量，整数
- parents_supported：是否赡养父母，布尔值
- health_conditions：健康状况或既往病史，字符串列表
- smoking：是否吸烟，布尔值
- drinking：是否饮酒，布尔值
- annual_income_range：年收入范围，带单位，例如「20-30 万元」
- budget：年保费预算，带单位，例如「3000 元/年」
- target_categories：意向险种列表，取值 medical / critical_illness / life / accident
- concerns：关注点列表，例如「重疾保障」「保费价格」
- recommended_products：已推荐产品列表，每项格式为「险种：产品名」，
  例如「医疗险：众安尊享e生百万医疗险2026版」；只记录助手明确推荐的产品，
  不要记录候选列表或查询结果
- plan_summary：当前推荐方案概述，一句话说明组合和理由
- price_sensitivity：价格敏感度，取值 高 / 中 / 低
- brand_preference：品牌偏好
- episodes：本轮值得跨会话记住的关键情节，1~3 条；每条包含 kind、content、importance。
  kind 取值 fact / preference / event / decision；content 一句话，具体可复用；
  例如「推荐了医疗险众安尊享e生 + 重疾险达尔文12号的组合」。

抽取规则：
1. 只输出本轮新出现或发生变化的信息；用户没提到的字段不要填写。
2. 用户明确说出的信息一律用 stated，包括「预算 X 元」「我有社保」等；
   只有当用户没有明确说、由你合理推断时才用 inferred。
3. 健康相关字段（health_conditions、smoking、drinking）只能使用 stated，禁止推断。
4. 用户明确要求「记住……」时，把对应事实填入相应字段，source 使用 stated。
5. recommended_products 和 plan_summary 可以从助手回复中提取，source 统一用 inferred。
6. 不要编造、不要猜测；无法确定的信息留空。
7. 已有画像中已经包含且本轮没有变化的信息，不要重复输出。
8. target_categories、concerns、recommended_products 必须使用列表。
9. episodes 只记录真正值得跨会话保留、且已有历史记忆中没有的信息；
   寒暄和工具原始输出不要写。
""".strip()


class ProfileExtractor:
    """用户画像抽取器。

    Attributes:
        _chain: 绑定 ExtractedProfile 结构化输出的模型调用链。
    """

    def __init__(self, model: BaseChatModel):
        """初始化画像抽取器。

        Args:
            model: 用于抽取的聊天模型，建议使用低温度模型。
        """
        self._chain = model.with_structured_output(ExtractedProfile)

    async def extract(
            self,
            user_message: str,
            assistant_message: str,
            existing_profile: str,
            existing_memories: str = "",
    ) -> tuple[dict[str, dict], list[dict]]:
        """从单轮对话中抽取画像增量与情节记忆。

        Args:
            user_message: 用户本轮消息。
            assistant_message: 助手本轮回复，用于提取推荐产品和方案。
            existing_profile: 已有画像的渲染文本。
            existing_memories: 已有情节记忆的渲染文本，用于避免重复。

        Returns:
            (画像字段增量, 情节记忆列表)；没有抽取到内容时返回空结果。
        """
        # 工作流失败时没有有效助手回答，仍可从用户消息抽取事实
        assistant_text = (
            assistant_message.strip()[:ASSISTANT_TEXT_LIMIT]
            or "（本轮未生成有效回答）"
        )
        human_text = (
            f"【已有画像】\n{existing_profile or '无'}\n\n"
            f"【已有历史记忆】\n{existing_memories or '无'}\n\n"
            f"【本轮对话】\n用户：{user_message}\n"
            f"助手：{assistant_text}"
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]

        result = await asyncio.wait_for(
            self._chain.ainvoke(messages),
            timeout=EXTRACTION_TIMEOUT_SECONDS,
        )
        if not isinstance(result, ExtractedProfile):
            logger.warning("用户画像抽取返回了非预期结果")
            return {}, []

        data = result.model_dump(exclude_none=True)
        episodes = data.pop("episodes", []) or []
        fields = {
            name: value
            for name, value in data.items()
            if isinstance(value, dict)
        }
        return fields, episodes


# 运行时单例，由 init_profile_extractor 初始化
_extractor: ProfileExtractor | None = None

# 持有后台任务引用，避免任务被垃圾回收
_background_tasks: set[asyncio.Task] = set()


def init_profile_extractor(model: BaseChatModel) -> None:
    """初始化画像抽取器。

    Args:
        model: 用于抽取的聊天模型。
    """
    global _extractor
    try:
        _extractor = ProfileExtractor(model)
        logger.info("用户画像抽取器初始化成功~☑️")
    except Exception as exc:
        logger.error(f"用户画像抽取器初始化失败: {exc}")
        _extractor = None


def schedule_profile_extraction(
        user_id: int,
        thread_id: str,
        user_message: str,
        assistant_message: str,
) -> None:
    """在后台异步执行画像抽取，不阻塞对话响应。

    Args:
        user_id: 用户 ID。
        thread_id: 会话 ID，用于追溯画像来源。
        user_message: 用户本轮消息。
        assistant_message: 助手本轮回复。
    """
    if _extractor is None:
        return
    if not user_message.strip():
        return

    task = asyncio.create_task(
        _extract_and_merge(user_id, thread_id, user_message, assistant_message),
        name=f"profile-extraction-{user_id}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _extract_and_merge(
        user_id: int,
        thread_id: str,
        user_message: str,
        assistant_message: str,
) -> None:
    """读取画像、抽取增量并落库。

    Args:
        user_id: 用户 ID。
        thread_id: 会话 ID。
        user_message: 用户本轮消息。
        assistant_message: 助手本轮回复。
    """
    extractor = _extractor
    if extractor is None:
        return

    try:
        # 1. 读取已有画像和记忆，避免重复抽取
        async with AsyncSessionFactory() as session:
            existing_profile = await UserProfileService(session).get_render_text(user_id)
            existing_memories = await UserMemoryService(session).get_render_text(user_id)

        # 2. 调用模型抽取画像增量与情节记忆
        fields, episodes = await extractor.extract(
            user_message,
            assistant_message,
            existing_profile,
            existing_memories,
        )

        # 3. 画像合并入库；使用独立事务和连接，不影响主对话
        if fields:
            async with AsyncSessionFactory() as session:
                await UserProfileService(session).merge_extraction(
                    user_id,
                    fields,
                    thread_id=thread_id,
                )

        # 4. 情节记忆入库，按内容去重
        if episodes:
            async with AsyncSessionFactory() as session:
                await UserMemoryService(session).add_episodes(
                    user_id,
                    thread_id,
                    episodes,
                )
    except asyncio.TimeoutError:
        logger.warning(f"用户画像抽取超时: user_id={user_id}")
    except Exception as exc:
        logger.warning(f"用户画像抽取失败: user_id={user_id}, error={exc}")
