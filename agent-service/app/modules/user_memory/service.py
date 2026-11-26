"""用户情节记忆服务。"""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.modules.user_memory.models import UserMemory
from app.modules.user_memory.repository import UserMemoryRepository

logger = get_logger(__name__)

# 默认注入的历史记忆条数
MEMORY_RENDER_LIMIT = 8

# 单条记忆最大字符数
MAX_MEMORY_CHARS = 500

# 合法记忆类型
VALID_MEMORY_KINDS = frozenset({"fact", "preference", "event", "decision"})


def render_memories(memories: list[UserMemory]) -> str:
    """把情节记忆渲染成可注入 system prompt 的文本。

    Args:
        memories: 记忆列表。

    Returns:
        渲染后的文本；没有记忆时返回空字符串。
    """
    if not memories:
        return ""

    ordered = sorted(
        memories,
        key=lambda memory: memory.created_at or datetime.min,
    )
    lines = ["## 历史记忆（来自过去的对话，仅供理解上下文）"]
    for memory in ordered:
        date_text = (
            memory.created_at.strftime("%Y-%m-%d")
            if memory.created_at
            else "未知日期"
        )
        lines.append(f"- [{date_text}] {memory.content}")

    lines.append(
        "使用要求：只在与当前问题相关时自然引用，不要主动复述全部记忆。"
    )
    return "\n".join(lines)


class UserMemoryService:
    """用户情节记忆服务。"""

    def __init__(self, session: AsyncSession):
        """初始化记忆服务。

        Args:
            session: 数据库异步会话。
        """
        self.session = session
        self.repository = UserMemoryRepository(session)

    async def add_episodes(
            self,
            user_id: int,
            thread_id: str | None,
            episodes: list[dict],
    ) -> int:
        """写入一轮对话抽取出的情节记忆，按内容去重。

        Args:
            user_id: 用户 ID。
            thread_id: 来源会话 ID。
            episodes: 抽取出的情节列表，每项包含 kind、content、importance。

        Returns:
            实际新增的条数。
        """
        if not episodes:
            return 0

        added = 0
        async with self.session.begin():
            existing = await self.repository.list_recent_contents(user_id)

            for episode in episodes:
                content = str(episode.get("content") or "").strip()[:MAX_MEMORY_CHARS]
                if not content or content in existing:
                    continue

                kind = str(episode.get("kind") or "event")
                if kind not in VALID_MEMORY_KINDS:
                    kind = "event"

                try:
                    importance = float(episode.get("importance", 0.5))
                except (TypeError, ValueError):
                    importance = 0.5

                self.session.add(
                    UserMemory(
                        user_id=user_id,
                        thread_id=thread_id,
                        kind=kind,
                        content=content,
                        importance=min(max(importance, 0.0), 1.0),
                    )
                )
                existing.add(content)
                added += 1

        if added:
            logger.info(f"用户情节记忆已更新: user_id={user_id}, added={added}~☑️")
        return added

    async def get_render_text(
            self,
            user_id: int,
            limit: int = MEMORY_RENDER_LIMIT,
    ) -> str:
        """读取并渲染用户最近的情节记忆。

        Args:
            user_id: 用户 ID。
            limit: 最多注入条数。

        Returns:
            渲染后的记忆文本；没有记忆时返回空字符串。
        """
        memories = await self.repository.list_recent(user_id, limit)
        return render_memories(memories)

    async def list_for_user(
            self,
            user_id: int,
            limit: int = 50,
    ) -> list[UserMemory]:
        """查询用户最近的记忆，供 API 展示。

        Args:
            user_id: 用户 ID。
            limit: 最多返回条数。

        Returns:
            记忆列表。
        """
        return await self.repository.list_recent(user_id, limit)

    async def delete_all(self, user_id: int) -> None:
        """删除用户全部记忆。

        Args:
            user_id: 用户 ID。
        """
        async with self.session.begin():
            await self.repository.delete_by_user(user_id)
        logger.info(f"用户情节记忆已删除: user_id={user_id}")
