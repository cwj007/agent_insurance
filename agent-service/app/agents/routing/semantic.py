"""L1 语义路由：用 embedding 做意图相似度初筛。"""

import math

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.agents.routing.examples import INTENT_EXAMPLES
from app.agents.routing.schemas import Intent
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 与 RAG 共用本地 Ollama 嵌入模型
OLLAMA_BASE_URL = settings.llm.ollama_base_url
EMBEDDING_MODEL_NAME = settings.rag.embedding_model
EMBEDDING_DIMENSIONS = settings.rag.embedding_dimensions

# L1 阈值：在 data/intent_eval.jsonl（500 条标注集，每类 100 条）上用
# `scripts/intent_eval.py calibrate --mode l1` 校准。
# 实测 qwen3-embedding 的最高相似度 p50≈0.73、p75≈0.80、max≈0.95，
# 0.96 会导致 L1 永不命中；0.80 时 precision≈94.5%、覆盖率≈14.6%，
# 0.85 时 precision≈96.2%、覆盖率≈5.2%。这里取 0.80 换取覆盖率，
# 同时用 margin 过滤掉第一名优势不明显的歧义输入。
# 数据集继续扩充或问法分布变化后需要重新校准。
L1_SIMILARITY_THRESHOLD = 0.80
L1_MARGIN_THRESHOLD = 0.02

# 批量编码的批大小；Ollama 的上下文有限，一次发送全部示例会失败
EMBED_BATCH_SIZE = 16


def _normalize(vector: list[float]) -> list[float]:
    """对向量做 L2 归一化，之后余弦相似度可以直接用点积计算。

    Args:
        vector: 原始向量。

    Returns:
        归一化后的向量；零向量原样返回。
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    """计算两个已归一化向量的点积。

    Args:
        left: 左侧向量。
        right: 右侧向量。

    Returns:
        点积结果，等价于余弦相似度。
    """
    return sum(a * b for a, b in zip(left, right))


class SemanticIntentIndex:
    """意图示例向量索引。

    启动时把所有示例句子编码成向量；查询时计算每个意图的最高相似度。
    只有最高分和领先优势都达到阈值，才认为 L1 命中。

    Attributes:
        _embeddings: 嵌入模型。
        _threshold: 最高相似度阈值。
        _margin: 第一名与第二名的最小分差。
        _vectors: (意图, 归一化向量) 列表。
    """

    def __init__(
            self,
            embeddings: Embeddings | None = None,
            threshold: float = L1_SIMILARITY_THRESHOLD,
            margin: float = L1_MARGIN_THRESHOLD,
    ):
        """初始化语义索引。

        Args:
            embeddings: 可选的嵌入模型，便于测试时注入。
            threshold: 最高相似度阈值。
            margin: 第一名与第二名的最小分差。
        """
        self._embeddings = embeddings or OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            dimensions=EMBEDDING_DIMENSIONS,
            base_url=OLLAMA_BASE_URL,
            client_kwargs={
                "trust_env": False,  # 忽略系统代理，避免连不上本地 Ollama
                "timeout": 60.0,
            },
        )
        self._threshold = threshold
        self._margin = margin
        self._vectors: list[tuple[Intent, list[float]]] = []

    @property
    def ready(self) -> bool:
        """索引是否已经构建完成。

        Returns:
            已构建且示例不为空时返回 True。
        """
        return bool(self._vectors)

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批编码文档，避免单次批量超过 Ollama 上下文长度。

        Args:
            texts: 待编码文本列表。

        Returns:
            与输入一一对应的向量列表。
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            try:
                vectors.extend(await self._embeddings.aembed_documents(batch))
            except Exception as exc:
                # 批量失败时降级逐条编码，避免单个坏批次拖垮整个索引
                logger.warning(f"批量嵌入失败，改为逐条编码: {exc}")
                for text in batch:
                    vectors.append(await self._embeddings.aembed_query(text))
        return vectors

    async def build(self) -> None:
        """编码全部意图示例并构建内存索引。

        Raises:
            Exception: 嵌入模型调用失败时向上抛出，由调用方决定是否降级。
        """
        texts: list[str] = []
        intents: list[Intent] = []

        for intent, examples in INTENT_EXAMPLES.items():
            for example in examples:
                text = example.strip()
                if not text:
                    continue
                texts.append(text)
                intents.append(intent)

        if not texts:
            self._vectors = []
            logger.warning("L1 语义索引没有可用示例，已跳过")
            return

        vectors = await self._embed_documents(texts)
        self._vectors = [
            (intent, _normalize(vector))
            for intent, vector in zip(intents, vectors)
        ]
        logger.info(f"L1 语义索引构建完成，共 {len(self._vectors)} 条示例~☑️")

    async def top_match(self, query: str) -> tuple[Intent, float, float] | None:
        """返回相似度最高的意图，不做阈值过滤。

        该方法主要给阈值校准脚本使用。

        Args:
            query: 用户输入。

        Returns:
            (意图, 最高相似度, 与第二名的分差)；索引为空时返回 None。
        """
        if not self._vectors:
            return None

        query_vector = _normalize(await self._embeddings.aembed_query(query))

        # 每个意图取所有示例中的最高分
        best_by_intent: dict[Intent, float] = {}
        for intent, vector in self._vectors:
            score = _dot(query_vector, vector)
            if intent not in best_by_intent or score > best_by_intent[intent]:
                best_by_intent[intent] = score

        ordered = sorted(best_by_intent.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        # 点积可能有极小浮点误差，这里 clamp 到 [0, 1]，避免 RouteResult 校验失败
        best_score = min(1.0, max(0.0, best_score))
        second_score = min(1.0, max(0.0, second_score))

        return best_intent, best_score, best_score - second_score

    async def search(self, query: str) -> tuple[Intent, float] | None:
        """按阈值搜索意图；不命中时返回 None，交给 L2。

        Args:
            query: 用户输入。

        Returns:
            (意图, 相似度)；最高分或领先优势不足时返回 None。
        """
        match = await self.top_match(query)
        if match is None:
            return None

        intent, score, margin = match
        if score < self._threshold or margin < self._margin:
            logger.debug(
                f"L1 未命中: best={intent}, score={score:.3f}, margin={margin:.3f}"
            )
            return None

        logger.debug(f"L1 命中: intent={intent}, score={score:.3f}, margin={margin:.3f}")
        return intent, score
