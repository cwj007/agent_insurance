"""RAG 检索降级与评测指标测试。"""

from types import SimpleNamespace

from app.rag.retriever import (
    ParentChunkRetriever,
    _expand_query,
    _is_reference_section,
    _prioritize_chunks,
)
from scripts.rag_eval import _match_level, _pick_target, score_sample
from tests.fakes import run_async


def _chunk(
        chunk_id: str = "c1",
        product_id: int = 1,
        sections: tuple[str, ...] = ("条款", "2. 保障", "2.2 等待期"),
        content: str = "内容",
):
    """构造测试用父块。

    Args:
        chunk_id: 父块 ID。
        product_id: 产品 ID。
        sections: 章节路径。
        content: 父块内容。

    Returns:
        父块测试桩。
    """
    return SimpleNamespace(
        id=chunk_id,
        product_id=product_id,
        section_path=list(sections),
        content=content,
    )


class TestRetrieverFallback:
    """重排服务故障时的检索降级。"""

    def test_rerank_failure_falls_back(self):
        """重排失败后应改用无重排混合检索。"""
        calls: list[dict] = []

        class FlakyStore:
            async def asimilarity_search(self, **kwargs):
                calls.append(kwargs)
                if "reranker" in kwargs:
                    raise RuntimeError("rerank unavailable")
                return []

        retriever = ParentChunkRetriever(FlakyStore())
        chunks = run_async(retriever.retrieve("等待期是多久", 3, 1))

        assert chunks == []
        assert len(calls) == 2
        assert "reranker" in calls[0]
        assert "reranker" not in calls[1]

    def test_normal_path_uses_rerank(self):
        """重排正常时只调用一次检索。"""
        calls: list[dict] = []

        class Store:
            async def asimilarity_search(self, **kwargs):
                calls.append(kwargs)
                return []

        retriever = ParentChunkRetriever(Store())
        run_async(retriever.retrieve("等待期是多久", 3, 1))

        assert len(calls) == 1
        assert "reranker" in calls[0]


class TestReferencePrioritization:
    """释义/附表章节降权。"""

    def test_business_sections_come_first(self):
        """业务章节应排在释义/附表之前。"""
        reference = _chunk(
            "ref",
            sections=("条款", "附表一 重大疾病", "1 恶性肿瘤"),
        )
        business = _chunk(
            "biz",
            sections=("条款", "2 保障内容", "2.2 保险期间"),
        )
        other = _chunk("other", sections=("条款", "3 其他"))

        result = _prioritize_chunks([reference, business, other], 2)

        assert [chunk.id for chunk in result] == ["biz", "other"]

    def test_reference_fills_when_not_enough(self):
        """业务章节不足时仍返回引用章节，避免空结果。"""
        reference = _chunk("ref", sections=("条款", "第七部分 释义", "等待期"))

        result = _prioritize_chunks([reference], 3)

        assert [chunk.id for chunk in result] == ["ref"]

    def test_is_reference_section(self):
        """释义和附表应被识别为引用章节。"""
        assert _is_reference_section(
            _chunk(sections=("条款", "第七部分 释义"))
        ) is True
        assert _is_reference_section(
            _chunk(sections=("条款", "附表一 重大疾病"))
        ) is True
        assert _is_reference_section(
            _chunk(sections=("条款", "2. 保障内容"))
        ) is False


class TestQueryExpansion:
    """查询词扩展。"""

    def test_expands_claim_terms(self):
        """理赔类问题应补充保险金申请等条款术语。"""
        expanded = _expand_query("理赔流程是什么？")

        assert "保险金申请" in expanded
        assert expanded.startswith("理赔流程是什么？")

    def test_expands_period_terms(self):
        """保险期间类问题应补充保障期限。"""
        expanded = _expand_query("保障到多少岁？")

        assert "保险期间" in expanded
        assert "保障期限" in expanded

    def test_exclusion_query_uses_exclusion_terms(self):
        """免责类问法只补责任免除术语，不补理赔相关词。"""
        expanded = _expand_query("什么情况下不能理赔？")

        assert "责任免除" in expanded
        assert "保险金申请" not in expanded

    def test_keeps_unknown_query(self):
        """没有命中规则时保持原问题。"""
        assert _expand_query("你好") == "你好"


class TestRagMetrics:
    """RAG 评估指标计算。"""

    def test_hit_at_first(self):
        """目标父块排第一时各项指标应为满分。"""
        sample = {
            "expected_parent_id": "c1",
            "expected_keywords": ["等待期"],
            "product_id": 1,
        }
        chunks = [_chunk("c1")]

        metrics = score_sample(sample, chunks, max_k=5)

        assert metrics["hit@1"] == 1.0
        assert metrics["strict_hit@1"] == 1.0
        assert metrics["precision@1"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_hit_at_second(self):
        """目标父块排第二时应计入 Hit@3、MRR=0.5。"""
        sample = {
            "expected_parent_id": "c1",
            "expected_keywords": ["等待期"],
            "product_id": 1,
        }
        chunks = [_chunk("c2", sections=("条款", "3. 其他")), _chunk("c1")]

        metrics = score_sample(sample, chunks, max_k=5)

        assert metrics["hit@1"] == 0.0
        assert metrics["hit@3"] == 1.0
        assert metrics["strict_hit@3"] == 1.0
        assert metrics["precision@3"] == 0.5
        assert metrics["mrr"] == 0.5

    def test_same_topic_counts_as_relevant(self):
        """同主题但不同父块应计为相关，但不计严格命中。"""
        sample = {
            "expected_parent_id": "c1",
            "expected_keywords": ["等待期"],
            "product_id": 1,
        }
        chunks = [_chunk("c2", sections=("条款", "2. 保障", "2.3 等待期"))]

        metrics = score_sample(sample, chunks, max_k=5)

        assert metrics["hit@1"] == 1.0
        assert metrics["strict_hit@1"] == 0.0
        assert metrics["precision@1"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_empty_results(self):
        """空结果时命中与精确率为 0，产品过滤视为通过。"""
        sample = {
            "expected_parent_id": "c1",
            "expected_keywords": ["等待期"],
            "product_id": 1,
        }

        metrics = score_sample(sample, [], max_k=5)

        assert metrics["hit@1"] == 0.0
        assert metrics["precision@1"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["returned"] == 0.0
        assert metrics["product_filter_ok"] == 1.0

    def test_product_filter(self):
        """返回其他产品的父块时产品过滤标记为失败。"""
        sample = {
            "expected_parent_id": "c1",
            "expected_keywords": ["等待期"],
            "product_id": 1,
        }

        metrics = score_sample(sample, [_chunk("c2", product_id=2)], max_k=5)

        assert metrics["product_filter_ok"] == 0.0


class TestTargetSelection:
    """评估集目标父块选择。"""

    def test_match_level(self):
        """末级匹配优先于祖先章节匹配。"""
        direct = _chunk(sections=("条款", "2. 保障", "2.2 等待期"))
        ancestor = _chunk(sections=("等待期", "2. 保障", "2.2 其他"))
        other = _chunk(sections=("条款", "3. 其他"))

        assert _match_level(direct, ["等待期"]) == 2
        assert _match_level(ancestor, ["等待期"]) == 1
        assert _match_level(other, ["等待期"]) == 0

    def test_excludes_glossary_sections(self):
        """释义和附表不应被选为流程类问题的标注目标。"""
        glossary = _chunk(
            "short",
            sections=("条款", "第七部分 释义", "保险金申请人"),
            content="很短",
        )
        real = _chunk(
            "real",
            sections=("条款", "3 保险金申领", "3.3 保险金申请"),
            content="实际流程内容",
        )

        target = _pick_target([glossary, real], ["保险金申请", "保险金的申请"])

        assert target.id == "real"
