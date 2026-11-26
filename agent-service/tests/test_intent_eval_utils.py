"""意图评测脚本的工具函数测试。"""

import json

import pytest

from scripts.intent_eval import load_dataset, sample_balanced


def test_sample_balanced_covers_all_intents():
    """分层采样应覆盖所有意图，而不是只取文件开头的单一类别。"""
    samples = [
        {"query": f"{intent}-{index}", "intent": intent}
        for intent in ("chitchat", "claim", "recommendation_plan")
        for index in range(2)
    ]

    selected = sample_balanced(samples, 4)

    assert len(selected) == 4
    assert {sample["intent"] for sample in selected} == {
        "chitchat", "claim", "recommendation_plan",
    }


def test_sample_balanced_returns_all_without_limit():
    """未指定上限时应返回全部样本。"""
    samples = [{"query": "你好", "intent": "chitchat"}]
    assert sample_balanced(samples, None) == samples
    assert sample_balanced(samples, 10) == samples


def test_sample_balanced_handles_limit_over_total():
    """上限超过样本总数时不应重复选取。"""
    samples = [
        {"query": f"{intent}-{index}", "intent": intent}
        for intent in ("claim", "fallback")
        for index in range(2)
    ]
    selected = sample_balanced(samples, 3)
    assert len(selected) == 3
    assert len({sample["query"] for sample in selected}) == 3


def test_load_dataset(tmp_path):
    """合法 JSONL 应解析为 query / intent 列表。"""
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"query": "住院了怎么理赔", "intent": "claim"}, ensure_ascii=False)
        + "\n"
        + "\n",
        encoding="utf-8",
    )

    assert load_dataset(path) == [{"query": "住院了怎么理赔", "intent": "claim"}]


def test_load_dataset_rejects_invalid_intent(tmp_path):
    """非法意图标签应直接报错。"""
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"query": "问题", "intent": "unknown"}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="intent 非法"):
        load_dataset(path)


def test_load_dataset_missing_file(tmp_path):
    """文件不存在时应抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.jsonl")
