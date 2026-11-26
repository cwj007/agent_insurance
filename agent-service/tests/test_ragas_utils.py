"""Ragas 导出/评判脚本的纯函数测试。"""

import json
import math

from scripts.ragas_export import sample_balanced
from scripts.ragas_judge import _json_safe, load_env, load_payload


def test_sample_balanced_covers_topics():
    """分层采样应覆盖多个主题。"""
    samples = [
        {"query": f"{topic}-{index}", "topic": topic}
        for topic in ("等待期", "理赔", "责任免除")
        for index in range(2)
    ]

    selected = sample_balanced(samples, 4)

    assert len(selected) == 4
    assert {sample["topic"] for sample in selected} == {"等待期", "理赔", "责任免除"}


def test_sample_balanced_without_limit():
    """未指定上限时返回全部样本。"""
    samples = [{"query": "问题", "topic": "等待期"}]
    assert sample_balanced(samples, None) == samples
    assert sample_balanced(samples, 10) == samples


def test_load_env(tmp_path):
    """应解析 .env 中的键值并忽略注释和空行。"""
    path = tmp_path / ".env"
    path.write_text(
        "# 注释\nDEEPSEEK_CHAT_MODEL=deepseek-v4-flash\n\nDEEPSEEK_API_KEY=secret\n",
        encoding="utf-8",
    )

    values = load_env(path)

    assert values == {
        "DEEPSEEK_CHAT_MODEL": "deepseek-v4-flash",
        "DEEPSEEK_API_KEY": "secret",
    }


def test_load_payload(tmp_path):
    """应读取 JSONL payload。"""
    path = tmp_path / "payload.jsonl"
    rows = [
        {"question": "问题1", "contexts": ["上下文"], "ground_truth": "答案"},
        {"question": "问题2", "contexts": [], "ground_truth": ""},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert load_payload(path) == rows


def test_json_safe_handles_nan():
    """NaN 和无穷值应转换为 None，便于 JSON 序列化。"""
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(0.5) == 0.5
    assert math.isclose(_json_safe(1.25), 1.25)
