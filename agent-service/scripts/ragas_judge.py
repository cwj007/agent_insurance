"""用 Ragas 评判检索上下文质量。

必须在隔离环境运行，避免 Ragas 0.2.x 与项目的 langchain 1.x 冲突：

    uv run --isolated \
      --with "ragas==0.2.15" \
      --with "langchain==0.3.27" \
      --with "langchain-core==0.3.79" \
      --with "langchain-community==0.3.27" \
      --with "langchain-openai==0.3.35" \
      python scripts/ragas_judge.py \
        --input data/ragas_eval_payload.jsonl \
        --output data/ragas_eval_report.json

先运行 `uv run python scripts/ragas_export.py` 生成 payload。
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

# 项目根目录，仅用于读取 .env，不导入项目代码
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "ragas_eval_payload.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ragas_eval_report.json"

# Ragas 指标原始名称到报告字段名的映射
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


def load_env(path: Path) -> dict[str, str]:
    """读取 .env 配置。

    Args:
        path: .env 路径。

    Returns:
        键值对字典。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    if not path.exists():
        raise FileNotFoundError(f".env 不存在: {path}")

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_payload(path: Path) -> list[dict[str, Any]]:
    """读取导出的 Ragas payload。

    Args:
        path: payload JSONL 路径。

    Returns:
        样本列表。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    if not path.exists():
        raise FileNotFoundError(f"payload 不存在: {path}，请先运行 scripts/ragas_export.py")

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _json_safe(value: Any) -> Any:
    """把 NaN/Inf 等值转换成可 JSON 序列化的形式。

    Args:
        value: 任意值。

    Returns:
        可序列化的值；NaN 转为 None。
    """
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="Ragas 检索上下文评判")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="payload 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="报告输出路径")
    parser.add_argument("--workers", type=int, default=4, help="judge 并发数")
    parser.add_argument("--timeout", type=int, default=120, help="单次 judge 超时秒数")
    args = parser.parse_args()

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig

    env = load_env(PROJECT_ROOT / ".env")
    model = env.get("DEEPSEEK_CHAT_MODEL") or "deepseek-chat"
    api_key = env.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY，无法运行 Ragas judge")

    payload = load_payload(args.input)
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=row["contexts"],
            reference=row["ground_truth"],
        )
        for row in payload
    ]
    dataset = EvaluationDataset(samples=samples)

    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0,
        )
    )

    print(f"开始 Ragas 评估: {len(samples)} 条，judge={model}，并发={args.workers}")
    metrics = [LLMContextPrecisionWithReference(), LLMContextRecall()]
    started = time.perf_counter()
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge,
        run_config=RunConfig(max_workers=args.workers, timeout=args.timeout),
        raise_exceptions=False,
        show_progress=True,
    )
    elapsed = time.perf_counter() - started

    frame = result.to_pandas()
    columns = {
        metric.name: METRIC_DISPLAY_NAMES.get(metric.name, metric.name)
        for metric in metrics
        if metric.name in frame.columns
    }
    summary = {
        display: _json_safe(float(frame[raw].mean()))
        for raw, display in columns.items()
    }
    failed = {
        display: int(frame[raw].isna().sum())
        for raw, display in columns.items()
    }

    per_sample = []
    for index, row in frame.iterrows():
        item = payload[index] if index < len(payload) else {}
        per_sample.append(
            {
                "question": item.get("question"),
                "topic": item.get("topic"),
                "product_name": item.get("product_name"),
                **{
                    display: _json_safe(float(row[raw]))
                    for raw, display in columns.items()
                },
            }
        )

    report = {
        "samples": len(payload),
        "judge_model": model,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": summary,
        "failed": failed,
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Ragas 结果 ===")
    for name, value in summary.items():
        print(f"{name}: {value}")
    print(f"失败样本: {failed}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"报告已写入: {args.output}")


if __name__ == "__main__":
    main()
