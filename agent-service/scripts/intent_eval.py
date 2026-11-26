"""意图识别评测与阈值校准脚本。

用法：
    uv run python scripts/intent_eval.py evaluate --mode l1
    uv run python scripts/intent_eval.py evaluate --mode l2 --limit 20
    uv run python scripts/intent_eval.py evaluate --mode full
    uv run python scripts/intent_eval.py calibrate --mode l1
    uv run python scripts/intent_eval.py calibrate --mode l2

说明：
- `l1` 只用 embedding，不依赖聊天模型。
- `l2` / `full` 需要本地 Ollama 聊天模型，可用 `--chat-model` 指定。
- 当前 `data/intent_eval.jsonl` 每类 100 条（共 500 条）。数据集扩充或问法分布
  变化后，需要重新运行 calibrate 并复核阈值。
- `--limit` 会按意图分层采样，避免只评测文件开头的单一类别。
- 本地小模型默认关闭 thinking；开启后结构化输出容易失败，可用 `--reasoning` 对比。
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, get_args

# 将项目根目录加入模块搜索路径，支持直接运行 `python scripts/intent_eval.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.chat_models import init_chat_model

from app.agents.routing.router import (
    INTENT_CONFIDENCE_THRESHOLDS,
    INTENT_MODEL_NAME,
    OLLAMA_BASE_URL,
    IntentRouter,
)
from app.agents.routing.schemas import Intent
from app.agents.routing.semantic import (
    L1_MARGIN_THRESHOLD,
    L1_SIMILARITY_THRESHOLD,
    SemanticIntentIndex,
)

VALID_INTENTS = set(get_args(Intent))
TARGET_PER_INTENT = 50
TARGET_TOTAL = 250


def load_dataset(path: Path) -> list[dict[str, str]]:
    """读取并校验 JSONL 评测集。

    Args:
        path: 数据集路径。

    Returns:
        样本列表，每条包含 query 和 intent。

    Raises:
        FileNotFoundError: 数据文件不存在时抛出。
        ValueError: 字段缺失或意图不在枚举内时抛出。
    """
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在: {path}")

    samples: list[dict[str, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc

        query = str(item.get("query", "")).strip()
        intent = str(item.get("intent", "")).strip()
        if not query:
            raise ValueError(f"第 {line_no} 行 query 为空")
        if intent not in VALID_INTENTS:
            raise ValueError(f"第 {line_no} 行 intent 非法: {intent}")
        samples.append({"query": query, "intent": intent})

    return samples


def report_coverage(samples: list[dict[str, str]]) -> list[str]:
    """输出评测集覆盖情况。

    Args:
        samples: 样本列表。

    Returns:
        报告文本行。
    """
    counter = Counter(sample["intent"] for sample in samples)
    lines = ["=== 数据集 ===", f"总数: {len(samples)}"]

    for intent in sorted(VALID_INTENTS):
        count = counter.get(intent, 0)
        flag = "" if count >= TARGET_PER_INTENT else "  <- 建议补充"
        lines.append(f"  {intent}: {count}{flag}")

    if len(samples) < TARGET_TOTAL:
        lines.append(
            f"提示: 当前样本少于每类 {TARGET_PER_INTENT} 条，阈值容易过拟合，建议先扩充。"
        )

    return lines


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """计算 precision / recall / F1。

    Args:
        tp: 真正例数。
        fp: 假正例数。
        fn: 假负例数。

    Returns:
        (precision, recall, f1)。
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    """计算整体准确率、宏平均 F1 和每类指标。

    Args:
        gold: 真实标签列表。
        pred: 预测标签列表。

    Returns:
        包含 accuracy、macro_f1 和 per_intent 的字典。
    """
    labels = sorted(set(gold) | set(pred))
    per_intent: dict[str, dict[str, float]] = {}

    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision, recall, f1 = _prf(tp, fp, fn)
        per_intent[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(sum(1 for g in gold if g == label)),
        }

    accuracy = (
        sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)
        if gold
        else 0.0
    )
    macro_f1 = (
        sum(item["f1"] for item in per_intent.values()) / len(per_intent)
        if per_intent
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_intent": per_intent,
    }


def _format_metrics(metrics: dict[str, Any]) -> list[str]:
    """格式化指标报告。

    Args:
        metrics: `_metrics` 的返回值。

    Returns:
        报告文本行。
    """
    lines = [
        "=== 指标 ===",
        f"accuracy: {metrics['accuracy']:.4f}",
        f"macro_f1: {metrics['macro_f1']:.4f}",
        "",
        f"{'意图':<22}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for intent, item in sorted(metrics["per_intent"].items()):
        lines.append(
            f"{intent:<22}{item['precision']:>12.4f}{item['recall']:>10.4f}"
            f"{item['f1']:>10.4f}{int(item['support']):>10d}"
        )
    return lines


def _format_confusion(gold: list[str], pred: list[str]) -> list[str]:
    """格式化混淆矩阵。

    Args:
        gold: 真实标签列表。
        pred: 预测标签列表。

    Returns:
        报告文本行。
    """
    labels = sorted(set(gold) | set(pred))
    matrix = {g: Counter() for g in labels}
    for g, p in zip(gold, pred):
        matrix[g][p] += 1

    lines = ["=== 混淆矩阵 ==="]
    lines.append("| 真实\\预测 | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels) + 1))
    for g in labels:
        values = " | ".join(str(matrix[g].get(p, 0)) for p in labels)
        lines.append(f"| {g} | {values} |")
    return lines


def _emit(lines: list[str], output: Path | None) -> None:
    """打印报告并按需写入文件。

    Args:
        lines: 报告文本行。
        output: 可选输出路径。
    """
    text = "\n".join(lines)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"\n报告已写入: {output}")


async def _l1_predictions(
        index: SemanticIntentIndex,
        samples: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """计算每条样本的 L1 最高分和领先优势。

    Args:
        index: 已构建的语义索引。
        samples: 样本列表。

    Returns:
        包含 gold、top_intent、score、margin 的列表。
    """
    rows: list[dict[str, Any]] = []
    for sample in samples:
        top = await index.top_match(sample["query"])
        if top is None:
            rows.append(
                {
                    "gold": sample["intent"],
                    "top_intent": None,
                    "score": 0.0,
                    "margin": 0.0,
                }
            )
            continue

        intent, score, margin = top
        rows.append(
            {
                "gold": sample["intent"],
                "top_intent": intent,
                "score": score,
                "margin": margin,
            }
        )
    return rows


def _l1_predict(
        rows: list[dict[str, Any]],
        similarity_threshold: float,
        margin_threshold: float,
) -> list[str]:
    """按 L1 阈值生成预测标签。

    Args:
        rows: `_l1_predictions` 的输出。
        similarity_threshold: 最高相似度阈值。
        margin_threshold: 领先第二名的最小分差。

    Returns:
        预测标签列表，未命中时记为 fallback。
    """
    predictions: list[str] = []
    for row in rows:
        if (
                row["top_intent"] is not None
                and row["score"] >= similarity_threshold
                and row["margin"] >= margin_threshold
        ):
            predictions.append(str(row["top_intent"]))
        else:
            predictions.append("fallback")
    return predictions


async def evaluate_l1(
        samples: list[dict[str, str]],
        similarity_threshold: float,
        margin_threshold: float,
        output: Path | None,
) -> None:
    """评测 L1 语义层。

    Args:
        samples: 样本列表。
        similarity_threshold: 最高相似度阈值。
        margin_threshold: 最小分差阈值。
        output: 可选报告输出路径。
    """
    index = SemanticIntentIndex(
        threshold=similarity_threshold,
        margin=margin_threshold,
    )
    await index.build()

    rows = await _l1_predictions(index, samples)
    gold = [row["gold"] for row in rows]
    pred = _l1_predict(rows, similarity_threshold, margin_threshold)

    hits = sum(1 for p in pred if p != "fallback")
    lines = [
        "=== L1 评测 ===",
        f"similarity_threshold={similarity_threshold}, margin_threshold={margin_threshold}",
        f"L1 命中: {hits}/{len(pred)} ({hits / len(pred):.1%})",
        "说明: 未命中记为 fallback，实际链路会继续交给 L2。",
        "",
    ]
    lines.extend(_format_confusion(gold, pred))
    lines.append("")
    lines.extend(_format_metrics(_metrics(gold, pred)))
    _emit(lines, output)


async def calibrate_l1(
        samples: list[dict[str, str]],
        output: Path | None,
) -> None:
    """网格搜索 L1 的最优相似度阈值和分差阈值。

    Args:
        samples: 样本列表。
        output: 可选报告输出路径。
    """
    index = SemanticIntentIndex(threshold=0.0, margin=0.0)
    await index.build()
    rows = await _l1_predictions(index, samples)
    gold = [row["gold"] for row in rows]

    similarity_thresholds = [
        round(0.80 + i * 0.01, 2) for i in range(21)
    ]
    margin_thresholds = [
        round(i * 0.005, 3) for i in range(21)
    ]

    results: list[dict[str, Any]] = []
    for similarity_threshold in similarity_thresholds:
        for margin_threshold in margin_thresholds:
            pred = _l1_predict(rows, similarity_threshold, margin_threshold)
            metrics = _metrics(gold, pred)
            hits = sum(1 for p in pred if p != "fallback")
            results.append(
                {
                    "similarity_threshold": similarity_threshold,
                    "margin_threshold": margin_threshold,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "coverage": hits / len(pred) if pred else 0.0,
                }
            )

    results.sort(
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            item["coverage"],
        ),
        reverse=True,
    )

    best = results[0]
    lines = [
        "=== L1 阈值校准 ===",
        "按 macro_f1 排序，前 10 组：",
        f"{'similarity':>12}{'margin':>10}{'accuracy':>12}{'macro_f1':>12}{'coverage':>12}",
    ]
    for item in results[:10]:
        lines.append(
            f"{item['similarity_threshold']:>12.2f}{item['margin_threshold']:>10.3f}"
            f"{item['accuracy']:>12.4f}{item['macro_f1']:>12.4f}"
            f"{item['coverage']:>12.1%}"
        )

    lines.extend(
        [
            "",
            "=== 推荐配置（需用更大标注集复核） ===",
            f"L1_SIMILARITY_THRESHOLD = {best['similarity_threshold']:.2f}",
            f"L1_MARGIN_THRESHOLD = {best['margin_threshold']:.3f}",
            "",
            "=== 最佳阈值下的 PR 曲线（固定 margin，扫描 similarity） ===",
            f"{'similarity':>12}{'precision':>12}{'recall':>10}{'f1':>10}{'hits':>8}",
        ]
    )

    best_margin = best["margin_threshold"]
    non_fallback_total = sum(1 for g in gold if g != "fallback")
    for similarity_threshold in similarity_thresholds:
        hits = [
            row for row in rows
            if row["top_intent"] is not None
            and row["score"] >= similarity_threshold
            and row["margin"] >= best_margin
        ]
        correct = sum(1 for row in hits if row["top_intent"] == row["gold"])
        precision = correct / len(hits) if hits else 0.0
        recall = correct / non_fallback_total if non_fallback_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        lines.append(
            f"{similarity_threshold:>12.2f}{precision:>12.4f}{recall:>10.4f}"
            f"{f1:>10.4f}{len(hits):>8d}"
        )

    _emit(lines, output)


def _build_router(args: argparse.Namespace, enable_l1: bool) -> IntentRouter:
    """按命令行参数创建意图路由器。

    Args:
        args: 命令行参数。
        enable_l1: 是否启用 L1。

    Returns:
        IntentRouter 实例。
    """
    model = init_chat_model(
        model=args.chat_model,
        model_provider="ollama",
        base_url=args.base_url,
        temperature=0,
        reasoning=args.reasoning,
    )
    fallback_model = None
    if args.fallback_model:
        fallback_model = init_chat_model(
            model=args.fallback_model,
            model_provider="ollama",
            base_url=args.base_url,
            temperature=0,
            reasoning=args.reasoning,
        )
    return IntentRouter(
        model=model,
        fallback_model=fallback_model,
        enable_l1=enable_l1,
    )


def sample_balanced(
        samples: list[dict[str, str]],
        limit: int | None,
) -> list[dict[str, str]]:
    """按意图分层均匀采样，避免 --limit 截断后只覆盖单一意图。

    Args:
        samples: 样本列表。
        limit: 最多选取多少条；为空或超过总数时返回全部样本。

    Returns:
        分层采样后的样本列表。
    """
    if not limit or limit >= len(samples):
        return list(samples)

    groups: dict[str, list[dict[str, str]]] = {}
    for sample in samples:
        groups.setdefault(sample["intent"], []).append(sample)

    selected: list[dict[str, str]] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for intent in sorted(groups):
            if len(selected) >= limit:
                break
            if round_index < len(groups[intent]):
                selected.append(groups[intent][round_index])
                added = True
        if not added:
            break
        round_index += 1

    return selected


async def _l2_predictions(
        router: IntentRouter,
        samples: list[dict[str, str]],
        limit: int | None,
) -> list[dict[str, Any]]:
    """逐条调用 L2 分类，收集意图和置信度。

    Args:
        router: 意图路由器。
        samples: 样本列表。
        limit: 可选的最大样本数。

    Returns:
        包含 gold、pred、confidence、error 的列表。
    """
    selected = sample_balanced(samples, limit)
    distribution = Counter(sample["intent"] for sample in selected)
    print(f"评测样本: {len(selected)} 条，分布 {dict(distribution)}")

    rows: list[dict[str, Any]] = []
    total = len(selected)

    for index, sample in enumerate(selected, 1):
        try:
            result = await router.classify_l2(sample["query"], "")
            rows.append(
                {
                    "query": sample["query"],
                    "gold": sample["intent"],
                    "pred": result.intent,
                    "confidence": result.confidence,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "query": sample["query"],
                    "gold": sample["intent"],
                    "pred": None,
                    "confidence": 0.0,
                    "error": str(exc),
                }
            )

        if index % 10 == 0 or index == total:
            print(f"L2 进度: {index}/{total}")

    return rows


def _l2_error_lines(rows: list[dict[str, Any]]) -> list[str]:
    """生成 L2 调用失败提示。

    Args:
        rows: L2 预测结果。

    Returns:
        报告文本行。
    """
    errors = [row for row in rows if row["error"] is not None]
    if not errors:
        return []

    lines = [f"L2 调用失败: {len(errors)}/{len(rows)}"]
    lines.append(f"示例错误: {errors[0]['error']}")
    for row in errors[:5]:
        lines.append(f"  失败样本: {row['query']}")
    if len(errors) == len(rows):
        lines.append("全部失败，请先确认 Ollama 模型可用（ollama list / 重新 pull）。")
        lines.append("若错误为 Invalid json output，检查模型是否开启了 thinking；"
                     "可用 --no-reasoning 重新评测。")
    return lines


async def evaluate_l2(
        samples: list[dict[str, str]],
        args: argparse.Namespace,
        output: Path | None,
) -> None:
    """评测 L2 置信度阈值下的分类效果。

    Args:
        samples: 样本列表。
        args: 命令行参数。
        output: 可选报告输出路径。
    """
    router = _build_router(args, enable_l1=False)
    rows = await _l2_predictions(router, samples, args.limit)

    gold = [row["gold"] for row in rows]
    pred = [row["pred"] or "fallback" for row in rows]

    # 低于当前阈值会进入澄清/兜底，单独统计，避免只看原始分类准确率
    low_confidence = 0
    for row in rows:
        if row["error"] is not None:
            low_confidence += 1
            continue
        threshold = INTENT_CONFIDENCE_THRESHOLDS.get(row["pred"], 1.0)
        if row["confidence"] < threshold:
            low_confidence += 1

    lines = ["=== L2 评测 ==="]
    lines.extend(_l2_error_lines(rows))
    lines.append(f"低于当前阈值（会澄清/兜底）: {low_confidence}/{len(rows)} "
                 f"({low_confidence / len(rows):.1%})")
    lines.append("")
    lines.extend(_format_confusion(gold, pred))
    lines.append("")
    lines.extend(_format_metrics(_metrics(gold, pred)))
    _emit(lines, output)


def _calibrate_threshold_for_intent(
        rows: list[dict[str, Any]],
        target: str,
        min_precision: float,
) -> dict[str, Any] | None:
    """为单个意图搜索最佳置信度阈值。

    Args:
        rows: L2 预测结果。
        target: 目标意图。
        min_precision: 最低 precision 要求，低于该值的阈值不采用。

    Returns:
        包含 best_threshold、best_metrics 和 curve 的字典；无候选时返回 None。
    """
    candidates = [
        row for row in rows
        if row["pred"] == target and row["error"] is None
    ]
    positives = sum(1 for row in rows if row["gold"] == target)
    if not candidates:
        return None

    curve: list[dict[str, float]] = []
    for step in range(101):
        threshold = round(step / 100, 2)
        selected = [row for row in candidates if row["confidence"] >= threshold]
        tp = sum(1 for row in selected if row["gold"] == target)
        fp = len(selected) - tp
        fn = positives - tp
        precision, recall, f1 = _prf(tp, fp, fn)
        curve.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "selected": float(len(selected)),
            }
        )

    eligible = [
        item for item in curve
        if item["precision"] >= min_precision and item["selected"] > 0
    ]
    pool = eligible or curve
    best = max(pool, key=lambda item: (item["f1"], item["precision"]))

    return {
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "positives": positives,
        "candidates": len(candidates),
        "curve": curve,
    }


async def calibrate_l2(
        samples: list[dict[str, str]],
        args: argparse.Namespace,
        output: Path | None,
) -> None:
    """用标注集校准每个意图的置信度阈值。

    Args:
        samples: 样本列表。
        args: 命令行参数。
        output: 可选报告输出路径。
    """
    router = _build_router(args, enable_l1=False)
    rows = await _l2_predictions(router, samples, args.limit)

    lines = ["=== L2 阈值校准 ==="]
    lines.extend(_l2_error_lines(rows))
    lines.append("")

    recommended: dict[str, float] = dict(INTENT_CONFIDENCE_THRESHOLDS)
    for intent in sorted(VALID_INTENTS):
        result = _calibrate_threshold_for_intent(
            rows,
            intent,
            args.min_precision,
        )
        if result is None:
            lines.append(f"{intent}: 没有预测样本，保留当前阈值")
            lines.append("")
            continue

        best = result["best_metrics"]
        recommended[intent] = result["best_threshold"]
        lines.append(
            f"{intent}: 推荐阈值={result['best_threshold']:.2f}, "
            f"precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
            f"f1={best['f1']:.4f}, 正例={result['positives']}, "
            f"候选={result['candidates']}"
        )

        lines.append(
            f"  {'threshold':>10}{'precision':>12}{'recall':>10}{'f1':>10}"
        )
        top_curve = sorted(
            result["curve"],
            key=lambda item: (item["f1"], item["precision"]),
            reverse=True,
        )[:5]
        for item in top_curve:
            lines.append(
                f"  {item['threshold']:>10.2f}{item['precision']:>12.4f}"
                f"{item['recall']:>10.4f}{item['f1']:>10.4f}"
            )
        lines.append("")

    lines.extend(
        [
            "=== 推荐配置（需用更大标注集复核） ===",
            "INTENT_CONFIDENCE_THRESHOLDS = {",
        ]
    )
    for intent in sorted(recommended):
        lines.append(f'    "{intent}": {recommended[intent]:.2f},')
    lines.append("}")

    _emit(lines, output)


async def evaluate_full(
        samples: list[dict[str, str]],
        args: argparse.Namespace,
        output: Path | None,
) -> None:
    """评测 L0 → L1 → L2 的完整路由链路。

    Args:
        samples: 样本列表。
        args: 命令行参数。
        output: 可选报告输出路径。
    """
    router = _build_router(args, enable_l1=True)
    await router.initialize()

    selected = sample_balanced(samples, args.limit)
    gold: list[str] = []
    pred: list[str] = []
    layers: Counter[str] = Counter()
    errors = 0
    clarify_count = 0

    for index, sample in enumerate(selected, 1):
        gold.append(sample["intent"])
        try:
            result, layer = await router.route(sample["query"], "")
            pred.append(result.intent)
            layers[layer] += 1
            if router.should_clarify(result, layer):
                clarify_count += 1
        except Exception:
            pred.append("fallback")
            layers["fallback"] += 1
            errors += 1

        if index % 10 == 0 or index == len(selected):
            print(f"完整链路进度: {index}/{len(selected)}")

    lines = [
        "=== 完整链路评测 ===",
        f"L0/L1/L2 来源分布: {dict(layers)}",
        f"路由异常: {errors}",
        f"会触发澄清: {clarify_count}/{len(selected)} "
        f"({clarify_count / len(selected):.1%})",
        "",
    ]
    lines.extend(_format_confusion(gold, pred))
    lines.append("")
    lines.extend(_format_metrics(_metrics(gold, pred)))
    _emit(lines, output)


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。

    Returns:
        ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(description="意图识别评测与阈值校准")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("evaluate", "calibrate"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "--mode",
            choices=["l1", "l2", "full"],
            default="l1",
            help="评测/校准的层",
        )
        sub.add_argument(
            "--data",
            type=Path,
            default=PROJECT_ROOT / "data" / "intent_eval.jsonl",
            help="JSONL 评测集路径",
        )
        sub.add_argument("--output", type=Path, help="报告输出路径")
        sub.add_argument("--limit", type=int, help="最多评测多少条（L2 很慢时使用）")
        sub.add_argument(
            "--chat-model",
            default=INTENT_MODEL_NAME,
            help="L2/完整链路使用的 Ollama 聊天模型",
        )
        sub.add_argument(
            "--fallback-model",
            default=None,
            help="主模型失败时使用的 Ollama 兜底模型名；默认不启用",
        )
        sub.add_argument(
            "--reasoning",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="是否开启本地模型的 thinking；默认关闭（开启会导致结构化输出失败）",
        )
        sub.add_argument(
            "--base-url",
            default=OLLAMA_BASE_URL,
            help="Ollama 服务地址",
        )
        sub.add_argument(
            "--min-precision",
            type=float,
            default=0.0,
            help="L2 校准的最低 precision 要求",
        )
        sub.add_argument(
            "--sim-threshold",
            type=float,
            default=L1_SIMILARITY_THRESHOLD,
            help="L1 评测使用的相似度阈值",
        )
        sub.add_argument(
            "--margin",
            type=float,
            default=L1_MARGIN_THRESHOLD,
            help="L1 评测使用的分差阈值",
        )

    return parser


async def main() -> None:
    """脚本入口。"""
    args = _build_parser().parse_args()
    samples = load_dataset(args.data)
    coverage = report_coverage(samples)
    print("\n".join(coverage))
    print()

    if args.command == "evaluate" and args.mode == "l1":
        await evaluate_l1(
            samples,
            args.sim_threshold,
            args.margin,
            args.output,
        )
    elif args.command == "evaluate" and args.mode == "l2":
        await evaluate_l2(samples, args, args.output)
    elif args.command == "evaluate" and args.mode == "full":
        await evaluate_full(samples, args, args.output)
    elif args.command == "calibrate" and args.mode == "l1":
        await calibrate_l1(samples, args.output)
    elif args.command == "calibrate" and args.mode == "l2":
        await calibrate_l2(samples, args, args.output)
    else:
        raise SystemExit("calibrate 只支持 --mode l1 / l2")


if __name__ == "__main__":
    asyncio.run(main())
