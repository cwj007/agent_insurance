"""RAG 条款检索评测：构建评估集并计算准确率、精确率。

用法：
    uv run python scripts/rag_eval.py build
    uv run python scripts/rag_eval.py evaluate --k 5
    uv run python scripts/rag_eval.py evaluate --k 5 --output data/rag_eval_report.md

说明：
- `build` 从 parent_chunks 表按主题选取父块，自动生成评估集。
- `evaluate` 调用 app.rag.retriever，计算 Hit@k、Precision@k、MRR 等指标。
- 相关性定义：命中目标父块，或 section_path 与目标主题匹配的父块。
- 评测集按产品 + 主题生成，查询不包含产品名，依赖 product_id 过滤。
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from app.rag.models import ParentChunk

DEFAULT_DATA = PROJECT_ROOT / "data" / "rag_eval.jsonl"
DEFAULT_K = 5
DEFAULT_FETCH_K = 10
KS: tuple[int, ...] = (1, 3, 5)

# 释义、疾病附表和不完整通知等辅助章节不适合作为“流程/规则”类问题的标注目标
_EXCLUDED_SECTION_HINTS: tuple[str, ...] = (
    "释义", "附表", "病种", "疾病定义", "申请人", "资料不完整",
)

# 条款主题：关键词用于定位父块，queries 是常见的用户问法
TOPICS: dict[str, dict[str, Any]] = {
    "等待期": {
        "keywords": ["等待期"],
        "queries": [
            "等待期是多久？",
            "这款保险买了之后多久生效？",
            "等待期内生病能理赔吗？",
            "保险的观察期是多长时间？",
        ],
    },
    "保险责任": {
        "keywords": ["保险责任", "保障责任", "保障内容", "保险金的给付"],
        "queries": [
            "这个产品保哪些责任？",
            "这份保险能赔哪些情况？",
            "保险责任具体包括什么？",
            "这个产品提供哪些保障？",
        ],
    },
    "责任免除": {
        "keywords": ["责任免除", "免责"],
        "queries": [
            "哪些情况保险公司不赔？",
            "这份保险的免责条款有哪些？",
            "什么情况下不能理赔？",
            "责任免除包括哪些情形？",
        ],
    },
    "犹豫期": {
        "keywords": ["犹豫期"],
        "queries": [
            "买完保险多久内可以全额退保？",
            "犹豫期是多少天？",
            "刚买的保险不想要了能退吗？",
            "这份产品有犹豫期吗？",
        ],
    },
    "理赔": {
        "keywords": ["理赔", "保险金申请", "保险金的申请", "索赔", "申请与给付"],
        "queries": [
            "理赔需要准备哪些材料？",
            "怎么申请保险金？",
            "理赔流程是什么？",
            "申请赔付要提交什么资料？",
        ],
    },
    "续保": {
        "keywords": ["续保"],
        "queries": [
            "这个保险可以续保吗？",
            "续保需要重新审核吗？",
            "续保条件是什么？",
            "产品停售了还能续保吗？",
        ],
    },
    "退保": {
        "keywords": ["退保", "解除合同"],
        "queries": [
            "退保怎么办理？",
            "退保能拿回多少钱？",
            "不想买了怎么解除合同？",
            "退保需要什么手续？",
        ],
    },
    "如实告知": {
        "keywords": ["如实告知", "健康告知", "告知义务"],
        "queries": [
            "投保时健康告知怎么填？",
            "没有如实告知会有什么后果？",
            "买保险需要告知既往病史吗？",
            "如实告知义务是什么？",
        ],
    },
    "免赔额": {
        "keywords": ["免赔额"],
        "queries": [
            "免赔额是多少？",
            "每次住院都要扣免赔额吗？",
            "免赔额怎么计算？",
            "一万免赔额是什么意思？",
        ],
    },
    "宽限期": {
        "keywords": ["宽限期", "效力中止"],
        "queries": [
            "忘记交保费会怎么样？",
            "保费宽限期有几天？",
            "宽限期内出险能赔吗？",
            "保单失效了怎么恢复？",
        ],
    },
    "保险期间": {
        "keywords": ["保险期间"],
        "queries": [
            "这份保险保多久？",
            "保险期间是多长时间？",
            "保障到多少岁？",
            "保险什么时候到期？",
        ],
    },
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """读取并校验 RAG 评估集。

    Args:
        path: JSONL 评估集路径。

    Returns:
        样本列表。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
        ValueError: 必填字段缺失时抛出。
    """
    if not path.exists():
        raise FileNotFoundError(f"评估集不存在: {path}，请先运行 build")

    samples: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc

        for field in ("query", "product_id", "topic", "expected_parent_id", "expected_keywords"):
            if field not in item:
                raise ValueError(f"第 {line_no} 行缺少字段: {field}")
        samples.append(item)

    return samples


def _match_level(chunk: ParentChunk, keywords: list[str]) -> int:
    """计算父块与主题的匹配程度。

    Args:
        chunk: 父知识块。
        keywords: 主题关键词列表。

    Returns:
        2 表示章节末级直接匹配，1 表示祖先章节匹配，0 表示不匹配。
    """
    section_path = chunk.section_path or []
    if section_path and any(
            keyword in section_path[-1] for keyword in keywords
    ):
        return 2
    if any(
            any(keyword in section for keyword in keywords)
            for section in section_path
    ):
        return 1
    return 0


def _pick_target(chunks: list[ParentChunk], keywords: list[str]) -> ParentChunk:
    """从匹配主题的父块中挑一个作为标注目标。

    优先排除释义、附表等辅助章节，再选择章节末级直接匹配且内容较短的父块。

    Args:
        chunks: 同一产品下匹配主题的父块列表。
        keywords: 主题关键词列表。

    Returns:
        选中的目标父块。
    """
    def is_excluded(chunk: ParentChunk) -> bool:
        """判断父块是否位于释义或附表章节。

        Args:
            chunk: 父知识块。

        Returns:
            章节路径包含排除词时返回 True。
        """
        return any(
            hint in section
            for section in (chunk.section_path or [])
            for hint in _EXCLUDED_SECTION_HINTS
        )

    candidates = [chunk for chunk in chunks if not is_excluded(chunk)] or chunks
    return min(
        candidates,
        key=lambda chunk: (-_match_level(chunk, keywords), len(chunk.content)),
    )


async def build_dataset(output: Path) -> None:
    """从父块表构建 RAG 评估集。

    Args:
        output: 评估集输出路径。
    """
    async with AsyncSessionFactory() as session:
        products = (await session.scalars(select(Product).order_by(Product.id))).all()
        chunks = (await session.scalars(select(ParentChunk))).all()

    by_product: dict[int, list[ParentChunk]] = defaultdict(list)
    for chunk in chunks:
        by_product[chunk.product_id].append(chunk)

    items: list[dict[str, Any]] = []
    for product in products:
        product_chunks = by_product.get(product.id, [])
        for topic_index, (topic, config) in enumerate(TOPICS.items()):
            keywords = config["keywords"]
            matches = [chunk for chunk in product_chunks if _match_level(chunk, keywords) > 0]
            if not matches:
                continue

            target = _pick_target(matches, keywords)
            queries = config["queries"]
            query = queries[(product.id + topic_index) % len(queries)]
            items.append(
                {
                    "query": query,
                    "product_id": product.id,
                    "product_name": product.name,
                    "topic": topic,
                    "expected_keywords": keywords,
                    "expected_parent_id": str(target.id),
                    "expected_section": list(target.section_path),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
        encoding="utf-8",
    )

    topic_counts: dict[str, int] = defaultdict(int)
    for item in items:
        topic_counts[item["topic"]] += 1

    print(f"评估集已写入: {output}")
    print(f"样本数: {len(items)}，覆盖产品: {len({item['product_id'] for item in items})} 个")
    print(f"主题分布: {dict(topic_counts)}")


def _is_relevant(chunk: ParentChunk, sample: dict[str, Any]) -> bool:
    """判断父块是否与样本相关。

    Args:
        chunk: 检索返回的父块。
        sample: 评估样本。

    Returns:
        命中目标父块或章节主题匹配时返回 True。
    """
    if str(chunk.id) == str(sample["expected_parent_id"]):
        return True
    keywords = sample["expected_keywords"]
    return any(
        any(keyword in section for keyword in keywords)
        for section in chunk.section_path
    )


def score_sample(
        sample: dict[str, Any],
        chunks: list[ParentChunk],
        max_k: int = DEFAULT_K,
) -> dict[str, float]:
    """计算单条样本的检索指标。

    Args:
        sample: 评估样本。
        chunks: 按相关性排序的检索结果。
        max_k: 参与计算的最大结果数量。

    Returns:
        包含 hit@k、precision@k、strict_hit@k、mrr 的指标字典。
    """
    top_chunks = chunks[:max_k]
    flags = [_is_relevant(chunk, sample) for chunk in top_chunks]
    strict_flags = [
        str(chunk.id) == str(sample["expected_parent_id"])
        for chunk in top_chunks
    ]

    metrics: dict[str, float] = {}
    for k in KS:
        if k > max_k:
            continue
        top_flags = flags[:k]
        metrics[f"hit@{k}"] = 1.0 if any(top_flags) else 0.0
        metrics[f"strict_hit@{k}"] = 1.0 if any(strict_flags[:k]) else 0.0
        denominator = min(k, len(top_chunks))
        metrics[f"precision@{k}"] = (
            sum(top_flags) / denominator if denominator else 0.0
        )

    first_relevant = next((index + 1 for index, flag in enumerate(flags) if flag), None)
    metrics["mrr"] = 1.0 / first_relevant if first_relevant else 0.0
    metrics["returned"] = float(len(top_chunks))
    metrics["product_filter_ok"] = 1.0 if all(
        chunk.product_id == sample["product_id"] for chunk in top_chunks
    ) else 0.0
    return metrics


def _average(rows: list[dict[str, float]], name: str) -> float:
    """计算某指标的平均值。

    Args:
        rows: 指标字典列表。
        name: 指标名。

    Returns:
        平均值；列表为空时返回 0.0。
    """
    values = [row[name] for row in rows if name in row]
    return sum(values) / len(values) if values else 0.0


def _group_report(
        groups: dict[str, list[dict[str, float]]],
        k: int,
) -> list[str]:
    """生成分组指标表。

    Args:
        groups: 分组名到指标列表的映射。
        k: 报表主要使用的 k。

    Returns:
        报告文本行。
    """
    lines = [
        f"{'分组':<24}{'样本':>6}{'Hit':>10}{'Precision':>12}{'StrictHit':>12}{'MRR':>8}",
    ]
    for name, rows in sorted(groups.items()):
        lines.append(
            f"{name:<24}{len(rows):>6}"
            f"{_average(rows, f'hit@{k}'):>10.4f}"
            f"{_average(rows, f'precision@{k}'):>12.4f}"
            f"{_average(rows, f'strict_hit@{k}'):>12.4f}"
            f"{_average(rows, 'mrr'):>8.4f}"
        )
    return lines


def build_report(
        results: list[dict[str, Any]],
        latencies: list[float],
        k: int,
) -> list[str]:
    """汇总评估结果并生成文本报告。

    Args:
        results: 每条样本的结果，包含指标与元信息。
        latencies: 每条样本的检索延迟（毫秒）。
        k: 报表主要使用的 k。

    Returns:
        报告文本行。
    """
    rows = [row["metrics"] for row in results]
    empty_count = sum(1 for row in rows if row["returned"] == 0)
    error_count = sum(1 for result in results if result.get("error"))
    lines = [
        "=== RAG 条款检索评测 ===",
        f"样本数: {len(results)}，k={k}，平均延迟: {sum(latencies) / len(latencies) if latencies else 0:.0f}ms",
        f"空结果: {empty_count}/{len(results)}，检索异常: {error_count}/{len(results)}",
        f"产品过滤正确: {_average(rows, 'product_filter_ok'):.1%}",
        "",
        "=== 总体指标 ===",
        f"{'指标':<18}{'值':>10}",
    ]

    for kb in KS:
        if kb > k:
            continue
        lines.append(f"Hit@{kb:<14}{_average(rows, f'hit@{kb}'):>10.4f}")
    for kb in KS:
        if kb > k:
            continue
        lines.append(f"Precision@{kb:<8}{_average(rows, f'precision@{kb}'):>10.4f}")
    for kb in KS:
        if kb > k:
            continue
        lines.append(f"StrictHit@{kb:<6}{_average(rows, f'strict_hit@{kb}'):>10.4f}")
    lines.append(f"{'MRR':<18}{_average(rows, 'mrr'):>10.4f}")

    by_topic: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_product: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in results:
        by_topic[row["topic"]].append(row["metrics"])
        by_product[row["product_name"]].append(row["metrics"])

    lines.extend(["", "=== 按主题 ==="])
    lines.extend(_group_report(by_topic, k))
    lines.extend(["", "=== 按产品 ==="])
    lines.extend(_group_report(by_product, k))

    failures = [
        row for row in results
        if row["metrics"].get(f"hit@{k}", 0.0) == 0.0
    ][:10]
    lines.extend(["", f"=== 失败样本（Hit@{k}=0，最多展示 10 条） ==="])
    if not failures:
        lines.append("无")
    for row in failures:
        suffix = f" 错误: {row['error'][:120]}" if row.get("error") else ""
        lines.append(
            f"- [{row['topic']}] {row['query']} "
            f"({row['product_name']}){suffix}"
        )
        for chunk in row["top_chunks"]:
            lines.append(f"    {chunk}")
        lines.append(f"    期望: {' > '.join(row['expected_section'])}")

    return lines


async def evaluate(
        data_path: Path,
        max_k: int,
        output: Path | None,
        fetch_k: int = DEFAULT_FETCH_K,
        use_rerank: bool = True,
) -> None:
    """执行 RAG 检索评测。

    Args:
        data_path: 评估集路径。
        max_k: 每条样本检索的父块数量上限。
        output: 可选报告输出路径。
        fetch_k: 重排前的子块候选数量。
        use_rerank: 是否启用重排。
    """
    from app.rag import retriever

    samples = load_dataset(data_path)
    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    for index, sample in enumerate(samples, 1):
        started = time.perf_counter()
        try:
            chunks = await retriever.retrieve(
                sample["query"],
                max_k,
                sample["product_id"],
                fetch_k=fetch_k,
                use_rerank=use_rerank,
            )
            error = None
        except Exception as exc:
            chunks = []
            error = str(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)

        results.append(
            {
                **sample,
                "metrics": score_sample(sample, chunks, max_k),
                "error": error,
                "top_chunks": [
                    " > ".join(chunk.section_path) or "未标注章节"
                    for chunk in chunks[:max_k]
                ],
            }
        )

        if index % 10 == 0 or index == len(samples):
            print(f"评测进度: {index}/{len(samples)}")

    lines = build_report(results, latencies, max_k)
    text = "\n".join(lines)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"\n报告已写入: {output}")


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。

    Returns:
        ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(description="RAG 条款检索评测")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="构建评估集")
    build_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA,
        help="评估集输出路径",
    )

    eval_parser = subparsers.add_parser("evaluate", help="执行检索评测")
    eval_parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="评估集路径",
    )
    eval_parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="每条样本检索的父块数量上限",
    )
    eval_parser.add_argument(
        "--fetch-k",
        type=int,
        default=DEFAULT_FETCH_K,
        help="重排前的子块候选数量",
    )
    eval_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="跳过重排，直接评测混合检索结果",
    )
    eval_parser.add_argument("--output", type=Path, help="报告输出路径")

    return parser


async def main() -> None:
    """脚本入口。"""
    args = _build_parser().parse_args()
    if args.command == "build":
        await build_dataset(args.output)
    else:
        await evaluate(
            args.data,
            args.k,
            args.output,
            args.fetch_k,
            use_rerank=not args.no_rerank,
        )


if __name__ == "__main__":
    asyncio.run(main())
