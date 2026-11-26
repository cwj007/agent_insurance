"""导出 Ragas 评估数据：问题、检索到的条款、参考答案。

在项目环境运行，只负责调用检索器并落盘，不依赖 Ragas。

用法：
    uv run python scripts/ragas_export.py --k 5 --fetch-k 10
    uv run python scripts/ragas_export.py --limit 10 --no-rerank

生成文件：data/ragas_eval_payload.jsonl
之后用 scripts/ragas_judge.py 在隔离环境运行 Ragas 评判。
"""

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from app.infra.database import AsyncSessionFactory
from app.rag.models import ParentChunk

DEFAULT_INPUT = PROJECT_ROOT / "data" / "rag_eval.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ragas_eval_payload.jsonl"
DEFAULT_K = 5
DEFAULT_FETCH_K = 10

# 单个上下文和参考答案的截断长度，避免 judge 输入过长
CONTEXT_MAX_CHARS = 1500
REFERENCE_MAX_CHARS = 1200


def load_samples(path: Path) -> list[dict[str, Any]]:
    """读取 RAG 评估集。

    Args:
        path: JSONL 评估集路径。

    Returns:
        样本列表。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    if not path.exists():
        raise FileNotFoundError(f"RAG 评估集不存在: {path}，请先运行 rag_eval.py build")

    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    return samples


def sample_balanced(
        samples: list[dict[str, Any]],
        limit: int | None,
) -> list[dict[str, Any]]:
    """按主题分层采样，避免小样本只覆盖单一主题。

    Args:
        samples: 全部样本。
        limit: 最多选取数量；为空或超过总数时返回全部。

    Returns:
        分层采样后的样本。
    """
    if not limit or limit >= len(samples):
        return list(samples)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample["topic"]].append(sample)

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for topic in sorted(groups):
            if len(selected) >= limit:
                break
            if round_index < len(groups[topic]):
                selected.append(groups[topic][round_index])
                added = True
        if not added:
            break
        round_index += 1

    return selected


async def load_references(
        samples: list[dict[str, Any]],
        max_chars: int,
) -> dict[str, str]:
    """按目标父块 ID 查询参考答案。

    Args:
        samples: 评估样本。
        max_chars: 参考答案最大字符数。

    Returns:
        父块 ID 到参考答案的映射。
    """
    ids = [UUID(sample["expected_parent_id"]) for sample in samples]
    async with AsyncSessionFactory() as session:
        parents = (
            await session.scalars(
                select(ParentChunk).where(ParentChunk.id.in_(ids))
            )
        ).all()

    references: dict[str, str] = {}
    for parent in parents:
        section = " > ".join(parent.section_path or [])
        text = f"{section}\n{parent.content.strip()}"
        references[str(parent.id)] = text[:max_chars]
    return references


async def export(
        input_path: Path,
        output_path: Path,
        k: int,
        fetch_k: int,
        limit: int | None,
        use_rerank: bool,
) -> None:
    """执行检索并导出 Ragas payload。

    Args:
        input_path: RAG 评估集路径。
        output_path: payload 输出路径。
        k: 每条样本返回的父块数量。
        fetch_k: 重排前的候选数量。
        limit: 最多导出的样本数。
        use_rerank: 是否启用重排。
    """
    from app.rag import retriever

    samples = sample_balanced(load_samples(input_path), limit)
    references = await load_references(samples, REFERENCE_MAX_CHARS)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for index, sample in enumerate(samples, 1):
        started = time.perf_counter()
        chunks = await retriever.retrieve(
            sample["query"],
            k,
            sample["product_id"],
            fetch_k=fetch_k,
            use_rerank=use_rerank,
        )
        latencies.append((time.perf_counter() - started) * 1000)

        contexts = []
        for chunk in chunks:
            section = " > ".join(chunk.section_path or [])
            content = chunk.content.strip()[:CONTEXT_MAX_CHARS]
            contexts.append(f"{section}\n{content}")

        rows.append(
            {
                "question": sample["query"],
                "contexts": contexts,
                "ground_truth": references.get(sample["expected_parent_id"], ""),
                "product_id": sample["product_id"],
                "product_name": sample.get("product_name"),
                "topic": sample["topic"],
            }
        )

        if index % 10 == 0 or index == len(samples):
            print(f"导出进度: {index}/{len(samples)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    empty_contexts = sum(1 for row in rows if not row["contexts"])
    empty_refs = sum(1 for row in rows if not row["ground_truth"])
    print(f"payload 已写入: {output_path}")
    print(f"样本数: {len(rows)}，空上下文: {empty_contexts}，空参考答案: {empty_refs}")
    print(
        f"重排: {'开启' if use_rerank else '关闭'}，"
        f"平均延迟: {sum(latencies) / len(latencies) if latencies else 0:.0f}ms"
    )


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。

    Returns:
        ArgumentParser 实例。
    """
    parser = argparse.ArgumentParser(description="导出 Ragas 评估数据")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="RAG 评估集路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="payload 输出路径")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="返回的父块数量")
    parser.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K, help="重排前候选数量")
    parser.add_argument("--limit", type=int, help="最多导出多少条（按主题分层）")
    parser.add_argument("--no-rerank", action="store_true", help="跳过重排")
    return parser


async def main() -> None:
    """脚本入口。"""
    args = _build_parser().parse_args()
    await export(
        args.input,
        args.output,
        args.k,
        args.fetch_k,
        args.limit,
        use_rerank=not args.no_rerank,
    )


if __name__ == "__main__":
    asyncio.run(main())
