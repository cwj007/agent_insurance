"""用 LangGraph 内置 draw_mermaid() 导出主图，并在本地渲染为 PNG。

运行：
    cd agent-service
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/draw_langgraph.py

说明：
- 图的节点和边来自 `graph.get_graph().draw_mermaid()`，不是手写结构。
- 本地用 Graphviz 渲染，避免把图结构发送到 mermaid.ink 等外部服务。
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
IMAGES_DIR = REPO_ROOT / "docs" / "images"

# 允许直接运行 python scripts/draw_langgraph.py
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.chat_models import init_chat_model  # noqa: E402

from app.agents.orchestration.graph import init_insurance_agent_graph  # noqa: E402
from app.agents.routing.router import IntentRouter  # noqa: E402
from app.agents.tools.contracts import WorkflowTools  # noqa: E402
from app.core.config import settings  # noqa: E402

# 节点展示文案；结构仍以 draw_mermaid() 输出为准
NODE_LABELS: dict[str, str] = {
    "__start__": "START",
    "prepare_context": "prepare_context\\n上下文摘要",
    "input_guard": "input_guard\\n输入安全校验 + 合并意图分析",
    "guard_reply": "guard_reply\\n固定拦截话术",
    "route": "route\\n意图路由",
    "clarify_node": "clarify_node\\n澄清追问",
    "recommendation_node": "recommendation_node\\n推荐子 Agent",
    "claim_node": "claim_node\\n理赔子 Agent",
    "chitchat_node": "chitchat_node\\n闲聊",
    "human_handoff_node": "human_handoff_node\\n转人工",
    "fallback_node": "fallback_node\\n兜底选项",
    "output_guard": "output_guard\\n输出合规校验",
    "__end__": "END",
}

# 边标签；仅用于展示，不改变图结构
EDGE_LABELS: dict[tuple[str, str], str] = {
    ("input_guard", "guard_reply"): "拦截",
    ("input_guard", "route"): "放行",
    ("route", "human_handoff_node"): "human_handoff",
    ("route", "recommendation_node"): "recommendation_plan",
    ("route", "claim_node"): "claim",
    ("route", "chitchat_node"): "chitchat",
    ("route", "fallback_node"): "fallback / 超限",
    ("route", "clarify_node"): "低置信度",
}

NODE_STYLES: dict[str, str] = {
    "__start__": "shape=circle,style=filled,fillcolor=\"#0F172A\",fontcolor=\"white\",width=0.75,fixedsize=true",
    "__end__": "shape=circle,style=filled,fillcolor=\"#0F172A\",fontcolor=\"white\",width=0.75,fixedsize=true",
    "prepare_context": "fillcolor=\"#E0F2FE\",color=\"#0284C7\"",
    "input_guard": "shape=diamond,style=filled,fillcolor=\"#FEF3C7\",color=\"#D97706\"",
    "guard_reply": "fillcolor=\"#FEE2E2\",color=\"#DC2626\"",
    "route": "shape=diamond,style=filled,fillcolor=\"#FEF3C7\",color=\"#D97706\"",
    "output_guard": "fillcolor=\"#FFEDD5\",color=\"#EA580C\"",
    "clarify_node": "fillcolor=\"#EDE9FE\",color=\"#7C3AED\"",
}

NODE_DEF_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")
EDGE_PATTERN = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(-\.->|-->)\s*([A-Za-z_]\w*)\s*;"
)


def _parse_mermaid(mermaid: str) -> tuple[list[str], list[tuple[str, str, bool]]]:
    """从 LangGraph 的 Mermaid 输出中提取节点和边。

    Args:
        mermaid: `draw_mermaid()` 返回的 Mermaid 文本。

    Returns:
        (节点列表, 边列表)；边为 (源, 目标, 是否动态边)。
    """
    nodes: list[str] = []
    edges: list[tuple[str, str, bool]] = []

    for line in mermaid.splitlines():
        edge_match = EDGE_PATTERN.match(line)
        if edge_match:
            source, arrow, target = edge_match.groups()
            edges.append((source, target, arrow == "-.->"))
            continue

        node_match = NODE_DEF_PATTERN.match(line)
        if node_match:
            name = node_match.group(1)
            if name not in nodes:
                nodes.append(name)

    for source, target, _ in edges:
        for name in (source, target):
            if name not in nodes:
                nodes.append(name)

    return nodes, edges


def _build_dot(
        nodes: list[str],
        edges: list[tuple[str, str, bool]],
) -> str:
    """把解析结果转换为 Graphviz DOT。

    Args:
        nodes: 节点列表。
        edges: 边列表。

    Returns:
        可直接交给 dot 渲染的 DOT 字符串。
    """
    lines = [
        "digraph LangGraph {",
        "    graph [",
        "        rankdir=TB,",
        '        bgcolor="#FFFFFF",',
        '        fontname="Microsoft YaHei",',
        "        fontsize=22,",
        "        labelloc=t,",
        '        label="保险顾问主 Graph（LangGraph draw_mermaid 生成）\\n虚线 = Command 动态跳转",',
        "        nodesep=0.55,",
        "        ranksep=0.70,",
        "        pad=0.35,",
        "        splines=true",
        "    ];",
        "    node [",
        '        fontname="Microsoft YaHei",',
        "        fontsize=14,",
        "        shape=box,",
        '        style="rounded,filled",',
        '        color="#64748B",',
        "        penwidth=1.5,",
        '        margin="0.20,0.12",',
        '        fillcolor="#DCFCE7"',
        "    ];",
        "    edge [",
        '        fontname="Microsoft YaHei",',
        "        fontsize=11,",
        '        color="#64748B",',
        "        arrowsize=0.8,",
        "        penwidth=1.3",
        "    ];",
        "",
    ]

    for name in nodes:
        label = NODE_LABELS.get(name, name)
        style = NODE_STYLES.get(name, "")
        attrs = [f'label="{label}"']
        if style:
            attrs.append(style)
        lines.append(f'    "{name}" [{",".join(attrs)}];')

    lines.append("")

    for source, target, is_dynamic in edges:
        attrs: list[str] = []
        label = EDGE_LABELS.get((source, target))
        if label:
            attrs.append(f'label="{label}"')
        if is_dynamic:
            attrs.extend([
                "style=dotted",
                'color="#7C3AED"',
                'fontcolor="#7C3AED"',
            ])
        if source == "input_guard" and target == "guard_reply":
            attrs.extend(['color="#DC2626"', 'fontcolor="#DC2626"'])
        elif source == "input_guard" and target == "route":
            attrs.extend(['color="#16A34A"', 'fontcolor="#16A34A"'])
        attr_text = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f'    "{source}" -> "{target}"{attr_text};')

    lines.append("}")
    return "\n".join(lines)


def _find_dot() -> str:
    """查找 Graphviz dot 可执行文件。

    Returns:
        dot 命令或绝对路径。

    Raises:
        RuntimeError: 找不到 Graphviz 时抛出。
    """
    for candidate in (
            "dot",
            r"E:\Graphviz\bin\dot.exe",
            r"C:\Program Files\Graphviz\bin\dot.exe",
    ):
        try:
            subprocess.run(
                [candidate, "-V"],
                check=True,
                capture_output=True,
                text=True,
            )
            return candidate
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("未找到 Graphviz dot，请先安装 Graphviz 并加入 PATH")


async def main() -> None:
    """编译真实 Graph，导出内置 Mermaid，并在本地渲染 PNG。"""
    model = init_chat_model(
        model=settings.llm.deepseek_chat_model,
        model_provider="deepseek",
        api_key=settings.llm.deepseek_api_key,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # 关闭 L1，避免构建语义索引时访问 Ollama
    router = IntentRouter(model=model, fallback_model=model, enable_l1=False)

    graph = await init_insurance_agent_graph(
        checkpointer=None,
        model=model,
        intent_router=router,
        fallback_model=model,
        tools=WorkflowTools(),
        tool_middlewares=[],
    )

    mermaid = graph.get_graph().draw_mermaid()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    mmd_path = IMAGES_DIR / "langgraph-built.mmd"
    dot_path = IMAGES_DIR / "langgraph-built.dot"
    png_path = IMAGES_DIR / "01-langgraph.png"
    mmd_path.write_text(mermaid, encoding="utf-8")

    nodes, edges = _parse_mermaid(mermaid)
    dot_path.write_text(_build_dot(nodes, edges), encoding="utf-8")

    subprocess.run(
        [_find_dot(), "-Tpng", "-Gdpi=200", str(dot_path), "-o", str(png_path)],
        check=True,
    )

    print("=== LangGraph draw_mermaid() ===")
    print(mermaid)
    print(f"节点 {len(nodes)} 个，边 {len(edges)} 条")
    print(f"已生成: {png_path}")


if __name__ == "__main__":
    asyncio.run(main())
