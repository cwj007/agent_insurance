"""不启动完整服务，单独编译 LangGraph 主图并导出内置 Mermaid/PNG。

运行：
    cd agent-service
    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/draw_langgraph.py

说明：
- 图的节点和边来自 `graph.get_graph().draw_mermaid()`，不是手写结构。
- 本地用 Graphviz 渲染，避免把图结构发送到 mermaid.ink 等外部服务。
- 应用启动时（`APP_DEBUG=true`）也会自动导出，本脚本用于单独生成。
"""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.chat_models import init_chat_model  # noqa: E402

from app.agents.orchestration.graph import init_insurance_agent_graph  # noqa: E402
from app.agents.orchestration.graph_export import export_graph_diagram  # noqa: E402
from app.agents.routing.router import IntentRouter  # noqa: E402
from app.agents.tools.contracts import WorkflowTools  # noqa: E402
from app.core.config import settings  # noqa: E402


async def main() -> None:
    """编译真实 Graph，打印内置 Mermaid，并导出 PNG。"""
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
    print("=== LangGraph draw_mermaid() ===")
    print(mermaid)

    png_path = export_graph_diagram(graph)
    print(f"已导出: {png_path}")


if __name__ == "__main__":
    asyncio.run(main())
