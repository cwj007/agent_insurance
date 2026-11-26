# Agent Insurance（安心保 · 保险智能商城）

基于 **LangGraph + RAG** 的保险智能商城示例项目：前端提供产品浏览、投保申请与 AI 保险顾问，后端提供多工作流智能体、条款知识库检索、运行管理与合规护栏。

> 当前版本已完成 Agent 主链路、RAG 检索优化、用户画像/情节记忆、运行事件流与评测脚本，后端单元测试全部通过。

## 核心能力

- **多工作流意图路由**：L0 规则（转人工/低信息量）→ L1 语义匹配 → L2 小模型结构化分类，逐层降级，低置信度先澄清、超限兜底。
- **推荐与理赔子 Agent**：推荐工作流调用产品查询 + 条款检索工具；理赔工作流基于条款回答，工具调用有次数上限、超时重试与错误转换。
- **RAG 条款问答**：MinerU 解析 PDF → Markdown 清洗 → 父子块切分 → Milvus 稠密 + BM25 混合检索 → DashScope 重排；口语问法自动扩展为条款术语，释义/附表章节自动降权。
- **输入/输出合规护栏**：识别提示词注入、骗保、隐私、误导性宣传、医疗诊断等风险；输入侧故障放行保可用，输出侧故障拦截保合规。
- **运行与连接解耦**：对话在后台执行，SSE 只订阅运行事件；支持断线重连、取消、重试、重放，多 worker 心跳与失联运行回收。
- **用户画像与情节记忆**：从对话中抽取年龄、预算、健康状况等画像字段，跨会话保存关键事实与决定；敏感健康字段只接受用户明确陈述。
- **可评测**：提供意图识别评测/阈值校准、RAG Hit@k/MRR 评测、Ragas 上下文质量评测，以及 164 条离线单元测试。

## 系统架构

```mermaid
flowchart LR
    U[React + Ant Design] -->|REST / SSE| API[FastAPI]
    API --> CHAT[ChatService]
    CHAT --> RUN[Harness Runner<br/>后台运行]
    RUN --> GRAPH[LangGraph 主图]
    GRAPH --> ROUTE[意图路由<br/>L0 → L1 → L2]
    ROUTE --> WF[推荐 / 理赔 / 闲聊 / 转人工 / 兜底]
    WF --> TOOLS[产品查询 / 条款检索]
    TOOLS --> PG[(PostgreSQL)]
    TOOLS --> MV[(Milvus)]
    RUN --> EVENTS[(agent_runs / run_events<br/>tool_invocations)]
    RUN --> MEM[(user_profiles / user_memories)]
```

### 一次对话的执行流程

```text
START → prepare_context（历史摘要/裁剪）
      → input_guard（输入合规 + 意图预计算）
          ├─ 拦截 → guard_reply → END
          └─ 放行 → route（复用或重算意图）
                      ├─ 信息不足 → clarify_node
                      └─ 明确意图 → recommendation / claim / chitchat / handoff / fallback
                                      → output_guard（输出合规）
                                      → END
```

- 输入侧把 **安全校验 + L2 意图分类** 合并为一次模型调用，减少首响延迟。
- 工作流回答必须经过 `output_guard` 才会被流式推送；被拦截时前端通过 `replace` 事件覆盖已展示内容。
- 推荐/理赔子 Agent 使用 `create_agent` 构建，共享主模型、兜底模型与工具中间件。

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React 18、TypeScript、Vite 6、Ant Design 5 |
| 后端 | Python 3.14+、FastAPI、Pydantic v2、SQLAlchemy 2（asyncpg）、uv |
| Agent | LangChain 1.x、LangGraph 1.x、DeepSeek（主模型）、DashScope（兜底/嵌入/重排） |
| 检索 | Milvus 3.0（dense + BM25 稀疏向量）、MinerU（PDF 解析）、Ollama（本地嵌入/意图模型） |
| 存储 | PostgreSQL 16（业务数据、画像、运行事件、LangGraph checkpoint）、MinIO、etcd |
| 评测 | pytest、自研意图/RAG 评测脚本、Ragas 0.2（隔离环境） |

## 目录结构

```text
agent_insurance/
├── frontend/                     # React 前端：首页、产品卡片、AI 顾问、投保流程
├── agent-service/                # FastAPI 后端
│   ├── app/
│   │   ├── agents/               # 纯 Agent 逻辑，不依赖业务数据结构
│   │   │   ├── orchestration/    # 主 Graph 编排与状态
│   │   │   ├── routing/          # L0/L1/L2 意图路由与合并输入分析
│   │   │   ├── workflows/        # 推荐、理赔、闲聊、转人工、澄清、兜底
│   │   │   ├── safety/           # 输入/输出合规护栏
│   │   │   ├── context/          # 历史摘要与画像注入中间件
│   │   │   ├── tools/            # 工具契约、结果协议、限流/重试中间件
│   │   │   └── models/           # 模型重试、兜底与熔断
│   │   ├── harness/              # 业务装配层：运行器、事件、心跳、恢复、工具实现
│   │   ├── modules/              # 产品、会话、聊天、运行、事件、工具调用、画像、记忆
│   │   ├── rag/                  # Milvus 向量库、父子块切分、混合检索与重排
│   │   ├── infra/                # 数据库、checkpoint、过期数据清理
│   │   └── core/                 # 配置、日志、异常
│   ├── data/
│   │   ├── raw/kb/               # 10 份保险条款 PDF（入库）
│   │   ├── raw/rate_tables/      # 10 份费率表 PDF（原始资料）
│   │   └── *.jsonl / *.md        # 意图、RAG、Ragas 评测数据与报告
│   ├── scripts/                  # 知识库构建、意图/RAG/Ragas 评测
│   └── tests/                    # 离线单元测试 + 真实依赖集成测试
└── deploy/                       # PostgreSQL、Milvus、MinIO、etcd、Attu
```

## 快速开始

### 0. 环境要求

- Python **3.14+**、[uv](https://docs.astral.sh/uv/)
- Node.js 20+、npm
- Docker / Docker Compose
- [Ollama](https://ollama.com/)（本地嵌入与意图模型）
- DeepSeek、DashScope（阿里云百炼）、MinerU 的 API Key

### 1. 启动基础设施

```bash
cd deploy
docker compose up -d
```

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| PostgreSQL | 5432 | 业务库，默认 `insurance / insurance / insurance123` |
| Milvus | 19530、9091 | 向量数据库 |
| MinIO | 9000、9001 | Milvus 对象存储 |
| etcd | 2379（内部） | Milvus 元数据 |
| Attu | 3000 | Milvus 可视化管理 |

### 2. 准备本地模型

```bash
ollama pull qwen3-embedding:0.6b   # 条款向量化 / L1 语义路由
ollama pull qwen3.5:0.8b           # L2 意图分类
```

### 3. 启动后端

```bash
cd agent-service
uv sync

# 复制配置模板，填写 DeepSeek / DashScope / MinerU 密钥
cp .env.backup .env

# 构建条款知识库（需数据库已有产品数据）
uv run python scripts/ingest_rag.py

uv run uvicorn app.main:app --reload --port 8001
```

健康检查：<http://127.0.0.1:8001/healthy>，接口文档：<http://127.0.0.1:8001/docs>。

### 4. 启动前端

```bash
cd frontend
npm install
cp .env.example .env.local   # 可选，默认代理到 127.0.0.1:8001
npm run dev
```

访问 <http://localhost:5173>。Vite 会把 `/api` 代理到后端。

## 配置说明

真实配置放在 `agent-service/.env`，**不提交到仓库**；`.env.backup` 为空白模板。主要变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_NAME` / `APP_VERSION` | — / `1.0.0` | 应用名称与版本 |
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `8001` | 服务监听地址 |
| `DB_HOST` / `DB_PORT` / `DB_NAME` | `192.168.10.113` / `5432` / `insurance` | PostgreSQL 连接 |
| `DB_USER` / `DB_PASSWORD` | `insurance` / `insurance123` | 数据库账号（部署时务必修改） |
| `DEEPSEEK_API_KEY` | — | 主聊天模型 Key |
| `DASHSCOPE_API_KEY` | 空 | 兜底模型 / 嵌入 / 重排 Key，缺失时跳过兜底 |
| `MINERU_API_KEY` | — | PDF 解析服务 Key |
| `MILVUS_HOST` / `MILVUS_PORT` | `127.0.0.1` / `19530` | Milvus 地址 |
| `RAG_ENABLE_RERANK` | `true` | 是否启用 DashScope 重排；故障时可关闭 |
| `AGENT_RUN_DURABILITY` | `sync` | checkpoint 持久化强度 |
| `STREAM_BEFORE_OUTPUT_GUARD` | `false` | 是否在输出合规校验前流式推送 |
| `AGENT_RUN_HEARTBEAT_INTERVAL_SECONDS` | `10` | 运行心跳写入间隔 |
| `AGENT_RUN_STALE_SECONDS` | `60` | 心跳超时判定为 worker 失联 |
| `RUN_EVENT_RETENTION_DAYS` | `30` | 运行事件保留天数 |
| `TOOL_INVOCATION_RETENTION_DAYS` | `30` | 工具调用记录保留天数 |
| `AGENT_RUN_RETENTION_DAYS` | `90` | 运行主记录保留天数 |
| `USER_MEMORY_RETENTION_DAYS` | `180` | 情节记忆保留天数 |
| `RETENTION_CLEANUP_INTERVAL_HOURS` | `24` | 过期数据清理周期 |

> 完整配置项（含默认值）见 `agent-service/.env.backup`。

> **不要把真实密钥写入 `.env.backup`**：该文件会被提交到仓库。需要本地留档时使用 `.env.backup.local`，该文件已被 `.gitignore` 忽略。

> 前端演示环境的用户身份来自 `localStorage`，通过 `x-user-id` 请求头传递给后端；生产环境应替换为真实登录鉴权。

## API 概览

所有业务接口前缀为 `/api/v1`，用户相关接口通过请求头 `x-user-id` 识别用户。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthy` | 健康检查 |
| GET | `/api/v1/products` | 产品列表，可按 `category` 过滤 |
| GET | `/api/v1/products/candidates` | 按险种/保费筛选推荐候选 |
| POST | `/api/v1/chat` | 发起对话，返回 SSE 事件流 |
| GET | `/api/v1/chat/runs/{run_id}` | 查询运行状态与结果 |
| GET | `/api/v1/chat/runs/{run_id}/events` | 订阅运行事件（支持 `Last-Event-ID`） |
| POST | `/api/v1/chat/runs/{run_id}/cancel` | 取消运行 |
| POST | `/api/v1/chat/runs/{run_id}/replay` | 基于原消息重放 |
| POST | `/api/v1/chat/runs/{run_id}/retry` | 重试失败/取消的运行 |
| GET | `/api/v1/chat/{thread_id}/events` | 按事件 ID 回放会话事件 |
| GET/PATCH/DELETE | `/api/v1/user-profile` | 查询 / 更正 / 删除用户画像 |
| GET/DELETE | `/api/v1/user-memories` | 查询 / 清空情节记忆 |
| POST/GET | `/api/v1/chat-threads` | 新建 / 列出会话 |
| PATCH/DELETE | `/api/v1/chat-threads/{thread_id}` | 重命名 / 删除会话 |
| GET | `/api/v1/chat-threads/{thread_id}/messages` | 查询会话历史消息 |

### SSE 事件

| 事件 | 载荷 | 说明 |
| --- | --- | --- |
| `run_started` | `{run_id}` | 运行创建成功 |
| `tool_call_started` | `{tool, args, tool_call_id}` | 工具开始执行（参数已截断） |
| `tool_call_completed` | `{tool, tool_call_id, duration_ms}` | 工具执行完成 |
| `tool_call_failed` | `{tool, tool_call_id, error}` | 工具执行失败 |
| `message` | 文本片段 | 经过输出校验的回答分片 |
| `replace` | `{content}` | 输出校验替换了回答，前端覆盖已展示内容 |
| `handoff` | `handoff` | 已记录转人工请求 |
| `done` | `end` | 运行结束 |

## Agent 设计要点

### 意图路由

| 层级 | 方式 | 典型场景 |
| --- | --- | --- |
| L0 | 正则/关键词规则 | “转人工”直接命中；极短模糊输入直接澄清 |
| L1 | embedding 语义匹配 | 与示例问法高度相似且领先第二名足够多 |
| L2 | LLM 结构化输出 | 常规复杂问法，输出意图、置信度、槽位与澄清建议 |
| 兜底 | 关键词 + 上下文继承 | 模型不可用时仍能给出可用路由 |

- 短追问（“多少钱”“多久”）且上一轮有主流程时，继承 `active_workflow`。
- 低置信度或 `fallback` 先澄清；连续澄清超过 2 次转入兜底，避免死循环。
- 阈值位于 `app/agents/routing/router.py`，可用评测脚本重新校准。

### 工具与工作流

| 工作流 | 工具 | 说明 |
| --- | --- | --- |
| 推荐 `recommendation_plan` | `query_candidate_products`、`retrieve_insurance_clause` | 先查候选产品，再按用户画像推荐；条款问题必须检索 |
| 理赔 `claim` | `retrieve_insurance_clause` | 基于条款说明责任、材料与流程，不承诺赔付 |
| 闲聊 / 转人工 / 兜底 | 无 | 轻量文本工作流，主模型失败时切兜底模型 |

工具调用约束：单次运行最多 10 次、单工具最多 3 次；瞬时异常重试 2 次；超时与上游错误统一转为 `ToolTransientError`，最终以结构化 `ToolResult` 返回。

### 安全与合规

- **输入侧**：提示词注入、违法（骗保/伪造材料）、有害内容、他人隐私、保险销售合规、超长输入。
- **输出侧**：保证收益/赔付、绝对化宣传、违法建议、医疗诊断、隐私与系统提示泄露。
- 校验模型故障时：输入侧默认放行，输出侧默认拦截；拦截话术为固定文案，不让校验模型直接生成用户可见内容。
- 过期数据定时清理，单轮对话的用户输入被拦截时跳过画像抽取。

## RAG 检索

### 知识库构建

1. MinerU 解析 `data/raw/kb/*.pdf` 为 Markdown。
2. LLM 修复 PDF 表格导致的标题层级错乱，只保留条款正文。
3. 按 Markdown 标题层级切分父块（写入 `parent_chunks`），再切子块写入 Milvus。
4. 子块同时写入稠密向量与 BM25 稀疏向量，检索时按 `product_id` 过滤。

### 检索优化

- **查询扩展**：把“理赔/报销”扩展为“保险金申请/索赔材料”，“等待期/观察期”扩展为“保险责任开始”等。
- **重排**：DashScope 交叉编码重排；服务异常时自动降级为无重排混合检索。
- **章节降权**：释义、附表、病种、疾病定义等引用章节排在业务章节之后，但仍会回填，避免业务章节不足时结果为空。
- **免责问法特判**：命中“不赔/免责/除外”等词时只补责任免除术语，避免扩展开关把结果带偏。

### 评测结果

评测集为 95 条产品条款问题（按产品 + 主题分层），`k=5`：

| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | 平均延迟 |
| --- | --- | --- | --- | --- | --- |
| 无重排 | 0.8105 | 0.9579 | 0.9789 | 0.8819 | 34 ms |
| DashScope 重排 | 0.7684 | 0.9474 | **1.0000** | 0.8640 | 734 ms |

Ragas 评判（95 条，judge 模型 `deepseek-v4-flash`）：`context_precision = 0.671`，`context_recall = 0.911`。

> 重排能保证 Hit@5 全覆盖，但会降低 Hit@1 并增加约 700ms 延迟；可通过 `RAG_ENABLE_RERANK` 按场景切换。

## 评测与测试

### 意图识别评测

`data/intent_eval.jsonl` 包含 5 类意图各 100 条、共 500 条标注样本。

```bash
cd agent-service

# L1 语义层评测（不依赖聊天模型）
uv run python scripts/intent_eval.py evaluate --mode l1

# L2 评测（需要 Ollama 本地模型，可用 --chat-model 指定）
uv run python scripts/intent_eval.py evaluate --mode l2 --limit 20
uv run python scripts/intent_eval.py evaluate --mode full

# 阈值校准
uv run python scripts/intent_eval.py calibrate --mode l1
uv run python scripts/intent_eval.py calibrate --mode l2
```

### RAG 评测

```bash
cd agent-service

uv run python scripts/rag_eval.py build
uv run python scripts/rag_eval.py evaluate --k 5

# 导出 Ragas 数据（项目环境）
uv run python scripts/ragas_export.py --k 5 --fetch-k 10

# Ragas 评判（隔离环境，避免依赖冲突）
uv run --isolated \
  --with "ragas==0.2.15" \
  --with "langchain==0.3.27" \
  --with "langchain-core==0.3.79" \
  --with "langchain-community==0.3.27" \
  --with "langchain-openai==0.3.35" \
  python scripts/ragas_judge.py
```

### 测试

```bash
cd agent-service

# 离线单元测试（不依赖 Ollama / Postgres / Milvus）
uv run pytest -q

# 集成冒烟测试（需要真实依赖）
RUN_AGENT_INTEGRATION=1 uv run pytest tests/test_agents_integration.py -v -s
```

当前离线测试结果：**164 passed, 1 skipped**。

## 已知限制

- 前端身份为演示实现（`localStorage` + `x-user-id`），未接入真实登录、权限和会话校验。
- 费率表 PDF 已随仓库提供，但当前入库脚本只处理条款 PDF；保费试算仍以产品表字段为准。
- L2 置信度为模型自报值，阈值来自标注集校准，不等于真实概率；问法分布变化后需要重新评测。
- 项目定位为学习/演示工程，生产部署前需补充鉴权、限流、密钥托管、监控与数据脱敏。

## 许可证

暂未指定。
