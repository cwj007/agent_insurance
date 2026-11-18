# Agent Insurance（保险智能商城）

基于 LLM Agent 的保险产品咨询与选购平台。前端已完成，后端（`agent-service`）仍在开发中。

## 目录结构

```
agent_insurance/
├── frontend/        # 前端：React 18 + TypeScript + Vite + Ant Design
├── agent-service/   # 后端：FastAPI + LangChain / LangGraph + PostgreSQL + Milvus
└── deploy/          # 本地基础设施：Docker Compose（PostgreSQL、Milvus、MinIO、etcd、Attu）
```

## 技术栈

**前端**：React 18、TypeScript、Vite 6、Ant Design 5

**后端**：Python 3.14+、FastAPI、LangChain、LangGraph、SQLAlchemy (asyncpg)、Milvus、uv 包管理

**基础设施**：PostgreSQL 16、Milvus 3.0、MinIO、etcd、Attu

## 快速开始

### 1. 启动基础设施

```bash
cd deploy
docker compose up -d
```

启动后将运行：

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| PostgreSQL | 5432 | 业务数据库，默认库/用户/密码见 `docker-compose.yml` |
| Milvus | 19530 / 9091 | 向量数据库 |
| MinIO | 9000 / 9001 | 对象存储（Milvus 依赖） |
| Attu | 3000 | Milvus 可视化管理界面 |

### 2. 启动后端（开发中）

```bash
cd agent-service
uv sync
# 复制模板并填写密钥（DEEPSEEK_API_KEY、DASHSCOPE_API_KEY、MINERU_TOKEN 等）
cp .env.backup .env
uv run uvicorn app.main:app --reload
```

### 3. 启动前端

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

## 环境变量

- `agent-service/.env`：真实配置，**不提交到仓库**。
- `agent-service/.env.backup`：配置模板（密钥为空），包含 `APP_*`、`DB_*`、`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`MINERU_TOKEN`、`MILVUS_*` 等变量。
- `frontend/.env.example`：前端模板，包含 `VITE_BACKEND_HOST`、`VITE_BACKEND_PORT`。

## 开发状态

- [x] 前端：页面、鉴权弹窗、产品卡片、顾问对话等
- [ ] 后端：基础框架（配置、日志、数据库、产品模块）已搭建，Agent 能力开发中

## 许可证

暂未指定。
