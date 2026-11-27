# 效果展示截图

本目录存放根目录 `README.md`「效果展示」章节使用的项目截图。

## 命名规范

| 文件 | 内容 | 优先级 |
|---|---|---|
| `01-langgraph.png` | LangGraph 主图：意图路由、工作流与统一输出护栏 | 必截 |
| `02-intent-eval.png` | 意图评测/阈值校准：precision、recall、coverage | 必截 |
| `03-tool-invocations.png` | 工具调用记录：成功/失败、错误码、耗时 | 必截 |
| `04-run-management.png` | `agent_runs` + `run_events`，以及 cancel/replay/retry API | 必截 |
| `05-context-summary.png` | 长对话摘要前后对比：40 条触发、保留 12 条 | 必截 |
| `06-swagger.png` | FastAPI Swagger 接口总览 | 建议 |
| `07-frontend-chat.png` | 前端 AI 顾问对话、工具进度与转人工 | 建议 |
| `08-guardrails.png` | 输入/输出护栏拦截示例 | 建议 |
| `09-rag-eval.png` | RAG 检索结果或 Hit@k / MRR 评测 | 建议 |
| `10-user-profile.png` | 用户画像/情节记忆：source、needs_confirmation、version | 可选 |

## 截图要求

- 统一 16:9 或 4:3，2x 分辨率，浅色主题，避免桌面壁纸和杂乱标签页。
- 评测截图要包含命令和输出，证明可复现。
- 数据库截图要露出表头，敏感字段打码。
- 不出现 `.env`、API Key、数据库密码、内网 IP、真实用户手机号/健康信息/保单号。
- 不截大段代码；代码直接放 GitHub 链接更清晰。

## LangGraph 主图

`01-langgraph.png` 为静态架构图，随仓库提供，不再由应用启动时自动生成。

## 上传后

把根目录 `README.md`「效果展示」里的图片注释块取消注释，GitHub 即可直接渲染。
