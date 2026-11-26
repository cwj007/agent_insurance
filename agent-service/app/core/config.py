"""应用配置。"""

from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# 以当前文件为基准，向上找两级的父目录，也就是项目的根目录
APP_ROOT = Path(__file__).resolve().parents[2]


class EnvSetting(BaseSettings):
    """公共 settings，配置 .env 文件位置。"""

    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # settings 中未定的字段，忽略掉
    )


class AppSetting(EnvSetting):
    """应用相关设置。"""

    name: str = Field(alias="APP_NAME", description="应用名称")
    version: str = Field(alias="APP_VERSION", default="1.0.0", description="应用版本")
    host: str = Field(alias="APP_HOST", default="127.0.0.1", description="应用主机地址")
    port: int = Field(alias="APP_PORT", default=8001, description="应用端口")
    debug: bool = Field(alias="APP_DEBUG", default=True, description="是否开启FastAPI调试模式")


class DatabaseSetting(EnvSetting):
    """数据库相关设置。"""

    host: str = Field(alias="DB_HOST", default="192.168.10.113", description="数据库主机地址")
    port: int = Field(alias="DB_PORT", default=5432, description="数据库端口")
    name: str = Field(alias="DB_NAME", default="insurance", description="数据库名称")
    user: str = Field(alias="DB_USER", default="insurance", description="数据库用户名")
    password: str = Field(alias="DB_PASSWORD", default="insurance123", description="数据库密码")
    echo: bool = Field(alias="DB_ECHO", default=False, description="是否输出数据库语句")
    pool_size: int = Field(alias="DB_POOL_SIZE", default=5, description="数据库连接池大小")
    max_overflow: int = Field(alias="DB_MAX_OVERFLOW", default=10, description="数据库最大溢出连接数")
    pool_timeout: int = Field(alias="DB_POOL_TIMEOUT", default=30, description="数据库连接池超时时间")
    pool_pre_ping: bool = Field(alias="DB_POOL_PRE_PING", default=True, description="是否开启数据库连接池预连接")
    pool_recycle: int = Field(alias="DB_POOL_RECYCLE", default=1800, description="数据库连接池回收时间")

    # 数据库连接字符串，针对 LangChain 会话以及后端数据的存储
    def build_url(self, scheme: str):
        """构建数据库连接字符串。

        Args:
            scheme: 数据库连接协议，如 postgresql+asyncpg、postgresql。

        Returns:
            拼接完成并转义后的数据库连接字符串。
        """
        return PostgresDsn.build(
            scheme=scheme,
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
        ).encoded_string()

    # 数据库链接使用
    @computed_field
    @property
    def url(self) -> str:
        """获取异步数据库连接字符串。

        Returns:
            使用 postgresql+asyncpg 协议的数据库连接字符串。
        """
        # 自动拼接 URL 路径
        return self.build_url("postgresql+asyncpg")

    # 针对 LangChain 会话链接使用
    @computed_field
    @property
    def checkpoint_url(self) -> str:
        """获取 LangChain 会话记忆使用的数据库连接字符串。

        Returns:
            使用 postgresql 协议的数据库连接字符串。
        """
        # 自动拼接 URL 路径
        return self.build_url("postgresql")


class LLMSetting(EnvSetting):
    """LLM 相关设置。"""

    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY", description="Deepseek API Key")
    deepseek_chat_model: str = Field(alias="DEEPSEEK_CHAT_MODEL", default="deepseek-v4-flash", description="Deepseek 大语言模型")
    dashscope_api_key: str | None = Field(
        alias="DASHSCOPE_API_KEY",
        default=None,
        description="DASHSCOPE API Key，可选；用于主模型故障时的兜底模型",
    )
    dashscope_chat_model: str = Field(alias="DASHSCOPE_CHAT_MODEL", description="DASHSCOPE 大语言模型")
    dashscope_embedding_model: str = Field(alias="DASHSCOPE_EMBEDDING_MODEL", description="DASHSCOPE 嵌入模型")
    dashscope_rerank_model: str = Field(alias="DASHSCOPE_RERANK_MODEL", description="DASHSCOPE 重排序模型")

    # 本地 Ollama 与模型可靠性设置
    ollama_base_url: str = Field(
        alias="OLLAMA_BASE_URL",
        default="http://127.0.0.1:11434",
        description="本地 Ollama 服务地址，供意图分类与嵌入使用",
    )
    intent_model_name: str = Field(
        alias="INTENT_MODEL_NAME",
        default="qwen3.5:0.8b",
        description="L2 意图分类使用的本地 Ollama 模型",
    )
    retry_attempts: int = Field(
        alias="LLM_RETRY_ATTEMPTS",
        default=2,
        ge=1,
        description="主模型通用重试次数（含首次调用）",
    )
    retry_backoff_seconds: float = Field(
        alias="LLM_RETRY_BACKOFF_SECONDS",
        default=0.3,
        ge=0,
        description="主模型重试退避基数（秒），按尝试次数线性增长",
    )
    breaker_failures: int = Field(
        alias="LLM_BREAKER_FAILURES",
        default=3,
        ge=1,
        description="主模型连续失败多少次后触发熔断",
    )
    breaker_cooldown_seconds: float = Field(
        alias="LLM_BREAKER_COOLDOWN_SECONDS",
        default=60.0,
        ge=0,
        description="主模型熔断后的冷却时间（秒）",
    )


class RAGSetting(EnvSetting):
    """RAG 相关设置。"""

    mineru_token: str = Field(alias="MINERU_API_KEY", description="Mineru API Key")
    milvus_host: str = Field(alias="MILVUS_HOST", default="127.0.0.1", description="Milvus 主机地址")
    milvus_port: int = Field(alias="MILVUS_PORT", default=19530, description="Milvus 端口")
    enable_rerank: bool = Field(
        alias="RAG_ENABLE_RERANK",
        default=True,
        description="条款检索是否启用 DashScope 重排；重排服务故障时可设为 false",
    )
    embedding_model: str = Field(
        alias="EMBEDDING_MODEL_NAME",
        default="qwen3-embedding:0.6b",
        description="本地 Ollama 嵌入模型",
    )
    embedding_dimensions: int = Field(
        alias="EMBEDDING_DIMENSIONS",
        default=1024,
        ge=1,
        description="嵌入向量维度，需与 Milvus collection 保持一致",
    )
    collection_name: str = Field(
        alias="MILVUS_COLLECTION_NAME",
        default="insurance_collection",
        description="Milvus collection 名称",
    )

    @computed_field
    @property
    def milvus_url(self) -> str:
        """获取 Milvus 服务地址。

        Returns:
            形如 http://host:port 的 Milvus 地址。
        """
        # http://192.168.10.113:19530/
        return f"http://{self.milvus_host}:{self.milvus_port}"


class LoggingSetting(EnvSetting):
    """日志相关设置。"""

    level: str = Field(alias="LOGGING_LEVEL", default="INFO", description="日志级别")


class AgentRunSetting(EnvSetting):
    """Agent 运行设置。"""

    durability: Literal["sync", "async", "exit"] = Field(
        alias="AGENT_RUN_DURABILITY",
        default="sync",
        description="checkpoint 持久化强度；写操作建议 sync",
    )
    stream_before_output_guard: bool = Field(
        alias="STREAM_BEFORE_OUTPUT_GUARD",
        default=False,
        description="是否在输出安全校验前流式推送回答；默认关闭以保证合规",
    )
    heartbeat_interval_seconds: float = Field(
        alias="AGENT_RUN_HEARTBEAT_INTERVAL_SECONDS",
        default=10.0,
        description="运行心跳写入间隔秒数",
    )
    stale_run_seconds: float = Field(
        alias="AGENT_RUN_STALE_SECONDS",
        default=60.0,
        description="心跳超过该秒数视为 worker 失联",
    )
    cancel_poll_interval_seconds: float = Field(
        alias="AGENT_RUN_CANCEL_POLL_INTERVAL_SECONDS",
        default=2.0,
        description="跨 worker 取消请求的轮询间隔秒数",
    )


class RetentionSetting(EnvSetting):
    """过期数据清理设置。"""

    run_event_days: int = Field(
        alias="RUN_EVENT_RETENTION_DAYS",
        default=30,
        description="运行事件保留天数",
    )
    tool_invocation_days: int = Field(
        alias="TOOL_INVOCATION_RETENTION_DAYS",
        default=30,
        description="工具调用记录保留天数",
    )
    agent_run_days: int = Field(
        alias="AGENT_RUN_RETENTION_DAYS",
        default=90,
        description="运行主记录保留天数",
    )
    user_memory_days: int = Field(
        alias="USER_MEMORY_RETENTION_DAYS",
        default=180,
        description="用户情节记忆保留天数",
    )
    cleanup_interval_hours: int = Field(
        alias="RETENTION_CLEANUP_INTERVAL_HOURS",
        default=24,
        description="清理任务运行间隔小时数",
    )


class ToolSetting(EnvSetting):
    """工具调用设置。"""

    timeout_seconds: float = Field(
        alias="TOOL_TIMEOUT_SECONDS",
        default=10.0,
        gt=0,
        description="单次工具执行超时秒数",
    )
    global_run_limit: int = Field(
        alias="TOOL_GLOBAL_RUN_LIMIT",
        default=10,
        ge=1,
        description="单次运行的工具调用总上限",
    )
    per_tool_run_limit: int = Field(
        alias="TOOL_PER_TOOL_RUN_LIMIT",
        default=3,
        ge=1,
        description="单次运行中单个工具的调用上限",
    )
    max_retries: int = Field(
        alias="TOOL_MAX_RETRIES",
        default=2,
        ge=0,
        description="工具瞬时异常的重试次数",
    )
    retry_initial_delay: float = Field(
        alias="TOOL_RETRY_INITIAL_DELAY",
        default=0.3,
        ge=0,
        description="工具重试初始退避秒数",
    )
    retry_max_delay: float = Field(
        alias="TOOL_RETRY_MAX_DELAY",
        default=2.0,
        ge=0,
        description="工具重试最大退避秒数",
    )


class ContextSetting(EnvSetting):
    """会话上下文治理设置。"""

    summary_trigger_messages: int = Field(
        alias="SUMMARY_TRIGGER_MESSAGES",
        default=40,
        ge=1,
        description="主图历史消息超过该数量时触发摘要",
    )
    summary_keep_messages: int = Field(
        alias="SUMMARY_KEEP_MESSAGES",
        default=12,
        ge=1,
        description="摘要后保留的最近消息条数",
    )
    subagent_summary_trigger_messages: int = Field(
        alias="SUBAGENT_SUMMARY_TRIGGER_MESSAGES",
        default=60,
        ge=1,
        description="子 Agent 历史消息超过该数量时触发摘要",
    )
    subagent_summary_keep_messages: int = Field(
        alias="SUBAGENT_SUMMARY_KEEP_MESSAGES",
        default=12,
        ge=1,
        description="子 Agent 摘要后保留的最近消息条数",
    )


class GuardrailSetting(EnvSetting):
    """输入输出安全校验设置。"""

    max_input_length: int = Field(
        alias="MAX_INPUT_LENGTH",
        default=4000,
        ge=1,
        description="用户输入最大字符数，超长直接拒绝",
    )
    input_fail_open: bool = Field(
        alias="INPUT_GUARD_FAIL_OPEN",
        default=True,
        description="输入校验模型故障时是否放行（保可用）",
    )
    output_fail_open: bool = Field(
        alias="OUTPUT_GUARD_FAIL_OPEN",
        default=False,
        description="输出校验模型故障时是否放行（默认拦截保合规）",
    )


class ChatSetting(EnvSetting):
    """对话流式输出设置。"""

    stream_chunk_size: int = Field(
        alias="STREAM_CHUNK_SIZE",
        default=2,
        ge=1,
        description="SSE 回答分片字符数",
    )
    stream_chunk_delay_seconds: float = Field(
        alias="STREAM_CHUNK_DELAY_SECONDS",
        default=0.025,
        ge=0,
        description="每个 SSE 分片之间的间隔秒数",
    )
    stream_max_total_delay_seconds: float = Field(
        alias="STREAM_MAX_TOTAL_DELAY_SECONDS",
        default=2.5,
        ge=0,
        description="单轮回答分片推送的总延迟上限秒数",
    )


class AgentSetting(EnvSetting):
    """Agent 流程设置。"""

    max_clarification_rounds: int = Field(
        alias="MAX_CLARIFICATION_ROUNDS",
        default=2,
        ge=0,
        description="连续澄清最大轮数，超过后进入兜底",
    )
    enable_combined_input_analysis: bool = Field(
        alias="ENABLE_COMBINED_INPUT_ANALYSIS",
        default=True,
        description="是否把输入安全校验与 L2 意图分类合并为一次模型调用",
    )


class Settings(BaseSettings):
    """汇总配置。"""

    app: AppSetting = Field(default_factory=AppSetting, description="应用相关设置")
    db: DatabaseSetting = Field(default_factory=DatabaseSetting, description="数据库相关设置")
    llm: LLMSetting = Field(default_factory=LLMSetting, description="LLM 相关设置")
    logging: LoggingSetting = Field(default_factory=LoggingSetting, description="日志相关设置")
    rag: RAGSetting = Field(default_factory=RAGSetting, description="RAG相关设置")
    agent_run: AgentRunSetting = Field(
        default_factory=AgentRunSetting,
        description="Agent 运行设置",
    )
    retention: RetentionSetting = Field(
        default_factory=RetentionSetting,
        description="过期数据清理设置",
    )
    tools: ToolSetting = Field(
        default_factory=ToolSetting,
        description="工具调用设置",
    )
    context: ContextSetting = Field(
        default_factory=ContextSetting,
        description="会话上下文治理设置",
    )
    guardrails: GuardrailSetting = Field(
        default_factory=GuardrailSetting,
        description="输入输出安全校验设置",
    )
    chat: ChatSetting = Field(
        default_factory=ChatSetting,
        description="对话流式输出设置",
    )
    agent: AgentSetting = Field(
        default_factory=AgentSetting,
        description="Agent 流程设置",
    )


# 项目启动初始化
settings = Settings()
