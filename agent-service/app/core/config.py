"""应用配置"""

from pathlib import Path

from pydantic import Field, computed_field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# 以当前文件为基准，向上找两级的父目录，也就是项目的根目录
APP_ROOT = Path(__file__).resolve().parents[2]


# print(APP_ROOT) # E:\practice_problems\agent_insurance

class EnvSetting(BaseSettings):
    """公共 settings，配置 .env 文件位置"""

    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # settings 中未定的字段，忽略掉
    )


class AppSetting(EnvSetting):
    """应用相关设置"""

    name: str = Field(alias="APP_NAME", description="应用名称")
    version: str = Field(alias="APP_VERSION", default="1.0.0", description="应用版本")
    host: str = Field(alias="APP_HOST", default="127.0.0.1", description="应用主机地址")
    port: int = Field(alias="APP_PORT", default=8001, description="应用端口")
    debug: bool = Field(alias="APP_DEBUG", default=True, description="是否开启FastAPI调试模式")


class DatabaseSetting(EnvSetting):
    """数据库相关设置"""

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
        """构建数据库连接字符串"""

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
        # 自动拼接 URL 路径
        return self.build_url("postgresql+asyncpg")

    # 针对 LangChain 会话链接使用
    @computed_field
    @property
    def checkpoint_url(self) -> str:
        # 自动拼接 URL 路径
        return self.build_url("postgresql")


class LLMSetting(EnvSetting):
    """LLM 相关设置"""

    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY", description="Deepseek API Key")
    deepseek_chat_model: str = Field(alias="DEEPSEEK_CHAT_MODEL", default="deepseek-v4-flash", description="Deepseek 大语言模型")
    dashscope_chat_model: str = Field(alias="DASHSCOPE_CHAT_MODEL", description="DASHSCOPE 大语言模型")
    dashscope_embedding_model: str = Field(alias="DASHSCOPE_EMBEDDING_MODEL", description="DASHSCOPE 嵌入模型")
    dashscope_rerank_model: str = Field(alias="DASHSCOPE_RERANK_MODEL", description="DASHSCOPE 重排序模型")


class RAGSetting(EnvSetting):
    """RAG 相关设置"""

    mineru_token: str = Field(alias="MINERU_API_KEY", description="Mineru API Key")
    milvus_host: str = Field(alias="MILVUS_HOST", default="127.0.0.1", description="Milvus 主机地址")
    milvus_port: int = Field(alias="MILVUS_PORT", default=19530, description="Milvus 端口")

    @computed_field
    @property
    def milvus_url(self) -> str:
        # http://192.168.10.113:19530/
        return f"http://{self.milvus_host}:{self.milvus_port}"


class LoggingSetting(EnvSetting):
    """日志相关设置"""

    level: str = Field(alias="LOGGING_LEVEL", default="INFO", description="日志级别")


class Settings(BaseSettings):
    """汇总配置"""

    app: AppSetting = Field(default_factory=AppSetting, description="应用相关设置")
    db: DatabaseSetting = Field(default_factory=DatabaseSetting, description="数据库相关设置")
    llm: LLMSetting = Field(default_factory=LLMSetting, description="LLM 相关设置")
    logging: LoggingSetting = Field(default_factory=LoggingSetting, description="日志相关设置")
    rag: RAGSetting = Field(default_factory=RAGSetting, description="RAG相关设置")


# 项目启动初始化
settings = Settings()
