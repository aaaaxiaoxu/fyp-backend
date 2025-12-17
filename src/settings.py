from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # -----------------------
    # LLM (DeepSeek / OpenAI-compatible)
    # -----------------------
    LLM_BASE_URL: str = "https://api.deepseek.com"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-chat"

    # DeepSeek / OpenAI SDK params
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1400
    TIMEOUT_S: int = 60
    MAX_RETRIES: int = 5
    RETRY_BACKOFF_S: float = 1.5

    # -----------------------
    # Neo4j
    # -----------------------
    NEO4J_URI: str = "neo4j://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    # 可选：有些环境需要指定数据库名（Neo4j 4+）
    NEO4J_DATABASE: str = "neo4j"


    SQLITE_PATH: str = "data/app.db"
    SQLITE_ECHO: bool = False


    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
