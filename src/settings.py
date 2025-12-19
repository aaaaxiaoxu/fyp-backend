from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    APP_BASE_URL: str = "http://127.0.0.1:8000"

    SQLITE_PATH: str = "data/app.db"
    SQLITE_ECHO: bool = False

    # JWT
    JWT_SECRET_KEY: str = "change_me"
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 20
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14

    # Cookies
    ACCESS_COOKIE_NAME: str = "access_token"
    REFRESH_COOKIE_NAME: str = "refresh_token"
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # "lax" | "strict" | "none"

    # SMTP
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""
    SMTP_TLS: bool = True

    # LLM
    LLM_BASE_URL: str = "https://api.deepseek.com"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-chat"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1400
    TIMEOUT_S: int = 60
    MAX_RETRIES: int = 5
    RETRY_BACKOFF_S: float = 1.5

    # Neo4j
    NEO4J_URI: str = "neo4j://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    


    # Graph API limits
    GRAPH_MAX_DEPTH: int = 3
    GRAPH_MAX_PATHS: int = 500
    GRAPH_MAX_SNIPPET_LEN: int = 300


settings = Settings()
