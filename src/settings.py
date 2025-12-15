from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    LLM_BASE_URL: str = "https://api.deepseek.com"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-chat"

    # DeepSeek / OpenAI SDK params
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1400
    TIMEOUT_S: int = 60
    MAX_RETRIES: int = 5
    RETRY_BACKOFF_S: float = 1.5

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
