"""Application settings loaded from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for OpenAI-compatible chat completion APIs."""

    api_key: str
    base_url: str | None = None
    model_name: str = "gpt-4o-mini"
    history_path: Path = Path("history.jsonl")
    max_iterations: int = 8

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings, using .env when it exists as a regular file."""

    env_path = Path(".env")
    if env_path.is_file():
        return Settings(_env_file=env_path, _env_file_encoding="utf-8")
    return Settings()
