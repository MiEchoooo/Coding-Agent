"""Tests for config.settings module."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep Settings tests independent from the developer shell environment."""

    for name in ("API_KEY", "BASE_URL", "MODEL_NAME", "HISTORY_PATH", "MAX_ITERATIONS"):
        monkeypatch.delenv(name, raising=False)
    yield


class TestSettingsFromEnvironment:
    """Settings loaded from environment variables."""

    def test_all_fields_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Load every field from environment variables."""
        monkeypatch.setenv("API_KEY", "sk-test-key")
        monkeypatch.setenv("BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("MODEL_NAME", "qwen-max")
        monkeypatch.setenv("HISTORY_PATH", "custom_history.jsonl")
        monkeypatch.setenv("MAX_ITERATIONS", "15")

        s = Settings()

        assert s.api_key == "sk-test-key"
        assert s.base_url == "https://test.example.com/v1"
        assert s.model_name == "qwen-max"
        assert s.history_path == Path("custom_history.jsonl")
        assert s.max_iterations == 15

    def test_api_key_is_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing API_KEY must raise a validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "api_key" in str(exc_info.value)

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Optional fields fall back to sensible defaults."""
        monkeypatch.setenv("API_KEY", "sk-test-key")
        # Deliberately omit BASE_URL, MODEL_NAME, HISTORY_PATH, MAX_ITERATIONS

        s = Settings()

        assert s.base_url is None
        assert s.model_name == "gpt-4o-mini"
        assert s.history_path == Path("history.jsonl")
        assert s.max_iterations == 8

    def test_extra_fields_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown env vars must not break initialization (extra='ignore')."""
        monkeypatch.setenv("API_KEY", "sk-test-key")
        monkeypatch.setenv("UNKNOWN_VAR", "should_be_ignored")

        # Should not raise
        s = Settings()
        assert s.api_key == "sk-test-key"


class TestSettingsFromDotenv:
    """Settings loaded from a local .env file."""

    def test_load_from_dotenv_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_settings() reads a .env file when it exists."""
        # Clear any cached settings and env vars first
        get_settings.cache_clear()
        monkeypatch.delenv("API_KEY", raising=False)

        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_file.write_text(
            "API_KEY=sk-from-dotenv\n"
            "BASE_URL=https://dotenv.example.com\n"
            "MODEL_NAME=deepseek-chat\n",
            encoding="utf-8",
        )

        # Change working directory so get_settings() finds our temp .env
        monkeypatch.chdir(tmp_path)

        s = get_settings()

        assert s.api_key == "sk-from-dotenv"
        assert s.base_url == "https://dotenv.example.com"
        assert s.model_name == "deepseek-chat"

    def test_fallback_when_no_dotenv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_settings() falls back to pure env vars when .env is absent."""
        get_settings.cache_clear()
        monkeypatch.chdir(tmp_path)  # tmp_path has no .env file
        monkeypatch.setenv("API_KEY", "sk-fallback")

        s = get_settings()

        assert s.api_key == "sk-fallback"


class TestSettingsCaching:
    """The lru_cache wrapper around get_settings()."""

    def test_caching_returns_same_instance(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling get_settings() twice must return the same cached object."""
        get_settings.cache_clear()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("API_KEY", "sk-cache-test")

        s1 = get_settings()
        s2 = get_settings()

        assert s1 is s2

    def test_cache_clear_allows_reload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """After cache_clear(), a new instance is created."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("API_KEY", "sk-before")

        s1 = get_settings()

        # Simulate a config change (normally not recommended mid-process,
        # but useful to prove caching behaviour)
        monkeypatch.setenv("API_KEY", "sk-after")
        s2 = get_settings()

        # Still cached old value
        assert s2.api_key == "sk-before"

        # Clear and reload
        get_settings.cache_clear()
        s3 = get_settings()
        assert s3.api_key == "sk-after"


class TestSettingsTypes:
    """Type correctness of parsed values."""

    def test_history_path_is_path_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HISTORY_PATH must be parsed into a pathlib.Path instance."""
        monkeypatch.setenv("API_KEY", "sk-test")
        monkeypatch.setenv("HISTORY_PATH", "logs/history.jsonl")

        s = Settings()
        assert isinstance(s.history_path, Path)
        assert str(s.history_path) == "logs" + os.sep + "history.jsonl"

    def test_max_iterations_int_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_ITERATIONS supplied as a string must be coerced to int."""
        monkeypatch.setenv("API_KEY", "sk-test")
        monkeypatch.setenv("MAX_ITERATIONS", "20")

        s = Settings()
        assert isinstance(s.max_iterations, int)
        assert s.max_iterations == 20
