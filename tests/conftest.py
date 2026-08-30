"""Shared pytest fixtures and optional OpenAI SDK test stub."""

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest


if importlib.util.find_spec("openai") is None:
    openai_stub = types.ModuleType("openai")

    class APIConnectionError(Exception):
        """Minimal stand-in for openai.APIConnectionError."""

    class APIStatusError(Exception):
        """Minimal stand-in for openai.APIStatusError."""

        status_code: int

    class APITimeoutError(Exception):
        """Minimal stand-in for openai.APITimeoutError."""

    class OpenAI:
        """Tiny OpenAI client stub used when the real package is unavailable."""

        def __init__(self, api_key: str, base_url: str | None = None) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.chat = MagicMock()
            self.chat.completions.create = MagicMock()

    openai_stub.APIConnectionError = APIConnectionError
    openai_stub.APIStatusError = APIStatusError
    openai_stub.APITimeoutError = APITimeoutError
    openai_stub.OpenAI = OpenAI
    sys.modules["openai"] = openai_stub


from config.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    """Prevent cached settings from leaking between tests."""

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    """Return deterministic settings that never require a real API key."""

    return Settings(
        api_key="sk-test",
        base_url="http://localhost:9999",
        model_name="fake-model",
        history_path=tmp_path / "history.jsonl",
    )


@pytest.fixture
def temp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point tool path safety checks at an isolated temporary project root."""

    import agent.tools as tools_module

    monkeypatch.setattr(tools_module, "PROJECT_ROOT", tmp_path)
    yield tmp_path


@pytest.fixture
def mock_chat_message() -> MagicMock:
    """Return a generic assistant message mock."""

    return MagicMock()


def make_mock_response(message: object) -> MagicMock:
    """Build an OpenAI-compatible chat completion response mock."""

    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response
