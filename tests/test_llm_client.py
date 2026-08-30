"""Tests for agent.llm_client module."""

from unittest.mock import MagicMock, patch

import pytest

from agent.llm_client import LLMClient
from config.settings import Settings


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(
        api_key="sk-test",
        base_url="http://localhost:9999",
        model_name="fake-model",
    )


class TestLLMClientInit:
    """Client initialization from settings."""

    def test_stores_model_name(self, fake_settings: Settings) -> None:
        client = LLMClient(fake_settings)
        assert client.model_name == "fake-model"

    def test_forwards_api_key(self, fake_settings: Settings) -> None:
        client = LLMClient(fake_settings)
        assert client.client.api_key == "sk-test"

    def test_forwards_base_url(self, fake_settings: Settings) -> None:
        client = LLMClient(fake_settings)
        # openai 库字符串化 base_url 时不带尾部斜杠
        assert str(client.client.base_url) == "http://localhost:9999"


class TestChatSuccess:
    """Normal completion paths."""

    def _make_mock_response(self, message: MagicMock | None = None) -> MagicMock:
        """Helper to build a mock chat.completions.create return value."""
        if message is None:
            message = MagicMock()
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_chat_without_tools(self, fake_settings: Settings) -> None:
        client = LLMClient(fake_settings)
        expected_msg = MagicMock()
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = self._make_mock_response(
            expected_msg
        )

        result = client.chat([{"role": "user", "content": "hello"}])

        assert result is expected_msg
        client.client.chat.completions.create.assert_called_once()
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "fake-model"
        assert kwargs["timeout"] == 60
        assert "tools" not in kwargs

    def test_chat_with_tools(self, fake_settings: Settings) -> None:
        client = LLMClient(fake_settings)
        expected_msg = MagicMock()
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = self._make_mock_response(
            expected_msg
        )

        tools = [{"type": "function", "function": {"name": "read_file"}}]
        result = client.chat([{"role": "user", "content": "hi"}], tools=tools)

        assert result is expected_msg
        kwargs = client.client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


class TestChatRetryOnServerError:
    """5xx and transient errors trigger retries."""

    def _make_api_status_error(self, status_code: int):
        """Build a real APIStatusError instance that can be raised by mock."""
        from openai import APIStatusError

        class FakeAPIStatusError(APIStatusError):
            def __init__(self, code: int):
                self.status_code = code
                self.message = "fake server error"
                self.body = None
                self.code = None
                self.param = None
                self.request_id = None
                self.response = None

        return FakeAPIStatusError(status_code)

    def test_retry_once_on_503_then_success(self, fake_settings: Settings, capsys) -> None:
        """A single 503 followed by success should retry and return the good result."""
        client = LLMClient(fake_settings)
        exc_503 = self._make_api_status_error(503)
        expected_msg = MagicMock()
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = [
            exc_503,
            TestChatSuccess()._make_mock_response(expected_msg),
        ]

        with patch("agent.llm_client.time.sleep") as mock_sleep:
            result = client.chat([{"role": "user", "content": "test"}])

        assert result is expected_msg
        assert client.client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(1)
        captured = capsys.readouterr()
        assert "Retrying API call (attempt 1/3)" in captured.out

    def test_retry_three_times_then_fail(self, fake_settings: Settings) -> None:
        """All retries exhausted on persistent 503."""
        from openai import APIStatusError

        client = LLMClient(fake_settings)
        exc = self._make_api_status_error(503)
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = [exc, exc, exc, exc]

        with patch("agent.llm_client.time.sleep"):
            with pytest.raises(APIStatusError):
                client.chat([{"role": "user", "content": "test"}])

        assert client.client.chat.completions.create.call_count == 4

    def test_retry_on_connection_error(self, fake_settings: Settings) -> None:
        """APIConnectionError is retried."""
        from openai import APIConnectionError

        client = LLMClient(fake_settings)

        class FakeConnError(APIConnectionError):
            def __init__(self):
                self.message = "connection failed"
                self.request = None

        exc = FakeConnError()
        expected_msg = MagicMock()
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = [
            exc,
            TestChatSuccess()._make_mock_response(expected_msg),
        ]

        with patch("agent.llm_client.time.sleep"):
            result = client.chat([{"role": "user", "content": "test"}])

        assert result is expected_msg
        assert client.client.chat.completions.create.call_count == 2

    def test_retry_on_timeout(self, fake_settings: Settings) -> None:
        """APITimeoutError is retried."""
        from openai import APITimeoutError

        client = LLMClient(fake_settings)

        class FakeTimeoutError(APITimeoutError):
            def __init__(self):
                self.message = "timed out"
                self.request = None

        exc = FakeTimeoutError()
        expected_msg = MagicMock()
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = [
            exc,
            exc,
            TestChatSuccess()._make_mock_response(expected_msg),
        ]

        with patch("agent.llm_client.time.sleep"):
            result = client.chat([{"role": "user", "content": "test"}])

        assert result is expected_msg
        assert client.client.chat.completions.create.call_count == 3


class TestChatNoRetry:
    """Client errors (4xx) must not be retried."""

    def _make_api_status_error(self, status_code: int):
        from openai import APIStatusError

        class FakeAPIStatusError(APIStatusError):
            def __init__(self, code: int):
                self.status_code = code
                self.message = "fake client error"
                self.body = None
                self.code = None
                self.param = None
                self.request_id = None
                self.response = None

        return FakeAPIStatusError(status_code)

    def test_400_bad_request_no_retry(self, fake_settings: Settings) -> None:
        client = LLMClient(fake_settings)
        exc = self._make_api_status_error(400)
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = exc

        with pytest.raises(Exception) as exc_info:
            client.chat([{"role": "user", "content": "test"}])

        assert exc_info.value.status_code == 400
        client.client.chat.completions.create.assert_called_once()

    def test_429_rate_limit_no_retry(self, fake_settings: Settings) -> None:
        """Current implementation treats 429 as a non-retryable 4xx."""
        client = LLMClient(fake_settings)
        exc = self._make_api_status_error(429)
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = exc

        with pytest.raises(Exception) as exc_info:
            client.chat([{"role": "user", "content": "test"}])

        assert exc_info.value.status_code == 429
        client.client.chat.completions.create.assert_called_once()


class TestExponentialBackoff:
    """Sleep timing between retries."""

    def test_backoff_sequence_1_2_4(self, fake_settings: Settings) -> None:
        from openai import APIConnectionError

        client = LLMClient(fake_settings)

        class FakeConnError(APIConnectionError):
            def __init__(self):
                self.message = "conn failed"
                self.request = None

        exc = FakeConnError()
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = [
            exc,
            exc,
            exc,
            TestChatSuccess()._make_mock_response(MagicMock()),
        ]

        with patch("agent.llm_client.time.sleep") as mock_sleep:
            client.chat([{"role": "user", "content": "test"}])

        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)