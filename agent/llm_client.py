"""Thin OpenAI-compatible chat completion client."""

import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from config.settings import Settings

MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 60


class LLMClient:
    """Wraps the OpenAI SDK without taking over tool execution."""

    def __init__(self, settings: Settings) -> None:
        client_kwargs: dict[str, Any] = {"api_key": settings.api_key}
        if settings.base_url:
            client_kwargs["base_url"] = settings.base_url

        self.model_name = settings.model_name
        self.client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Create one chat completion with optional native tool schemas."""

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "timeout": REQUEST_TIMEOUT_SECONDS,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        for attempt in range(MAX_RETRIES + 1):
            try:
                response: Any = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message
            except APIStatusError as exc:
                if 400 <= exc.status_code < 500:
                    raise
                if attempt >= MAX_RETRIES:
                    raise
                self._print_retry(attempt + 1)
                self._sleep_before_retry(attempt)
            except (APIConnectionError, APITimeoutError):
                if attempt >= MAX_RETRIES:
                    raise
                self._print_retry(attempt + 1)
                self._sleep_before_retry(attempt)

        raise RuntimeError("API call failed after retry handling completed")

    def _print_retry(self, retry_number: int) -> None:
        """Print a retry notice before the next API request."""

        print(f"Retrying API call (attempt {retry_number}/{MAX_RETRIES})...")

    def _sleep_before_retry(self, attempt: int) -> None:
        """Sleep with exponential backoff before retrying."""

        delay_seconds: int = 2**attempt
        time.sleep(delay_seconds)
