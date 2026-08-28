"""Thin OpenAI-compatible chat completion client."""

from typing import Any

from openai import OpenAI

from config.settings import Settings


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
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message
