"""Parsing helpers for OpenAI-compatible tool-calling responses."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCallRequest:
    """A normalized model request to call a local tool."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ParsedAssistantMessage:
    """Assistant content plus any requested tool calls."""

    content: str | None
    tool_calls: list[ToolCallRequest]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def message_to_dict(message: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK message object into a plain serializable dict."""

    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return {key: value for key, value in message.items() if value is not None}
    raise TypeError(f"Unsupported message type: {type(message)!r}")


def parse_assistant_message(message: Any) -> ParsedAssistantMessage:
    """Extract assistant content and native function tool calls."""

    raw_message = message_to_dict(message)
    tool_calls: list[ToolCallRequest] = []

    for tool_call in raw_message.get("tool_calls", []) or []:
        function = tool_call.get("function", {})
        tool_calls.append(
            ToolCallRequest(
                id=tool_call["id"],
                name=function.get("name", ""),
                arguments=function.get("arguments", "{}"),
            )
        )

    return ParsedAssistantMessage(
        content=raw_message.get("content"),
        tool_calls=tool_calls,
    )
