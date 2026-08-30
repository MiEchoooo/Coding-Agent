"""Tests for response parsing."""

from typing import Any

from agent.parser import parse_assistant_message


class FakeSdkMessage:
    """Small stand-in for SDK objects that expose model_dump()."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        if not exclude_none:
            return self.payload
        return {key: value for key, value in self.payload.items() if value is not None}


class TestParseAssistantMessage:
    def test_no_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": "I will help you.",
        }
        parsed = parse_assistant_message(msg)
        assert parsed.content == "I will help you."
        assert parsed.has_tool_calls is False

    def test_with_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.py"}',
                    },
                }
            ],
        }
        parsed = parse_assistant_message(msg)
        assert parsed.has_tool_calls is True
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "read_file"
        assert parsed.tool_calls[0].id == "call_123"
        assert parsed.tool_calls[0].arguments == '{"path": "test.py"}'

    def test_content_and_tools_together(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Let me check the file.",
            "tool_calls": [
                {
                    "id": "call_456",
                    "function": {
                        "name": "list_directory",
                        "arguments": '{"path": "."}',
                    },
                }
            ],
        }
        parsed = parse_assistant_message(msg)
        assert parsed.content == "Let me check the file."
        assert parsed.has_tool_calls is True

    def test_sdk_message_object_with_model_dump(self) -> None:
        msg = FakeSdkMessage(
            {
                "role": "assistant",
                "content": "Done",
                "tool_calls": None,
            }
        )

        parsed = parse_assistant_message(msg)

        assert parsed.content == "Done"
        assert parsed.tool_calls == []
