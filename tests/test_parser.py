"""Tests for response parsing."""

from agent.parser import parse_assistant_message, ToolCallRequest


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