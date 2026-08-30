"""Integration tests for the agent loop, tool dispatch, and history."""

import json
from pathlib import Path
from typing import Any

import pytest

from agent.core import CodingAgent
from config.settings import Settings


class ScriptedLLM:
    """Fake LLM that returns pre-scripted assistant messages."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise RuntimeError("No scripted response available")
        return self.responses.pop(0)


class FailingLLM:
    """Fake LLM that simulates an API failure."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("simulated API outage")


def test_agent_runs_tool_loop_and_persists_history(
    fake_settings: Settings,
    temp_project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (temp_project / "notes.txt").write_text("hello integration", encoding="utf-8")
    agent = CodingAgent(settings=fake_settings, max_iterations=3)
    scripted_llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": "I will inspect the file.",
                "tool_calls": [
                    {
                        "id": "call_read_notes",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "notes.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The file contains hello integration.",
            },
        ]
    )
    agent.llm = scripted_llm  # type: ignore[assignment]

    answer = agent.run("Read notes.txt")

    captured = capsys.readouterr()
    history_lines = fake_settings.history_path.read_text(encoding="utf-8").splitlines()
    history_messages = [json.loads(line) for line in history_lines]
    tool_message = next(message for message in history_messages if message["role"] == "tool")
    tool_payload = json.loads(tool_message["content"])

    assert answer == "The file contains hello integration."
    assert "[Round 1/3]" in captured.out
    assert '[Tool] read_file({"path": "notes.txt"})' in captured.out
    assert "[Result] ok=True" in captured.out
    assert tool_payload["ok"] is True
    assert tool_payload["content"] == "hello integration"
    assert [message["role"] for message in history_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert scripted_llm.calls[1][-1]["role"] == "tool"


def test_agent_returns_friendly_message_on_api_failure(
    fake_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = CodingAgent(settings=fake_settings, max_iterations=2)
    agent.llm = FailingLLM()  # type: ignore[assignment]

    answer = agent.run("Do something")

    captured = capsys.readouterr()

    assert "could not complete the task" in answer
    assert "simulated API outage" in answer
    assert "[Error] API call failed: simulated API outage" in captured.out


def test_agent_warns_when_max_iterations_is_reached(
    fake_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = CodingAgent(settings=fake_settings, max_iterations=1)
    agent.llm = ScriptedLLM(
        [
            {
                "role": "assistant",
                "content": "I need to keep checking.",
                "tool_calls": [
                    {
                        "id": "call_missing",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "missing.txt"}',
                        },
                    }
                ],
            }
        ]
    )  # type: ignore[assignment]

    answer = agent.run("Read missing.txt")

    captured = capsys.readouterr()

    assert "maximum iteration limit was reached" in answer
    assert "[Warning] Stopped because the maximum iteration limit was reached (1)." in captured.out
