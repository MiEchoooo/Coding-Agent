"""Agent main loop: think, call tools, observe, continue."""

import json
from pathlib import Path
from typing import Any

from agent.history import ConversationHistory
from agent.llm_client import LLMClient
from agent.parser import message_to_dict, parse_assistant_message
from agent.tools import execute_tool, get_tool_schemas
from config.settings import Settings

SYSTEM_PROMPT = """You are a simplified local coding agent.
Use tools when local workspace inspection or changes are needed.
When tools are unavailable or not implemented, explain the limitation clearly.
Return a final answer once the task is complete or cannot progress further."""


class CodingAgent:
    """A small self-contained coding agent skeleton."""

    def __init__(
        self,
        settings: Settings,
        history_path: Path | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)
        self.history = ConversationHistory(history_path or settings.history_path)
        self.max_iterations = max_iterations or settings.max_iterations

    def run(self, task: str) -> str:
        """Run the agent loop until a final answer or iteration limit is reached."""

        messages = self._initial_messages(task)
        final_answer = ""

        for _ in range(self.max_iterations):
            assistant_message = self.llm.chat(messages, tools=get_tool_schemas())
            assistant_dict = message_to_dict(assistant_message)
            parsed = parse_assistant_message(assistant_message)

            messages.append(assistant_dict)
            self.history.append(assistant_dict)

            if not parsed.has_tool_calls:
                final_answer = parsed.content or ""
                break

            for tool_call in parsed.tool_calls:
                result = execute_tool(tool_call.name, tool_call.arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                messages.append(tool_message)
                self.history.append(tool_message)
        else:
            final_answer = (
                "Stopped because the maximum iteration limit was reached "
                f"({self.max_iterations})."
            )

        return final_answer

    def _initial_messages(self, task: str) -> list[dict[str, Any]]:
        """Load prior context and append the new user task."""

        messages = self.history.load()
        if not messages:
            system_message = {"role": "system", "content": SYSTEM_PROMPT}
            messages.append(system_message)
            self.history.append(system_message)

        user_message = {"role": "user", "content": task}
        messages.append(user_message)
        self.history.append(user_message)
        return messages
