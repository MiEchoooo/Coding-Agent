"""Agent main loop: think, call tools, observe, continue."""

import json
from pathlib import Path
from typing import Any

from agent.history import ConversationHistory
from agent.llm_client import LLMClient
from agent.parser import message_to_dict, parse_assistant_message
from agent.tools import execute_tool, get_tool_schemas
from config.settings import Settings

SYSTEM_PROMPT = """You are a programming assistant running inside a local project.
You have five available tools: read_file, write_file, run_shell, list_directory, and search_files.
First analyze the user's task, then decide which tools are necessary before acting.
Tool results are real observations from the local runtime. Do not fabricate file contents, command output, or tool results.
The working directory is the project root. Use relative paths for all file and directory arguments.
When the task is complete, provide a clear final answer that explains what was done or why progress stopped."""


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

        try:
            messages: list[dict[str, Any]] = self._initial_messages(task)
            final_answer: str = ""

            for round_index in range(1, self.max_iterations + 1):
                print(f"[Round {round_index}/{self.max_iterations}]")

                try:
                    assistant_message = self.llm.chat(messages, tools=get_tool_schemas())
                except Exception as exc:
                    error_message: str = f"API call failed: {exc}"
                    print(f"[Error] {error_message}")
                    return (
                        "The agent could not complete the task because the model API "
                        f"call failed. Details: {exc}"
                    )

                assistant_dict = message_to_dict(assistant_message)
                parsed = parse_assistant_message(assistant_message)

                if parsed.content:
                    print(f"Assistant: {self._truncate(parsed.content, 200)}")

                messages.append(assistant_dict)
                self.history.append(assistant_dict)

                if not parsed.has_tool_calls:
                    final_answer = parsed.content or ""
                    break

                for tool_call in parsed.tool_calls:
                    print(f"[Tool] {tool_call.name}({tool_call.arguments})")
                    result: dict[str, Any] = execute_tool(
                        tool_call.name,
                        tool_call.arguments,
                    )
                    result_content: str = str(result.get("content", ""))
                    print(
                        "[Result] "
                        f"ok={result.get('ok')}, "
                        f"content={self._truncate(result_content, 300)}"
                    )
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
                print(f"[Warning] {final_answer}")

            return final_answer
        except Exception as exc:
            print(f"[Error] Unexpected agent failure: {exc}")
            return (
                "The agent stopped because an unexpected runtime error occurred. "
                f"Details: {exc}"
            )

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

    def _truncate(self, content: str, max_length: int) -> str:
        """Trim long runtime output for readable console progress."""

        if len(content) <= max_length:
            return content
        return f"{content[:max_length]}..."
