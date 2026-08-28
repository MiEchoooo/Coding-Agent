"""Tool declarations and local execution dispatch.

The functions are placeholders by design. They define stable local tool entry
points without enabling real filesystem or shell side effects yet.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

ToolArguments: TypeAlias = dict[str, Any]
ToolResult: TypeAlias = dict[str, Any]
ToolFunction: TypeAlias = Callable[..., ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    """A callable local tool plus its OpenAI-compatible JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the native chat-completions tool schema."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def read_file(path: str) -> ToolResult:
    """Placeholder for a future local file read implementation."""

    return {
        "ok": False,
        "error": "read_file is not implemented yet",
        "path": path,
    }


def write_file(path: str, content: str) -> ToolResult:
    """Placeholder for a future local file write implementation."""

    return {
        "ok": False,
        "error": "write_file is not implemented yet",
        "path": path,
        "bytes": len(content.encode("utf-8")),
    }


def run_shell(command: str, timeout_seconds: int = 30) -> ToolResult:
    """Placeholder for a future local shell execution implementation."""

    return {
        "ok": False,
        "error": "run_shell is not implemented yet",
        "command": command,
        "timeout_seconds": timeout_seconds,
    }


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "read_file": ToolDefinition(
        name="read_file",
        description="Read a UTF-8 text file from the local workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Local path to read.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        function=read_file,
    ),
    "write_file": ToolDefinition(
        name="write_file",
        description="Write UTF-8 text content to a local workspace file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Local path to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        function=write_file,
    ),
    "run_shell": ToolDefinition(
        name="run_shell",
        description="Run a shell command locally in the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum runtime in seconds.",
                    "default": 30,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        function=run_shell,
    ),
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return all tool schemas in OpenAI-compatible format."""

    return [tool.to_openai_schema() for tool in TOOL_DEFINITIONS.values()]


def execute_tool(name: str, arguments_json: str) -> ToolResult:
    """Parse model tool arguments and execute the matching local tool."""

    tool = TOOL_DEFINITIONS.get(name)
    if tool is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid JSON arguments: {exc}"}

    if not isinstance(arguments, dict):
        return {"ok": False, "error": "Tool arguments must be a JSON object"}

    try:
        return tool.function(**arguments)
    except TypeError as exc:
        return {"ok": False, "error": f"Invalid tool arguments: {exc}"}
