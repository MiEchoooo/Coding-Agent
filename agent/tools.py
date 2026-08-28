"""Tool declarations and local execution dispatch."""

import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

ToolArguments: TypeAlias = dict[str, Any]
ToolResult: TypeAlias = dict[str, Any]
ToolFunction: TypeAlias = Callable[..., ToolResult]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_READ_NAMES = {".env", ".git"}
SENSITIVE_WRITE_NAMES = {".env", ".env.example", ".git"}
SKIPPED_SEARCH_NAMES = {".env", ".git", "__pycache__"}
BLOCKED_COMMANDS = {"sudo", "format", "fdisk", "mkfs"}
BLOCKED_CHMOD_MODES = {"777", "775", "666", "a+rwx", "ugo+rwx"}


def _tool_result(ok: bool, content: str = "", error: str | None = None) -> ToolResult:
    """Build the uniform tool result shape expected by the agent."""

    return {"ok": ok, "content": content, "error": error}


def _is_inside_project(path: Path) -> bool:
    """Return whether path stays within the project root."""

    try:
        path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _resolve_project_path(path: str) -> tuple[Path | None, str | None]:
    """Resolve a user path and reject paths outside the project root."""

    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"Invalid path: {exc}"

    if not _is_inside_project(resolved):
        return None, f"Path is outside the project root: {path}"

    return resolved, None


def _has_sensitive_part(path: Path, sensitive_names: set[str]) -> bool:
    """Check whether any path component targets a blocked sensitive location."""

    return any(part.lower() in sensitive_names for part in path.parts)


def _parse_command(command: str) -> tuple[list[str] | None, str | None]:
    """Split a shell-like command string for subprocess.run(shell=False)."""

    try:
        tokens = shlex.split(command, posix=os.name != "nt")
    except ValueError as exc:
        return None, f"Invalid command syntax: {exc}"

    if not tokens:
        return None, "Command cannot be empty"

    return tokens, None


def _is_blocked_command(tokens: list[str]) -> str | None:
    """Reject obviously destructive local commands."""

    lowered = [token.lower() for token in tokens]
    executable_path = Path(lowered[0])
    executable = executable_path.name
    executable_stem = executable_path.stem

    if executable in BLOCKED_COMMANDS or executable_stem in BLOCKED_COMMANDS:
        return f"Command is blocked for safety: {tokens[0]}"

    if executable_stem == "rm" and "-rf" in lowered and "/" in lowered:
        return "Command is blocked for safety: rm -rf /"

    if executable_stem == "chmod":
        if any(token in BLOCKED_CHMOD_MODES for token in lowered[1:]):
            return "Command is blocked for safety: high-risk chmod mode"
        if "-r" in lowered or "--recursive" in lowered:
            return "Command is blocked for safety: recursive chmod"

    if any(token.startswith("mkfs.") for token in lowered):
        return "Command is blocked for safety: mkfs"

    return None


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


def read_file(path: str, offset: int = 0, limit: int | None = None) -> ToolResult:
    """Read a UTF-8 text file within the project root."""

    resolved, error = _resolve_project_path(path)
    if error or resolved is None:
        return _tool_result(False, error=error)

    if _has_sensitive_part(resolved, SENSITIVE_READ_NAMES):
        return _tool_result(False, error="Reading .env or .git paths is not allowed")

    if offset < 0:
        return _tool_result(False, error="offset must be greater than or equal to 0")
    if limit is not None and limit < 0:
        return _tool_result(False, error="limit must be greater than or equal to 0")

    try:
        if not resolved.exists():
            return _tool_result(False, error=f"File does not exist: {path}")
        if not resolved.is_file():
            return _tool_result(False, error=f"Path is not a file: {path}")

        content = resolved.read_text(encoding="utf-8")
        end = None if limit is None else offset + limit
        return _tool_result(True, content=content[offset:end])
    except UnicodeDecodeError as exc:
        return _tool_result(False, error=f"File is not valid UTF-8 text: {exc}")
    except PermissionError as exc:
        return _tool_result(False, error=f"Permission denied: {exc}")
    except OSError as exc:
        return _tool_result(False, error=f"Failed to read file: {exc}")


def write_file(path: str, content: str) -> ToolResult:
    """Write UTF-8 text to a file within the project root."""

    resolved, error = _resolve_project_path(path)
    if error or resolved is None:
        return _tool_result(False, error=error)

    if _has_sensitive_part(resolved, SENSITIVE_WRITE_NAMES):
        return _tool_result(
            False,
            error="Writing .env, .env.example, or .git paths is not allowed",
        )

    try:
        if resolved.exists() and not resolved.is_file():
            return _tool_result(False, error=f"Path is not a file: {path}")

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return _tool_result(
            True,
            content=f"Wrote {len(content.encode('utf-8'))} bytes to {resolved}",
        )
    except PermissionError as exc:
        return _tool_result(False, error=f"Permission denied: {exc}")
    except OSError as exc:
        return _tool_result(False, error=f"Failed to write file: {exc}")


def run_shell(command: str, timeout_seconds: int = 30) -> ToolResult:
    """Run a local command with shell=False and project-root cwd."""

    if timeout_seconds <= 0:
        return _tool_result(False, error="timeout_seconds must be greater than 0")

    tokens, parse_error = _parse_command(command)
    if parse_error or tokens is None:
        return _tool_result(False, error=parse_error)

    blocked_error = _is_blocked_command(tokens)
    if blocked_error:
        return _tool_result(False, error=blocked_error)

    try:
        completed = subprocess.run(
            tokens,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
            check=False,
        )
    except FileNotFoundError as exc:
        return _tool_result(False, error=f"Command not found: {exc}")
    except subprocess.TimeoutExpired as exc:
        return _tool_result(
            False,
            content=(
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}\n"
                "return_code: timeout"
            ),
            error=f"Command timed out after {timeout_seconds} seconds",
        )
    except PermissionError as exc:
        return _tool_result(False, error=f"Permission denied: {exc}")
    except OSError as exc:
        return _tool_result(False, error=f"Failed to run command: {exc}")

    output = (
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
        f"return_code: {completed.returncode}"
    )
    return _tool_result(completed.returncode == 0, content=output)


def list_directory(path: str = ".") -> ToolResult:
    """List files and directories within the project root."""

    resolved, error = _resolve_project_path(path)
    if error or resolved is None:
        return _tool_result(False, error=error)

    try:
        if not resolved.exists():
            return _tool_result(False, error=f"Directory does not exist: {path}")
        if not resolved.is_dir():
            return _tool_result(False, error=f"Path is not a directory: {path}")

        entries = []
        for child in sorted(resolved.iterdir(), key=lambda item: item.name.lower()):
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
        return _tool_result(True, content="\n".join(entries))
    except PermissionError as exc:
        return _tool_result(False, error=f"Permission denied: {exc}")
    except OSError as exc:
        return _tool_result(False, error=f"Failed to list directory: {exc}")


def search_files(pattern: str, path: str = ".") -> ToolResult:
    """Search UTF-8 text files inside the project root."""

    resolved, error = _resolve_project_path(path)
    if error or resolved is None:
        return _tool_result(False, error=error)

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return _tool_result(False, error=f"Invalid search pattern: {exc}")

    try:
        if not resolved.exists():
            return _tool_result(False, error=f"Search path does not exist: {path}")

        files = [resolved] if resolved.is_file() else resolved.rglob("*")
        matches: list[str] = []

        for candidate in files:
            if not candidate.is_file():
                continue
            if _has_sensitive_part(candidate, SKIPPED_SEARCH_NAMES):
                continue

            try:
                relative = candidate.relative_to(PROJECT_ROOT)
                for line_number, line in enumerate(
                    candidate.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    if regex.search(line):
                        matches.append(f"{relative}:{line_number}: {line}")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

        return _tool_result(True, content="\n".join(matches))
    except PermissionError as exc:
        return _tool_result(False, error=f"Permission denied: {exc}")
    except OSError as exc:
        return _tool_result(False, error=f"Failed to search files: {exc}")


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
                "offset": {
                    "type": "integer",
                    "description": "Optional character offset to start reading from.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum number of characters to return.",
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
    "list_directory": ToolDefinition(
        name="list_directory",
        description="List files and directories in a local workspace directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Local directory path to list.",
                    "default": ".",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        function=list_directory,
    ),
    "search_files": ToolDefinition(
        name="search_files",
        description="Search UTF-8 text files in the local workspace.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Local file or directory path to search.",
                    "default": ".",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        function=search_files,
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
