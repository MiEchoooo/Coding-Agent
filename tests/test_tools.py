"""Tests for agent tools and safety mechanisms."""

import os
from pathlib import Path

import pytest

from agent.tools import (
    execute_tool,
    get_tool_schemas,
    list_directory,
    read_file,
    run_shell,
    search_files,
    write_file,
)


@pytest.fixture
def temp_project(tmp_path: Path):
    """Create a temporary project directory for isolated testing."""
    import agent.tools as tools_module
    original_root = tools_module.PROJECT_ROOT
    tools_module.PROJECT_ROOT = tmp_path
    yield tmp_path
    tools_module.PROJECT_ROOT = original_root


class TestPathSafety:
    """Path sandbox and sensitive file protections."""

    def test_read_file_outside_project(self, temp_project: Path) -> None:
        result = read_file("../outside.txt")
        assert result["ok"] is False
        assert "outside" in result["error"].lower()

    def test_write_file_dotenv_blocked(self, temp_project: Path) -> None:
        result = write_file(".env", "secret=123")
        assert result["ok"] is False
        assert ".env" in result["error"]

    def test_write_file_git_blocked(self, temp_project: Path) -> None:
        result = write_file(".git/config", "malicious")
        assert result["ok"] is False

    def test_resolve_traversal_attack(self, temp_project: Path) -> None:
        result = read_file("foo/../../../etc/passwd")
        assert result["ok"] is False


class TestReadWrite:
    """Basic file operations."""

    def test_write_and_read_roundtrip(self, temp_project: Path) -> None:
        write_file("test.txt", "hello world")
        result = read_file("test.txt")
        assert result["ok"] is True
        assert result["content"] == "hello world"

    def test_read_file_with_offset_limit(self, temp_project: Path) -> None:
        write_file("test.txt", "abcdefghij")
        result = read_file("test.txt", offset=2, limit=3)
        assert result["content"] == "cde"

    def test_read_nonexistent_file(self, temp_project: Path) -> None:
        result = read_file("not_exist.txt")
        assert result["ok"] is False

    def test_write_file_creates_nested_dirs(self, temp_project: Path) -> None:
        result = write_file("deep/nested/file.txt", "deep content")
        assert result["ok"] is True
        assert (temp_project / "deep" / "nested" / "file.txt").exists()


class TestShellSafety:
    """Command execution restrictions."""

    def test_blocked_rm_rf(self, temp_project: Path) -> None:
        result = run_shell("rm -rf /")
        assert result["ok"] is False
        assert "blocked" in result["error"].lower()

    def test_blocked_format_drive(self, temp_project: Path) -> None:
        result = run_shell("format C:")
        assert result["ok"] is False
        assert "blocked" in result["error"].lower()

    def test_blocked_sudo(self, temp_project: Path) -> None:
        result = run_shell("sudo apt-get install something")
        assert result["ok"] is False

    def test_valid_command_python(self, temp_project: Path) -> None:
        """Windows-compatible: use python instead of echo."""
        result = run_shell("python --version")
        assert result["ok"] is True
        assert "Python" in result["content"]


class TestDirectoryAndSearch:
    """list_directory and search_files."""

    def test_list_directory(self, temp_project: Path) -> None:
        write_file("a.txt", "content")
        (temp_project / "subdir").mkdir()
        result = list_directory(".")
        assert result["ok"] is True
        assert "a.txt" in result["content"]
        assert "subdir/" in result["content"]

    def test_search_files(self, temp_project: Path) -> None:
        write_file("file1.py", "def hello(): pass")
        write_file("file2.py", "def world(): pass")
        result = search_files("def hello", ".")
        assert result["ok"] is True
        assert "file1.py" in result["content"]
        assert "file2.py" not in result["content"]


class TestExecuteToolDispatch:
    """The execute_tool dispatcher."""

    def test_unknown_tool(self) -> None:
        result = execute_tool("nonexistent", "{}")
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]

    def test_invalid_json(self) -> None:
        result = execute_tool("read_file", "not json")
        assert result["ok"] is False
        assert "Invalid JSON" in result["error"]

    def test_missing_required_arg(self) -> None:
        result = execute_tool("read_file", "{}")
        assert result["ok"] is False