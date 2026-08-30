"""Tests for agent.history module."""

import json
from pathlib import Path

from agent.history import ConversationHistory


class TestConversationHistoryInit:
    """Basic initialization behaviour."""

    def test_starts_empty(self, tmp_path: Path) -> None:
        """A newly created history with a fresh path has no messages."""
        history_path = tmp_path / "history.jsonl"
        history = ConversationHistory(history_path)

        assert history.messages == []
        assert history.path == history_path


class TestLoad:
    """Loading persisted history from disk."""

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Loading a missing file yields an empty list without error."""
        history = ConversationHistory(tmp_path / "missing.jsonl")
        loaded = history.load()

        assert loaded == []
        assert history.messages == []

    def test_load_existing_file(self, tmp_path: Path) -> None:
        """Messages are restored from a valid JSONL file."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text(
            '{"role": "system", "content": "sys"}\n'
            '{"role": "user", "content": "hello"}\n',
            encoding="utf-8",
        )

        history = ConversationHistory(history_path)
        loaded = history.load()

        assert len(loaded) == 2
        assert loaded[0] == {"role": "system", "content": "sys"}
        assert loaded[1] == {"role": "user", "content": "hello"}
        assert history.messages == loaded

    def test_load_ignores_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines in the JSONL file are skipped gracefully."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text(
            '{"role": "user", "content": "a"}\n\n\n'
            '{"role": "user", "content": "b"}\n',
            encoding="utf-8",
        )

        history = ConversationHistory(history_path)
        loaded = history.load()

        assert len(loaded) == 2
        assert loaded[0]["content"] == "a"
        assert loaded[1]["content"] == "b"

    def test_load_overwrites_memory(self, tmp_path: Path) -> None:
        """Calling load() replaces any in-memory messages."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text('{"role": "user", "content": "from_disk"}\n', encoding="utf-8")

        history = ConversationHistory(history_path)
        history.messages.append({"role": "assistant", "content": "stale"})
        history.load()

        assert len(history.messages) == 1
        assert history.messages[0]["content"] == "from_disk"


class TestAppend:
    """Appending single messages."""

    def test_append_updates_memory(self, tmp_path: Path) -> None:
        """The message is added to the in-memory list."""
        history = ConversationHistory(tmp_path / "history.jsonl")
        msg = {"role": "user", "content": "task"}

        history.append(msg)

        assert history.messages == [msg]

    def test_append_persists_to_disk(self, tmp_path: Path) -> None:
        """The message is written to the JSONL file."""
        history_path = tmp_path / "history.jsonl"
        history = ConversationHistory(history_path)
        msg = {"role": "assistant", "content": "done"}

        history.append(msg)

        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == msg

    def test_append_creates_parent_directories(self, tmp_path: Path) -> None:
        """Intermediate directories are created automatically."""
        nested_path = tmp_path / "logs" / "nested" / "history.jsonl"
        history = ConversationHistory(nested_path)
        history.append({"role": "user", "content": "x"})

        assert nested_path.exists()

    def test_append_unicode_content(self, tmp_path: Path) -> None:
        """Non-ASCII characters are preserved correctly."""
        history = ConversationHistory(tmp_path / "history.jsonl")
        msg = {"role": "user", "content": "你好，世界 🌍"}

        history.append(msg)

        loaded = history.load()
        assert loaded[0]["content"] == "你好，世界 🌍"

    def test_append_multiple_messages(self, tmp_path: Path) -> None:
        """Sequential appends produce multiple JSONL lines."""
        history = ConversationHistory(tmp_path / "history.jsonl")
        history.append({"role": "user", "content": "a"})
        history.append({"role": "assistant", "content": "b"})

        lines = tmp_path.joinpath("history.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "a"
        assert json.loads(lines[1])["content"] == "b"


class TestExtend:
    """Batch appending multiple messages."""

    def test_extend_updates_memory_and_disk(self, tmp_path: Path) -> None:
        """All messages are added in order."""
        history = ConversationHistory(tmp_path / "history.jsonl")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]

        history.extend(messages)

        assert len(history.messages) == 2
        lines = tmp_path.joinpath("history.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2


class TestSaveAll:
    """Rewriting the entire JSONL file from memory."""

    def test_save_all_overwrites_file(self, tmp_path: Path) -> None:
        """The file reflects only current in-memory messages."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text('{"role": "user", "content": "old"}\n', encoding="utf-8")

        history = ConversationHistory(history_path)
        history.load()
        history.messages = [{"role": "user", "content": "new"}]
        history.save_all()

        lines = history_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["content"] == "new"

    def test_save_all_creates_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if necessary."""
        nested_path = tmp_path / "deep" / "history.jsonl"
        history = ConversationHistory(nested_path)
        history.messages = [{"role": "user", "content": "x"}]
        history.save_all()

        assert nested_path.exists()


class TestClear:
    """Clearing in-memory and on-disk state."""

    def test_clear_removes_file(self, tmp_path: Path) -> None:
        """The history file is deleted."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text('{"role": "user", "content": "x"}\n', encoding="utf-8")

        history = ConversationHistory(history_path)
        history.clear()

        assert not history_path.exists()

    def test_clear_empties_memory(self, tmp_path: Path) -> None:
        """The in-memory message list is emptied."""
        history = ConversationHistory(tmp_path / "history.jsonl")
        history.append({"role": "user", "content": "x"})
        history.clear()

        assert history.messages == []

    def test_clear_idempotent(self, tmp_path: Path) -> None:
        """Clearing when the file is already absent does not raise."""
        history = ConversationHistory(tmp_path / "never_existed.jsonl")
        history.clear()  # should not raise


class TestJsonlFormat:
    """Structural guarantees of the persisted format."""

    def test_each_line_is_independent_json(self, tmp_path: Path) -> None:
        """Every line must be a valid, self-contained JSON object."""
        history = ConversationHistory(tmp_path / "history.jsonl")
        history.append({"role": "user", "content": "a"})
        history.append({"role": "assistant", "content": "b"})

        raw = tmp_path.joinpath("history.jsonl").read_text(encoding="utf-8")
        for line in raw.strip().splitlines():
            obj = json.loads(line)
            assert isinstance(obj, dict)
            assert "role" in obj

    def test_no_trailing_newline_issues(self, tmp_path: Path) -> None:
        """Blank trailing lines must not break subsequent loads."""
        history_path = tmp_path / "history.jsonl"
        history_path.write_text('{"role": "user", "content": "a"}\n\n', encoding="utf-8")

        history = ConversationHistory(history_path)
        loaded = history.load()

        assert len(loaded) == 1
