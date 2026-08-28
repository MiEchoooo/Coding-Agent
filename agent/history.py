"""JSONL-backed conversation history management."""

import json
from pathlib import Path
from typing import Any, TypeAlias

ChatMessage: TypeAlias = dict[str, Any]


class ConversationHistory:
    """Stores chat messages and persists them as one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.messages: list[ChatMessage] = []

    def load(self) -> list[ChatMessage]:
        """Load messages from disk if the history file exists."""

        if not self.path.exists():
            self.messages = []
            return self.messages

        loaded: list[ChatMessage] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if stripped:
                    loaded.append(json.loads(stripped))

        self.messages = loaded
        return self.messages

    def append(self, message: ChatMessage) -> None:
        """Append a message to memory and persist it to disk."""

        self.messages.append(message)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    def extend(self, messages: list[ChatMessage]) -> None:
        """Append multiple messages in order."""

        for message in messages:
            self.append(message)

    def save_all(self) -> None:
        """Rewrite the JSONL file with the current in-memory messages."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for message in self.messages:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        """Clear in-memory and on-disk history."""

        self.messages = []
        if self.path.exists():
            self.path.unlink()
