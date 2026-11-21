"""Conversation history storage with turn-based trimming."""

from typing import Any


class ChatHistory:
    """Keep a bounded list of user/assistant messages."""

    def __init__(self, max_turns: int = 10) -> None:
        """Store the maximum number of turns to retain."""
        self.max_turns = max_turns
        self._messages: list[dict[str, Any]] = []

    def add_user(self, content: str) -> None:
        """Append a user message and trim old turns."""
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        """Append an assistant message and trim old turns."""
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a copy of the stored messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def _trim(self) -> None:
        """Keep only the most recent ``max_turns`` turns."""
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
