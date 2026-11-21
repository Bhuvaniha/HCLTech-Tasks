"""Prompt definitions and message assembly."""

from typing import Any

SYSTEM_PROMPT: str = "You are a helpful, concise assistant."


def build_messages(
    history_messages: list[dict[str, Any]], user_input: str
) -> list[dict[str, Any]]:
    """Return the full message list with the system prompt first."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_input})
    return messages
