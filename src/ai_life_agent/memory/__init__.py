"""Memory management."""

from typing import Any


class Memory:
    """Simple memory store for conversation history and facts."""

    def __init__(self):
        self.conversation: list[dict[str, Any]] = []
        self.facts: dict[str, Any] = {}

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn."""
        self.conversation.append({"role": role, "content": content})

    def add_fact(self, key: str, value: Any) -> None:
        """Store a fact."""
        self.facts[key] = value

    def get_facts(self) -> dict[str, Any]:
        """Retrieve all facts."""
        return self.facts.copy()

    def get_conversation_history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Get conversation history."""
        if last_n is None:
            return self.conversation.copy()
        return self.conversation[-last_n:]
