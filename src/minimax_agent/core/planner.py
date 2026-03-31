"""Task planner — uses LLM to break down and plan tasks."""

from typing import Any


class Planner:
    """Plans tasks using the LLM."""

    def __init__(self, model: str = "MiniMax-M2.7"):
        self.model = model

    def plan(self, user_input: str, context: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Break down user input into executable steps."""
        # TODO: implement with MiniMax API
        return [{"action": "respond", "content": f"Planned: {user_input}"}]
