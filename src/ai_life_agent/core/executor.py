"""Tool executor — runs tools and returns results."""

from collections.abc import Callable
from typing import Any


class Executor:
    """Executes tools based on plan."""

    def __init__(self):
        self.tools: dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a tool."""
        self.tools[name] = func

    def execute(self, action: str, params: dict[str, Any]) -> Any:
        """Execute a single action."""
        if action not in self.tools:
            return {"error": f"Unknown tool: {action}"}
        return self.tools[action](**params)
