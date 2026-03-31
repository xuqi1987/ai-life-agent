"""MiniMax Agent core module."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool: str
    success: bool
    result: Any = None
    error: str | None = None


class Agent:
    """Main agent class — handles the agent loop."""

    def __init__(self, model: str = "MiniMax-M2.7"):
        self.model = model
        self.messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))

    def run(self, user_input: str) -> str:
        """Run the agent with a user input."""
        self.add_message("user", user_input)
        # TODO: call MiniMax M2.7 API
        response = f"[Agent: Received '{user_input}']"
        self.add_message("assistant", response)
        return response
