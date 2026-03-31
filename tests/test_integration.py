"""Integration tests."""

import pytest
from ai_life_agent.core.agent import Agent
from ai_life_agent.memory import Memory


def test_agent_with_memory():
    """Test agent stores conversation in memory."""
    agent = Agent()
    memory = Memory()

    user_input = "My name is Alice"
    response = agent.run(user_input)

    memory.add_turn("user", user_input)
    memory.add_turn("assistant", response)

    history = memory.get_conversation_history()
    assert len(history) == 2
    assert history[0]["content"] == user_input
