"""Tests for ai_life_agent core."""

from ai_life_agent.core.agent import Agent
from ai_life_agent.core.executor import Executor
from ai_life_agent.memory import Memory


def test_agent_add_message():
    """Test adding messages to agent."""
    agent = Agent()
    agent.add_message("user", "Hello")
    assert len(agent.messages) == 1
    assert agent.messages[0].role == "user"
    assert agent.messages[0].content == "Hello"


def test_agent_run():
    """Test agent run loop — 验证 Agent 能处理输入并产生回复（含 LLM 调用）。"""
    agent = Agent()
    response = agent.run("Hello")
    assert len(agent.messages) == 2  # user + assistant
    assert isinstance(response, str)
    assert len(response) > 0  # 有实质性回复（LLM 模式或降级模式均有输出）


def test_memory_add_fact():
    """Test memory fact storage."""
    memory = Memory()
    memory.add_fact("name", "Alice")
    assert memory.get_facts() == {"name": "Alice"}


def test_memory_conversation_history():
    """Test conversation history."""
    memory = Memory()
    memory.add_turn("user", "Hello")
    memory.add_turn("assistant", "Hi there")
    history = memory.get_conversation_history()
    assert len(history) == 2


def test_executor_register_and_run():
    """Test tool registration and execution."""
    executor = Executor()

    def dummy_tool(a: int, b: int) -> int:
        return a + b

    executor.register("add", dummy_tool)
    result = executor.execute("add", {"a": 1, "b": 2})
    assert result == 3


def test_executor_unknown_tool():
    """Test unknown tool returns error."""
    executor = Executor()
    result = executor.execute("unknown", {})
    assert "error" in result
