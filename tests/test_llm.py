"""LLM 客户端测试 — 使用 Mock 对象测试 MiniMaxClient，无需真实 API。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_life_agent.llm.client import LLMResponse, MiniMaxClient

# ---------------------------------------------------------------------------
# 辅助函数 — 构造 Anthropic 响应 Mock
# ---------------------------------------------------------------------------


def _make_mock_message(
    text: str = "测试回答",
    thinking: str | None = None,
    tool_calls: list[dict] | None = None,
    stop_reason: str = "end_turn",
):
    """构造模拟的 Anthropic Message 对象。"""
    content_blocks = []

    if thinking:
        tb = MagicMock()
        tb.type = "thinking"
        tb.thinking = thinking
        content_blocks.append(tb)

    if tool_calls:
        for tc in tool_calls:
            tb = MagicMock()
            tb.type = "tool_use"
            tb.id = tc.get("id", "toolu_test")
            tb.name = tc["name"]
            tb.input = tc["input"]
            content_blocks.append(tb)
    else:
        tb = MagicMock()
        tb.type = "text"
        tb.text = text
        content_blocks.append(tb)

    msg = MagicMock()
    msg.content = content_blocks
    msg.stop_reason = stop_reason
    return msg


# ---------------------------------------------------------------------------
# LLMResponse 测试
# ---------------------------------------------------------------------------


class TestLLMResponse:
    """测试 LLMResponse 数据类。"""

    def test_basic_response(self):
        """基本文本响应。"""
        resp = LLMResponse(text="你好")
        assert resp.text == "你好"
        assert not resp.has_tool_calls
        assert resp.is_final

    def test_response_with_tool_calls(self):
        """带工具调用的响应。"""
        resp = LLMResponse(
            text="",
            tool_calls=[{"id": "t1", "name": "calculator", "input": {"expression": "2+2"}}],
            stop_reason="tool_use",
        )
        assert resp.has_tool_calls
        assert not resp.is_final
        assert len(resp.tool_calls) == 1

    def test_response_with_thinking(self):
        """带思维链的响应。"""
        resp = LLMResponse(text="答案是 4", thinking="我需要计算 2+2=4")
        assert resp.thinking == "我需要计算 2+2=4"

    def test_repr(self):
        """__repr__ 不抛出异常。"""
        resp = LLMResponse(text="hello")
        assert "LLMResponse" in repr(resp)


# ---------------------------------------------------------------------------
# MiniMaxClient 测试（Mock Anthropic SDK）
# ---------------------------------------------------------------------------


class TestMiniMaxClient:
    """测试 MiniMaxClient，使用 Mock 替代真实 API 调用。"""

    @patch("ai_life_agent.llm.client.anthropic.Anthropic")
    def test_chat_basic(self, mock_anthropic_cls):
        """基本对话调用。"""
        # 配置 Mock
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(text="你好！我是 AI 助手。")

        client = MiniMaxClient(api_key="test-key")
        response = client.chat(messages=[{"role": "user", "content": "你好"}])

        assert response.text == "你好！我是 AI 助手。"
        assert not response.has_tool_calls
        mock_client.messages.create.assert_called_once()

    @patch("ai_life_agent.llm.client.anthropic.Anthropic")
    def test_chat_with_system(self, mock_anthropic_cls):
        """带系统提示词的对话。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(text="OK")

        client = MiniMaxClient(api_key="test-key")
        client.chat(
            messages=[{"role": "user", "content": "测试"}],
            system="你是助手",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "你是助手"

    @patch("ai_life_agent.llm.client.anthropic.Anthropic")
    def test_chat_with_tools(self, mock_anthropic_cls):
        """带工具列表的对话。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            tool_calls=[{"id": "t1", "name": "calculator", "input": {"expression": "2+2"}}],
            stop_reason="tool_use",
        )

        tools = [
            {
                "name": "calculator",
                "description": "计算",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]

        client = MiniMaxClient(api_key="test-key")
        response = client.chat(
            messages=[{"role": "user", "content": "计算 2+2"}],
            tools=tools,
        )

        assert response.has_tool_calls
        assert response.stop_reason == "tool_use"
        assert response.tool_calls[0]["name"] == "calculator"
        assert response.tool_calls[0]["input"]["expression"] == "2+2"

    @patch("ai_life_agent.llm.client.anthropic.Anthropic")
    def test_chat_with_thinking(self, mock_anthropic_cls):
        """带思维链的响应。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message(
            text="答案是 4",
            thinking="让我仔细思考：2+2=4",
        )

        client = MiniMaxClient(api_key="test-key")
        response = client.chat(
            messages=[{"role": "user", "content": "2+2=?"}],
        )

        assert response.thinking == "让我仔细思考：2+2=4"
        assert response.text == "答案是 4"

    @patch("ai_life_agent.llm.client.anthropic.Anthropic")
    def test_is_configured(self, mock_anthropic_cls):
        """is_configured 属性反映 API Key 状态。"""
        mock_anthropic_cls.return_value = MagicMock()

        client_with_key = MiniMaxClient(api_key="test-key")
        assert client_with_key.is_configured is True

        client_no_key = MiniMaxClient(api_key="")
        assert client_no_key.is_configured is False

    @patch("ai_life_agent.llm.client.anthropic.Anthropic")
    def test_raw_content_preserved(self, mock_anthropic_cls):
        """原始 content 列表被保留（多轮工具调用需要）。"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_msg = _make_mock_message(text="回答")
        mock_client.messages.create.return_value = mock_msg

        client = MiniMaxClient(api_key="test-key")
        response = client.chat(messages=[{"role": "user", "content": "问题"}])

        assert response.raw_content is not None
        assert len(response.raw_content) > 0
