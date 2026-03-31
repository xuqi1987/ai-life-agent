"""ReAct 循环测试 — 使用 Mock 测试推理循环逻辑，无需真实 API。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_life_agent.core.react import ReActExecutor, ReActResult, ReActStep
from ai_life_agent.llm.client import LLMResponse
from ai_life_agent.tools.builtin import register_builtin_tools
from ai_life_agent.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> LLMResponse:
    """创建最终文本响应 Mock。"""
    return LLMResponse(
        text=text,
        stop_reason="end_turn",
        raw_content=[MagicMock(type="text", text=text)],
    )


def _make_tool_call_response(tool_name: str, tool_input: dict, tool_id: str = "t1") -> LLMResponse:
    """创建工具调用响应 Mock。"""
    raw_block = MagicMock()
    raw_block.type = "tool_use"
    raw_block.id = tool_id
    raw_block.name = tool_name
    raw_block.input = tool_input

    return LLMResponse(
        text="",
        tool_calls=[{"id": tool_id, "name": tool_name, "input": tool_input}],
        stop_reason="tool_use",
        raw_content=[raw_block],
    )


# ---------------------------------------------------------------------------
# ReActStep 测试
# ---------------------------------------------------------------------------


class TestReActStep:
    """测试 ReActStep 数据类。"""

    def test_is_final_with_answer(self):
        """有最终答案时 is_final() 返回 True。"""
        step = ReActStep(iteration=1, final_answer="最终答案")
        assert step.is_final() is True

    def test_is_final_without_answer(self):
        """无最终答案时 is_final() 返回 False。"""
        step = ReActStep(iteration=1)
        assert step.is_final() is False


# ---------------------------------------------------------------------------
# ReActResult 测试
# ---------------------------------------------------------------------------


class TestReActResult:
    """测试 ReActResult 数据类。"""

    def test_format_trace_basic(self):
        """format_trace 不抛出异常。"""
        result = ReActResult(
            answer="最终答案",
            steps=[
                ReActStep(
                    iteration=1,
                    thinking="我需要计算",
                    tool_calls=[{"name": "calculator", "input": {"expression": "2+2"}}],
                    observations=["4"],
                ),
                ReActStep(iteration=2, final_answer="2+2=4"),
            ],
            total_iterations=2,
        )
        trace = result.format_trace()
        assert "迭代 1" in trace
        assert "calculator" in trace
        assert "4" in trace

    def test_format_trace_with_limit(self):
        """达到迭代限制时，trace 中包含警告信息。"""
        result = ReActResult(
            answer="超时",
            steps=[],
            total_iterations=10,
            stopped_by_limit=True,
        )
        trace = result.format_trace()
        assert "最大迭代次数" in trace


# ---------------------------------------------------------------------------
# ReActExecutor 测试（Mock LLM）
# ---------------------------------------------------------------------------


class TestReActExecutor:
    """测试 ReAct 执行器的推理循环逻辑。"""

    def setup_method(self):
        """每个测试前创建注册了内置工具的注册表。"""
        self.registry = ToolRegistry()
        register_builtin_tools(self.registry)

    def test_direct_answer_no_tools(self):
        """LLM 直接给出答案，不调用工具。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _make_text_response("你好！我是 AI 助手。")

        executor = ReActExecutor(registry=self.registry, llm_client=mock_llm)
        result = executor.run("你好")

        assert result.answer == "你好！我是 AI 助手。"
        assert result.total_iterations == 1
        assert not result.stopped_by_limit
        mock_llm.chat.assert_called_once()

    def test_single_tool_call(self):
        """一次工具调用后给出最终答案。"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            # 第一次调用：工具调用
            _make_tool_call_response(
                "calculator",
                {"expression": "2+3*4"},
                tool_id="t1",
            ),
            # 第二次调用：最终答案
            _make_text_response("2+3*4 = 14"),
        ]

        executor = ReActExecutor(registry=self.registry, llm_client=mock_llm)
        result = executor.run("计算 2+3*4")

        assert result.answer == "2+3*4 = 14"
        assert result.total_iterations == 2
        assert mock_llm.chat.call_count == 2

        # 验证工具被正确执行（观测结果应包含 14）
        step1 = result.steps[0]
        assert len(step1.tool_calls) == 1
        assert step1.tool_calls[0]["name"] == "calculator"
        assert "14" in step1.observations[0]

    def test_multiple_tool_calls(self):
        """多次工具调用后给出最终答案。"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            _make_tool_call_response("calculator", {"expression": "sqrt(16)"}, "t1"),
            _make_tool_call_response("get_current_time", {}, "t2"),
            _make_text_response("sqrt(16)=4，当前时间已获取。"),
        ]

        executor = ReActExecutor(registry=self.registry, llm_client=mock_llm)
        result = executor.run("sqrt(16) 是多少？现在几点？")

        assert result.answer == "sqrt(16)=4，当前时间已获取。"
        assert result.total_iterations == 3

    def test_max_iterations_limit(self):
        """超出最大迭代次数时强制停止。"""
        mock_llm = MagicMock()
        # 始终返回工具调用，不给最终答案
        mock_llm.chat.return_value = _make_tool_call_response("echo", {"text": "loop"})

        executor = ReActExecutor(
            registry=self.registry,
            llm_client=mock_llm,
            max_iterations=3,
        )
        result = executor.run("无限循环测试")

        assert result.stopped_by_limit is True
        assert result.total_iterations <= 3

    def test_unknown_tool_returns_error_observation(self):
        """调用未注册工具时，观测结果为错误信息，不崩溃。"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            _make_tool_call_response("nonexistent_tool", {"param": "value"}, "t1"),
            _make_text_response("工具调用失败了"),
        ]

        executor = ReActExecutor(registry=self.registry, llm_client=mock_llm)
        result = executor.run("调用不存在的工具")

        # 不应崩溃，应该继续并给出最终答案
        assert result.answer == "工具调用失败了"
        step1 = result.steps[0]
        assert any("失败" in obs or "未找到" in obs for obs in step1.observations)

    def test_tool_call_result_fed_back_to_llm(self):
        """工具结果正确回传给 LLM（第二次调用包含工具结果）。"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            _make_tool_call_response("echo", {"text": "回声测试"}, "t1"),
            _make_text_response("收到: 回声测试"),
        ]

        executor = ReActExecutor(registry=self.registry, llm_client=mock_llm)
        executor.run("测试 echo")

        # 第二次调用的 messages 应包含工具结果
        second_call_messages = mock_llm.chat.call_args_list[1][1]["messages"]
        # 最后一条消息应是工具结果（tool_result 类型）
        last_msg = second_call_messages[-1]
        assert last_msg["role"] == "user"
        content = last_msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "tool_result"
        assert content[0]["content"] == "回声测试"

    def test_conversation_history_passed_to_llm(self):
        """历史对话正确传递给 LLM。"""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _make_text_response("我记得你说过你喜欢 Python。")

        executor = ReActExecutor(registry=self.registry, llm_client=mock_llm)

        history = [
            {"role": "user", "content": "我喜欢 Python"},
            {"role": "assistant", "content": "太好了！"},
        ]
        executor.run("你还记得我说什么吗？", conversation_history=history)

        first_call_messages = mock_llm.chat.call_args[1]["messages"]
        # 历史消息 + 当前用户消息
        assert len(first_call_messages) == 3
        assert first_call_messages[0]["content"] == "我喜欢 Python"
        assert first_call_messages[-1]["content"] == "你还记得我说什么吗？"

    def test_thinking_preserved_in_steps(self):
        """思维链内容被保存在步骤记录中。"""
        mock_llm = MagicMock()
        response = _make_text_response("42 是生命的意义。")
        response.thinking = "让我思考这个深刻的问题..."
        mock_llm.chat.return_value = response

        executor = ReActExecutor(registry=self.registry, llm_client=mock_llm)
        result = executor.run("生命的意义是什么？")

        assert result.steps[0].thinking == "让我思考这个深刻的问题..."


# ---------------------------------------------------------------------------
# Agent 集成测试（Mock LLM）
# ---------------------------------------------------------------------------


class TestAgentWithMockLLM:
    """测试 Agent 类与 Mock LLM 的集成。"""

    @patch("ai_life_agent.core.agent.settings")
    @patch("ai_life_agent.llm.client.anthropic.Anthropic")
    def test_agent_run_with_api_key(self, mock_anthropic_cls, mock_settings):
        """配置 API Key 时 Agent 正常运行。"""
        # 配置 Mock Settings
        mock_settings.minimax_api_key = "test-key"
        mock_settings.chat_model = "MiniMax-M2.7"
        mock_settings.max_tokens = 4096
        mock_settings.max_react_iterations = 10
        mock_settings.verbose = False
        mock_settings.minimax_anthropic_base_url = "https://api.minimaxi.com/anthropic"

        # 配置 Mock Anthropic
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "你好！"

        mock_msg = MagicMock()
        mock_msg.content = [text_block]
        mock_msg.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_msg

        from ai_life_agent.core.agent import Agent

        agent = Agent()
        response = agent.run("你好")

        assert "你好" in response
        assert agent.history_length == 2

    def test_agent_run_without_api_key(self):
        """未配置 API Key 时 Agent 降级运行。"""
        import os

        # 临时清除环境变量
        original = os.environ.get("MINIMAX_API_KEY")
        os.environ["MINIMAX_API_KEY"] = ""

        try:
            from ai_life_agent.core.agent import Agent

            # 用空 key 的 settings 创建 agent（注意：需要重新创建 agent 以清除缓存）
            agent = Agent.__new__(Agent)
            agent.model = "MiniMax-M2.7"
            agent.system_prompt = None
            agent.enable_tools = True
            agent.messages = []
            agent._react_executor = None
            agent._initialized = False

            # 注入空 api key 的配置
            with patch("ai_life_agent.core.agent.settings") as mock_cfg:
                mock_cfg.minimax_api_key = ""
                mock_cfg.chat_model = "MiniMax-M2.7"
                mock_cfg.verbose = False

                response = agent.run("测试")

            assert "降级模式" in response or "MINIMAX_API_KEY" in response
        finally:
            if original is not None:
                os.environ["MINIMAX_API_KEY"] = original
            elif "MINIMAX_API_KEY" in os.environ:
                del os.environ["MINIMAX_API_KEY"]
