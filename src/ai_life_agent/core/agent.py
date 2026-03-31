"""MiniMax Agent 核心模块 — 整合 LLM、工具系统和 ReAct 推理循环。

Agent 是整个系统的统一入口，负责：
1. 维护对话历史（Memory）
2. 管理工具注册表
3. 驱动 ReAct 推理循环
4. 优雅降级（无 API Key 时返回提示信息）

示例::

    from ai_life_agent.core.agent import Agent

    agent = Agent()
    response = agent.run("现在几点了？")
    print(response)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ai_life_agent.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """对话中的单条消息。

    Attributes:
        role: 消息角色，"user" | "assistant" | "system"。
        content: 消息内容文本。
        metadata: 附加元数据（工具调用信息、思维链摘要等）。
    """

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行结果。

    Attributes:
        tool: 工具名称。
        success: 是否执行成功。
        result: 执行结果（成功时）。
        error: 错误信息（失败时）。
    """

    tool: str
    success: bool
    result: Any = None
    error: str | None = None


class Agent:
    """主 Agent 类 — 统一入口，整合 LLM + 工具 + ReAct 循环。

    支持两种运行模式：
    1. **完整模式**（有 API Key）：使用 MiniMax M2.7 + ReAct 循环处理复杂任务
    2. **降级模式**（无 API Key）：返回提示信息，所有功能保持可测试

    示例::

        agent = Agent()
        print(agent.run("帮我计算 sqrt(144)"))
        print(agent.run("现在几点了？"))

        # 查看对话历史
        for msg in agent.messages:
            print(f"{msg.role}: {msg.content}")
    """

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        enable_tools: bool = True,
    ):
        """初始化 Agent。

        Args:
            model: LLM 模型名称，默认使用配置中的 chat_model。
            system_prompt: 系统提示词，若为 None 则使用 ReAct 默认提示词。
            enable_tools: 是否启用工具系统，默认 True。
        """
        self.model = model or settings.chat_model
        self.system_prompt = system_prompt
        self.enable_tools = enable_tools
        self.messages: list[Message] = []

        # 延迟初始化（仅在有 API Key 时创建）
        self._react_executor: Any = None
        self._initialized = False

    def _init_executor(self) -> None:
        """懒加载初始化 ReAct 执行器（首次调用时）。"""
        if self._initialized:
            return

        if not settings.minimax_api_key:
            logger.warning(
                "MINIMAX_API_KEY 未配置，Agent 将以降级模式运行。\n"
                "请设置环境变量 MINIMAX_API_KEY 或在 .env 文件中配置。"
            )
            self._initialized = True
            return

        try:
            from ai_life_agent.core.react import ReActExecutor
            from ai_life_agent.llm.client import MiniMaxClient
            from ai_life_agent.tools.builtin import register_builtin_tools
            from ai_life_agent.tools.registry import ToolRegistry

            # 创建工具注册表
            registry = ToolRegistry()
            if self.enable_tools:
                register_builtin_tools(registry)
                logger.info("已注册内置工具: %s", registry.list_tools())

            # 创建 LLM 客户端
            llm_client = MiniMaxClient(model=self.model)

            # 创建 ReAct 执行器
            self._react_executor = ReActExecutor(
                registry=registry,
                llm_client=llm_client,
                system_prompt=self.system_prompt,
            )

            logger.info("Agent 初始化成功: model=%s, tools=%d个", self.model, len(registry))

        except Exception as e:
            logger.error("Agent 初始化失败: %s", e)

        self._initialized = True

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """向对话历史中添加消息。

        Args:
            role: 消息角色（"user" / "assistant" / "system"）。
            content: 消息内容。
            **metadata: 附加元数据（如 thinking、tool_calls 等）。
        """
        self.messages.append(
            Message(role=role, content=content, metadata=dict(metadata))
        )

    def run(self, user_input: str) -> str:
        """处理用户输入并返回 Agent 回答。

        如果 API Key 已配置，使用 ReAct 循环智能处理；
        否则返回降级提示信息。

        Args:
            user_input: 用户输入文本。

        Returns:
            Agent 回答文本。
        """
        self._init_executor()
        self.add_message("user", user_input)

        if self._react_executor is None:
            # 降级模式：无 API Key
            response = (
                f"[Agent 降级模式] 收到: '{user_input}'\n"
                "请配置 MINIMAX_API_KEY 以启用完整 AI 功能。"
            )
        else:
            # 完整模式：使用 ReAct 循环
            # 将对话历史转换为 LLM 格式（只传文本消息，不传系统消息）
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in self.messages[:-1]  # 不包含刚添加的用户消息
                if msg.role in ("user", "assistant")
            ]

            result = self._react_executor.run(
                user_input=user_input,
                conversation_history=history,
            )
            response = result.answer

            if settings.verbose and result.steps:
                logger.info("推理链路:\n%s", result.format_trace())

        self.add_message("assistant", response)
        return response

    def clear_history(self) -> None:
        """清空对话历史。"""
        self.messages.clear()

    @property
    def history_length(self) -> int:
        """当前对话历史长度（消息条数）。"""
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"Agent(model={self.model!r}, "
            f"history={len(self.messages)}条消息, "
            f"tools={'启用' if self.enable_tools else '禁用'})"
        )
