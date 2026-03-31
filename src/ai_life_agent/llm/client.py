"""MiniMax LLM 客户端 — 通过 Anthropic 兼容接口调用 MiniMax M2.7。

MiniMax 完全兼容 Anthropic SDK，因此我们使用 anthropic 库，
只需将 base_url 指向 MiniMax 的 Anthropic 兼容端点即可。

关键特性：
- 支持文本对话、工具调用（Function Calling）、思维链（Thinking）
- 多轮工具调用时自动保留完整 assistant 消息（含 thinking block），保持思维链连续性
- 通过 LLMResponse 统一封装响应

API 文档: https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from ai_life_agent.config import settings

logger = logging.getLogger(__name__)


class LLMResponse:
    """LLM 响应的标准化封装。

    Attributes:
        text: 模型生成的文本内容。
        thinking: 思维链内容（仅在模型开启 thinking 时有值）。
        tool_calls: 工具调用列表，每项为 {"id": ..., "name": ..., "input": {...}}。
        stop_reason: 停止原因，"end_turn" 表示正常结束，"tool_use" 表示需要执行工具。
        raw_content: Anthropic 原始 content 列表（多轮工具调用时需要完整回传）。
    """

    def __init__(
        self,
        text: str,
        thinking: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        stop_reason: str = "end_turn",
        raw_content: list[Any] | None = None,
    ):
        self.text = text
        self.thinking = thinking
        self.tool_calls = tool_calls or []
        self.stop_reason = stop_reason
        self.raw_content = raw_content or []

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用请求。"""
        return len(self.tool_calls) > 0

    @property
    def is_final(self) -> bool:
        """是否为最终答案（无需继续工具调用）。"""
        return self.stop_reason == "end_turn" and not self.has_tool_calls

    def __repr__(self) -> str:
        return (
            f"LLMResponse(stop_reason={self.stop_reason!r}, "
            f"tool_calls={len(self.tool_calls)}, "
            f"text={self.text[:50]!r})"
        )


class MiniMaxClient:
    """MiniMax LLM 客户端。

    使用 Anthropic SDK 格式调用 MiniMax M2.7 模型，支持工具调用和思维链。

    MiniMax 兼容端点:
        https://api.minimaxi.com/anthropic

    示例::

        client = MiniMaxClient()

        # 简单对话
        response = client.chat(
            messages=[{"role": "user", "content": "你好，请介绍一下自己"}]
        )
        print(response.text)

        # 带工具调用
        tools = [
            {
                "name": "calculator",
                "description": "计算数学表达式",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]
        response = client.chat(
            messages=[{"role": "user", "content": "计算 2+3*4"}],
            tools=tools,
        )
        if response.has_tool_calls:
            print("工具调用:", response.tool_calls)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        """初始化 MiniMax 客户端。

        Args:
            api_key: MiniMax API 密钥，默认从 MINIMAX_API_KEY 环境变量读取。
            model: 模型名称，默认使用配置中的 chat_model（MiniMax-M2.7）。
            max_tokens: 最大生成 token 数，默认使用配置中的 max_tokens（4096）。
        """
        self.api_key = api_key if api_key is not None else settings.minimax_api_key
        self.model = model or settings.chat_model
        self.max_tokens = max_tokens or settings.max_tokens

        self._client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=settings.minimax_anthropic_base_url,
        )

    @property
    def is_configured(self) -> bool:
        """API Key 是否已配置。"""
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """发送对话请求并获取响应。

        Args:
            messages: 对话消息列表，格式为::

                [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好！"},
                    ...
                ]

                注意：多轮工具调用时，assistant 消息的 content 必须是完整的
                raw_content 列表（包含 thinking/tool_use 等所有块），
                以保持思维链的连续性。

            system: 系统提示词（人设、行为约束等）。
            tools: Anthropic 格式的工具定义列表，例如::

                [{"name": "...", "description": "...", "input_schema": {...}}]

            temperature: 生成温度，范围 (0, 1]，默认 0.7。

        Returns:
            LLMResponse 对象，包含文本、思维链和工具调用。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": temperature,
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = tools

        logger.debug("调用 MiniMax API: model=%s, messages=%d条", self.model, len(messages))

        raw_response = self._client.messages.create(**kwargs)
        return self._parse_response(raw_response)

    def _parse_response(self, raw_response: anthropic.types.Message) -> LLMResponse:
        """将 Anthropic 原始响应解析为 LLMResponse。"""
        text = ""
        thinking = None
        tool_calls: list[dict[str, Any]] = []

        for block in raw_response.content:
            if block.type == "thinking":
                thinking = block.thinking
                logger.debug("思维链: %s...", (thinking or "")[:100])
            elif block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.input),
                    }
                )
                logger.debug("工具调用: %s(%s)", block.name, block.input)

        return LLMResponse(
            text=text,
            thinking=thinking,
            tool_calls=tool_calls,
            stop_reason=raw_response.stop_reason or "end_turn",
            raw_content=list(raw_response.content),
        )
