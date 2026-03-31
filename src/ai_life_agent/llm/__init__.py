"""LLM 接口层 — 封装 MiniMax API 调用。

提供 MiniMaxClient，通过 Anthropic 兼容接口与 MiniMax M2.7 对话。
"""

from .client import LLMResponse, MiniMaxClient

__all__ = ["MiniMaxClient", "LLMResponse"]
