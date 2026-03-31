"""Tool integrations — 工具集合导出。"""

from ai_life_agent.tools.asr import ASR
from ai_life_agent.tools.builtin import register_builtin_tools
from ai_life_agent.tools.registry import ToolParameter, ToolRegistry, default_registry
from ai_life_agent.tools.tts import TTS
from ai_life_agent.tools.vision import FaceRecognition

__all__ = [
    "ToolRegistry",
    "ToolParameter",
    "default_registry",
    "TTS",
    "ASR",
    "FaceRecognition",
    "register_builtin_tools",
]
