"""Tool integrations."""

from minimax_agent.tools.tts import TTS
from minimax_agent.tools.asr import ASR
from minimax_agent.tools.vision import FaceRecognition

__all__ = ["TTS", "ASR", "FaceRecognition"]
