"""Text-to-Speech tool using MiniMax TTS API."""

import os
import requests
from pathlib import Path


class TTS:
    """MiniMax TTS integration."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.base_url = "https://api.minimax.chat/v1"

    def speak(self, text: str, voice: str = "male-qn-qingse") -> bytes:
        """Convert text to speech, return audio bytes."""
        # TODO: implement MiniMax TTS API call
        raise NotImplementedError("TTS not yet implemented")

    def speak_to_file(self, text: str, output_path: str, voice: str = "male-qn-qingse") -> Path:
        """Convert text to speech and save to file."""
        audio = self.speak(text, voice)
        path = Path(output_path)
        path.write_bytes(audio)
        return path
