"""Automatic Speech Recognition tool."""

import os


class ASR:
    """MiniMax ASR integration."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")

    def recognize(self, audio_bytes: bytes) -> str:
        """Convert speech audio to text."""
        # TODO: implement MiniMax ASR API call
        raise NotImplementedError("ASR not yet implemented")
