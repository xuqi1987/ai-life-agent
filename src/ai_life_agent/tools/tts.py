"""文字转语音工具 — 使用 MiniMax TTS API（同步模式）。

支持多种音色，输出 mp3/wav 格式音频。

API 文档: https://platform.minimaxi.com/docs/api-reference/speech-t2a-http

使用示例::

    tts = TTS()
    audio_bytes = tts.speak("你好，我是 AI 助手！", voice_id="male-qn-qingse")
    Path("output.mp3").write_bytes(audio_bytes)

    # 或者直接保存到文件
    tts.speak_to_file("你好！", "output.mp3")
"""

from __future__ import annotations

import binascii
import logging
from pathlib import Path

import requests

from ai_life_agent.config import settings

logger = logging.getLogger(__name__)

# MiniMax 内置音色列表（部分）
# 完整列表见 https://platform.minimaxi.com/docs/api-reference/speech-t2a-http
BUILTIN_VOICES = {
    "male-qn-qingse": "青涩青年男声",
    "male-qn-jingying": "精英青年男声",
    "female-shaonv": "少女音",
    "female-yujie": "御姐音",
    "female-tianmei": "甜美女声",
    "presenter_male": "男性播音员",
    "presenter_female": "女性播音员",
}


class TTS:
    """MiniMax 文字转语音客户端。

    通过 MiniMax TTS API 将文本转换为语音，返回音频字节流。

    Attributes:
        api_key: MiniMax API 密钥。
        model: TTS 模型名称，默认 speech-02-hd。
        tts_url: TTS API 端点。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """初始化 TTS 客户端。

        Args:
            api_key: MiniMax API 密钥，默认从 MINIMAX_API_KEY 环境变量读取。
            model: TTS 模型，默认使用配置中的 tts_model。
        """
        self.api_key = api_key or settings.minimax_api_key
        self.model = model or settings.tts_model
        self.tts_url = settings.minimax_tts_url

    @property
    def is_configured(self) -> bool:
        """API Key 是否已配置。"""
        return bool(self.api_key)

    def speak(
        self,
        text: str,
        voice_id: str = "male-qn-qingse",
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: int = 0,
    ) -> bytes:
        """将文本转换为语音，返回音频字节流（mp3 格式）。

        Args:
            text: 需要合成的文本，长度不超过 10000 字符。
            voice_id: 音色 ID，默认 "male-qn-qingse"（标准男声）。
                      可用音色见 BUILTIN_VOICES 字典。
            speed: 语速倍率，范围 [0.5, 2.0]，1.0 为正常速度。
            volume: 音量，范围 [0.1, 10.0]，1.0 为正常音量。
            pitch: 音调偏移，范围 [-12, 12]，0 为不变。

        Returns:
            MP3 格式音频字节流。

        Raises:
            RuntimeError: API Key 未配置或 API 调用失败。
        """
        if not self.is_configured:
            raise RuntimeError(
                "TTS 未配置：请设置 MINIMAX_API_KEY 环境变量。\n"
                "参考: https://platform.minimaxi.com/user-center/basic-information/interface-key"
            )

        payload = {
            "model": self.model,
            "text": text,
            "stream": False,
            "output_format": "hex",  # 返回 hex 编码，再解码为 bytes
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": volume,
                "pitch": pitch,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.debug("调用 TTS API: model=%s, voice=%s, text_len=%d", self.model, voice_id, len(text))

        response = requests.post(self.tts_url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        result = response.json()

        # 检查 API 状态码
        base_resp = result.get("base_resp", {})
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            status_msg = base_resp.get("status_msg", "未知错误")
            raise RuntimeError(f"TTS API 错误 [{status_code}]: {status_msg}")

        # 解码 hex 格式音频数据
        data = result.get("data", {})
        if not data:
            raise RuntimeError("TTS API 返回空数据")

        audio_hex = data.get("audio", "")
        if not audio_hex:
            raise RuntimeError("TTS API 返回音频数据为空")

        audio_bytes = binascii.unhexlify(audio_hex)
        logger.debug("TTS 合成成功: %d 字节", len(audio_bytes))
        return audio_bytes

    def speak_to_file(
        self,
        text: str,
        output_path: str,
        voice_id: str = "male-qn-qingse",
        **kwargs,
    ) -> Path:
        """将文本转换为语音并保存到文件。

        Args:
            text: 需要合成的文本。
            output_path: 输出文件路径（建议 .mp3 后缀）。
            voice_id: 音色 ID。
            **kwargs: 传递给 speak() 的其他参数（speed, volume, pitch）。

        Returns:
            保存的文件 Path 对象。
        """
        audio = self.speak(text, voice_id=voice_id, **kwargs)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        logger.info("TTS 音频已保存: %s", path)
        return path
