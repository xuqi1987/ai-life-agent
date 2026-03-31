"""应用配置管理 — 使用 pydantic-settings 从环境变量或 .env 文件加载配置。

配置优先级（高 → 低）：
1. 系统环境变量
2. .env 文件
3. 代码中的默认值

使用方式::

    from ai_life_agent.config import settings

    print(settings.minimax_api_key)
    print(settings.chat_model)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置。

    所有字段均可通过同名大写环境变量覆盖。
    例如: MINIMAX_API_KEY=xxx uv run ai-life-agent
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- MiniMax API 配置 ----
    minimax_api_key: str = ""
    """MiniMax API 密钥，从 https://platform.minimaxi.com/user-center/basic-information/interface-key 获取。"""

    minimax_anthropic_base_url: str = "https://api.minimaxi.com/anthropic"
    """MiniMax Anthropic 兼容接口地址，用于文本对话。"""

    minimax_tts_url: str = "https://api.minimaxi.com/v1/t2a_v2"
    """MiniMax TTS（文字转语音）API 地址。"""

    # ---- 模型配置 ----
    chat_model: str = "MiniMax-M2.7"
    """对话模型名称。可选: MiniMax-M2.7, MiniMax-M2.7-highspeed, MiniMax-M2.5"""

    tts_model: str = "speech-2.8-hd"
    """TTS 模型名称。可选: speech-2.8-hd, speech-2.8-turbo, speech-02-hd, speech-02-turbo"""

    # ---- Agent 运行参数 ----
    max_tokens: int = 4096
    """LLM 单次最大生成 token 数。"""

    max_react_iterations: int = 10
    """ReAct 循环最大迭代次数，防止工具调用无限循环。"""

    verbose: bool = False
    """是否输出 Agent 的详细运行日志（思维链、工具调用等）。"""


# 全局单例配置对象
settings = Settings()
