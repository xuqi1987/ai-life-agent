"""
╔══════════════════════════════════════════════════════════════╗
║  第 5 步：让 Agent 开口说话 —— TTS 文字转语音                  ║
╚══════════════════════════════════════════════════════════════╝

学习目标
--------
使用 MiniMax TTS API 将 Agent 的文字回答转换为语音。

TTS 是什么？
-----------
TTS = Text To Speech（文字转语音）
  MiniMax Speech-2.8-HD 模型可以将任意文字转为自然的语音，
  支持多种音色、语速、音调调节。

运行方法
--------
  uv run python examples/step5_tts.py

  运行后会在当前目录生成 .mp3 文件，用你的播放器打开即可收听。
"""

from pathlib import Path

from ai_life_agent.core.react import ReActExecutor
from ai_life_agent.tools.builtin import register_builtin_tools
from ai_life_agent.tools.registry import ToolRegistry
from ai_life_agent.tools.tts import BUILTIN_VOICES, TTS


def main():
    print("=" * 60)
    print("  第 5 步：文字转语音（TTS）")
    print("=" * 60)

    tts = TTS()

    if not tts.is_configured:
        print("\n❌ 未配置 MINIMAX_API_KEY，无法运行 TTS 示例")
        print("   请在 .env 文件中设置 MINIMAX_API_KEY=你的密钥")
        return

    print(f"\n使用模型: {tts.model}")
    print("\n可用音色:")
    for voice_id, desc in BUILTIN_VOICES.items():
        print(f"  {voice_id:<25} {desc}")

    # --------------------------------------------------------
    # 【演示 1】最简单的 TTS 调用
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("【演示 1】最简单的 TTS")
    text1 = "你好！我是 AI Life Agent，由 MiniMax M2.7 驱动。很高兴认识你！"
    print(f"文字: {text1}")

    output1 = "output_demo1.mp3"
    tts.speak_to_file(text1, output1)
    size = Path(output1).stat().st_size
    print(f"✅ 已生成: {output1}（{size // 1024} KB）")

    # --------------------------------------------------------
    # 【演示 2】不同音色对比
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("【演示 2】不同音色对比")
    text2 = "这是一段测试语音，用来对比不同音色的效果。"

    voices_to_test = ["male-qn-qingse", "female-shaonv", "presenter_male"]
    for voice in voices_to_test:
        output_path = f"output_voice_{voice}.mp3"
        try:
            tts.speak_to_file(text2, output_path, voice_id=voice)
            size = Path(output_path).stat().st_size
            desc = BUILTIN_VOICES.get(voice, "")
            print(f"  ✅ {voice:<25} ({desc}) → {output_path} ({size//1024} KB)")
        except Exception as e:
            print(f"  ❌ {voice}: {e}")

    # --------------------------------------------------------
    # 【演示 3】调节语速和音调
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("【演示 3】调节语速")
    text3 = "这句话用不同语速朗读，感受一下速度的差异。"

    for speed, label in [(0.7, "慢速"), (1.0, "正常"), (1.5, "快速")]:
        output_path = f"output_speed_{label}.mp3"
        tts.speak_to_file(text3, output_path, speed=speed)
        size = Path(output_path).stat().st_size
        print(f"  ✅ {label} (speed={speed}) → {output_path} ({size//1024} KB)")

    # --------------------------------------------------------
    # 【演示 4】Agent 回答自动转为语音
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("【演示 4】Agent 回答 → 自动转语音")

    # 先用 Agent 生成一个诗意的回答
    registry = ToolRegistry()
    register_builtin_tools(registry)
    executor = ReActExecutor(registry=registry)

    result = executor.run(
        "请用优美的语言写一段关于春天的描述，大约50字。"
    )
    answer_text = result.answer
    print(f"\nAgent 回答: {answer_text}")

    # 把回答转为语音
    tts.speak_to_file(answer_text, "output_agent_answer.mp3")
    print("\n✅ 语音文件: output_agent_answer.mp3")

    # --------------------------------------------------------
    # 小结
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TTS 使用总结")
    print("=" * 60)
    print("""
  基本用法：
    tts = TTS()
    audio_bytes = tts.speak("你好")          # 返回音频字节
    tts.speak_to_file("你好", "out.mp3")     # 直接保存文件

  可调参数：
    voice_id  = 音色（见上面列表）
    speed     = 语速（0.5~2.0，默认1.0）
    volume    = 音量（0.1~10.0，默认1.0）
    pitch     = 音调（-12~12，默认0）

  生成的文件: output_*.mp3（用你的音乐播放器打开）

  下一步：完整的多轮对话 Agent → step6_full_agent.py
""")


if __name__ == "__main__":
    main()
