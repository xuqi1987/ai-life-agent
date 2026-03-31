"""
╔══════════════════════════════════════════════════════════════╗
║           AI Life Agent —— 一键体验演示                       ║
╚══════════════════════════════════════════════════════════════╝

运行方法：
  uv run python demo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEMOS = [
    {
        "id": "1",
        "title": "第 1 步：你好，大模型！",
        "desc": "最简单的 LLM 对话（10行代码 → AI 回答）",
        "file": "examples/step1_hello_llm.py",
    },
    {
        "id": "2",
        "title": "第 2 步：工具调用",
        "desc": "让 AI 使用「天气查询」和「BMI 计算」工具",
        "file": "examples/step2_tools.py",
    },
    {
        "id": "3",
        "title": "第 3 步：ReAct 推理循环",
        "desc": "AI 自动多步推理（Thought → Action → Observation）",
        "file": "examples/step3_react.py",
    },
    {
        "id": "4",
        "title": "第 4 步：自定义工具",
        "desc": "自己写工具（励志名言、待办事项、单位换算）",
        "file": "examples/step4_custom_tool.py",
    },
    {
        "id": "5",
        "title": "第 5 步：语音合成（TTS）",
        "desc": "将 AI 回答转为语音文件（生成 .mp3）",
        "file": "examples/step5_tts.py",
    },
    {
        "id": "6",
        "title": "第 6 步：完整多轮 Agent",
        "desc": "带记忆的完整 Agent（多轮对话 + 笔记工具）",
        "file": "examples/step6_full_agent.py",
    },
    {
        "id": "7",
        "title": "交互式对话（全功能）",
        "desc": "启动命令行交互模式，自由对话",
        "file": None,  # 特殊处理
        "cmd": ["uv", "run", "ai-life-agent"],
    },
]


def print_menu():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║     AI Life Agent —— 学习 Agent 开发的 6 步演示        ║")
    print("╠" + "═" * 58 + "╣")
    for demo in DEMOS:
        print(f"║  [{demo['id']}] {demo['title']:<32}         ║")
        print(f"║      {demo['desc']:<50}  ║")
        print("║" + " " * 58 + "║")
    print("║  [q] 退出                                            ║")
    print("╚" + "═" * 58 + "╝")


def check_env():
    """检查环境配置。"""
    from ai_life_agent.config import settings

    print("\n环境检查：")
    if settings.minimax_api_key:
        masked = settings.minimax_api_key[:8] + "..." + settings.minimax_api_key[-4:]
        print(f"  ✅ MINIMAX_API_KEY: {masked}")
    else:
        print("  ❌ MINIMAX_API_KEY: 未配置！")
        print("     请在 .env 文件中添加: MINIMAX_API_KEY=你的密钥")
        print("     获取密钥: https://platform.minimaxi.com/user-center/basic-information/interface-key")
        return False

    print(f"  ✅ 对话模型: {settings.chat_model}")
    print(f"  ✅ TTS 模型: {settings.tts_model}")
    return True


def run_demo(demo: dict) -> None:
    """运行指定演示。"""
    print(f"\n{'=' * 60}")
    print(f"  运行: {demo['title']}")
    print(f"{'=' * 60}\n")

    if demo.get("cmd"):
        # 特殊命令（交互式）
        subprocess.run(demo["cmd"])
        return

    script = Path(demo["file"])
    if not script.exists():
        print(f"❌ 文件不存在: {script}")
        return

    result = subprocess.run(
        [sys.executable, str(script)],
        env={**__import__("os").environ},
    )

    if result.returncode != 0:
        print(f"\n❌ 演示运行出错（返回码 {result.returncode}）")


def main():
    print("\n" + "=" * 60)
    print("  欢迎使用 AI Life Agent 演示程序")
    print("=" * 60)

    # 检查环境
    ok = check_env()
    if not ok:
        print("\n继续运行（部分功能可能不可用）...")

    while True:
        print_menu()

        try:
            choice = input("\n请选择演示 [1-7, q 退出]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if choice in ("q", "quit", "exit", "退出"):
            print("\n再见！感谢使用 AI Life Agent 🎉")
            break

        demo = next((d for d in DEMOS if d["id"] == choice), None)
        if demo is None:
            print(f"❌ 无效选择: {choice!r}，请输入 1-7 或 q")
            continue

        try:
            run_demo(demo)
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")

        input("\n按 Enter 继续...")


if __name__ == "__main__":
    main()
