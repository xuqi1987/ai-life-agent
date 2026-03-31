"""CLI entry point — 交互式命令行界面。

使用方法::

    uv run ai-life-agent            # 启动交互式对话
    uv run ai-life-agent --verbose  # 启用详细日志（显示思维链）
"""

from __future__ import annotations

import logging
import sys


def _setup_logging(verbose: bool) -> None:
    """配置日志级别和格式。"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    """运行 Agent 命令行交互界面。"""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    _setup_logging(verbose)

    print("=" * 60)
    print("  ai-life-agent v0.3.0 — MiniMax M2.7 Multi-modal Agent")
    print("=" * 60)

    from ai_life_agent.config import settings
    from ai_life_agent.core.agent import Agent

    if not settings.minimax_api_key:
        print("\n⚠️  提示：MINIMAX_API_KEY 未配置，Agent 以降级模式运行。")
        print("   请在 .env 文件中设置 MINIMAX_API_KEY=<your_key>\n")
    else:
        print(f"\n✅ 已连接 MiniMax {settings.chat_model}")
        print("   工具: calculator / get_current_time / echo\n")

    print("输入 'quit' 或 'exit' 退出，输入 'clear' 清空对话历史。\n")

    agent = Agent()

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见！")
            break

        if user_input.lower() in ("clear", "清空"):
            agent.clear_history()
            print("（对话历史已清空）\n")
            continue

        response = agent.run(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
