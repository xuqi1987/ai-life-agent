"""
╔══════════════════════════════════════════════════════════════╗
║  第 6 步：完整 Agent —— 多轮对话 + 工具 + 记忆                 ║
╚══════════════════════════════════════════════════════════════╝

学习目标
--------
把前面所有学到的内容整合起来：
- 多轮对话（AI 记住你说过什么）
- 工具调用（AI 自动使用工具）
- 对话历史管理

这一步展示了一个"真正可用"的 AI 助手的雏形。

运行方法
--------
  uv run python examples/step6_full_agent.py
"""

from ai_life_agent.core.react import ReActExecutor
from ai_life_agent.memory import Memory
from ai_life_agent.tools.builtin import register_builtin_tools
from ai_life_agent.tools.registry import ToolParameter, ToolRegistry

# ============================================================
# 构建一个功能丰富的工具集
# ============================================================

def build_tools() -> ToolRegistry:
    """构建包含多种工具的注册表。"""
    registry = ToolRegistry()

    # 内置工具：计算器、时间、echo
    register_builtin_tools(registry)

    # 自定义：笔记工具
    notes: dict[str, str] = {}

    def save_note(title: str, content: str) -> str:
        notes[title] = content
        return f"✅ 笔记「{title}」已保存"

    def get_note(title: str) -> str:
        return notes.get(title, f"❌ 未找到笔记「{title}」")

    def list_notes() -> str:
        if not notes:
            return "📓 没有笔记"
        return "📓 笔记列表：\n" + "\n".join(f"  - {t}" for t in notes)

    registry.register_tool(
        "save_note", "保存一条笔记，需要标题和内容。",
        [ToolParameter("title", "笔记标题", "string"),
         ToolParameter("content", "笔记内容", "string")],
        save_note,
    )
    registry.register_tool(
        "get_note", "根据标题读取笔记内容。",
        [ToolParameter("title", "笔记标题", "string")],
        get_note,
    )
    registry.register_tool(
        "list_notes", "列出所有已保存的笔记标题。",
        [],
        list_notes,
    )

    return registry


# ============================================================
# 多轮 Agent —— 带记忆的完整对话
# ============================================================

class MultiTurnAgent:
    """
    带记忆的多轮对话 Agent。

    核心设计：
    - Memory：存储对话历史（用户说了什么、AI 回了什么）
    - ReActExecutor：每次对话都能使用工具
    - 历史传递：每次都把历史对话带给 LLM，让它"记住"上文
    """

    def __init__(self):
        self.memory = Memory()      # 存对话历史
        self.registry = build_tools()
        self.executor = ReActExecutor(
            registry=self.registry,
            verbose=False,  # 关闭详细日志，让输出更干净
            system_prompt="""你是一个聪明的 AI 助手，名字叫"小智"。

你的特点：
- 亲切友好，说话自然
- 会记住用户在本次对话中说过的事情
- 善于使用工具完成任务
- 回答简洁，不啰嗦

已有工具：计算器、当前时间、笔记（保存/读取/列表）
""",
        )
        self.turn_count = 0

    def chat(self, user_input: str) -> str:
        """
        发送一条消息给 Agent，返回回答。

        内部流程：
        1. 把用户输入加入历史
        2. 将完整历史传给 ReAct 执行器
        3. 获得回答
        4. 把回答也加入历史（供下次使用）
        """
        self.turn_count += 1

        # 把当前对话历史格式化为 LLM 可用的格式
        history = self.memory.get_conversation_history()

        # 运行 ReAct（传入历史 + 当前问题）
        result = self.executor.run(
            user_input=user_input,
            conversation_history=history,
        )

        answer = result.answer

        # 把这一轮对话存入记忆
        self.memory.add_turn("user", user_input)
        self.memory.add_turn("assistant", answer)

        return answer

    def show_history(self):
        """显示完整对话历史。"""
        history = self.memory.get_conversation_history()
        print(f"\n--- 对话历史（共 {len(history)} 条） ---")
        for msg in history:
            who = "你    " if msg["role"] == "user" else "小智  "
            content = msg["content"]
            # 截断长文本
            if len(content) > 80:
                content = content[:80] + "..."
            print(f"{who}: {content}")
        print()


def run_demo():
    """运行一个模拟的多轮对话演示。"""
    print("=" * 60)
    print("  第 6 步：完整多轮对话 Agent")
    print("=" * 60)
    print("\n演示场景：和 AI 助手「小智」进行多轮对话\n")

    agent = MultiTurnAgent()

    # ----------------------------------------------------------
    # 模拟多轮对话
    # ----------------------------------------------------------
    conversations = [
        # 第 1 轮：自我介绍
        "你好！你叫什么名字？",

        # 第 2 轮：记忆测试（AI 应该记住用户名）
        "我叫大明，我是一名 Python 初学者",

        # 第 3 轮：工具调用（计算）
        "帮我算一下：如果我每天学习 2.5 小时，一个月（30天）共学多少小时？",

        # 第 4 轮：工具调用（时间）
        "现在几点了？",

        # 第 5 轮：记忆 + 工具（笔记）
        "帮我保存一条笔记，标题是「学习计划」，内容是「每天学习2.5小时Python」",

        # 第 6 轮：读取笔记
        "刚才我保存的笔记内容是什么？",

        # 第 7 轮：记忆测试
        "你还记得我的名字吗？我今天学了什么？",
    ]

    for i, user_msg in enumerate(conversations, 1):
        print(f"[第 {i} 轮]")
        print(f"你    : {user_msg}")
        response = agent.chat(user_msg)
        print(f"小智  : {response}")
        print()

    # 显示完整对话历史
    agent.show_history()

    print("=" * 60)
    print("  多轮对话演示完成")
    print("=" * 60)
    print(f"""
  总结：
  ✅ Agent 记住了用户名字（大明）
  ✅ Agent 正确使用了计算器工具
  ✅ Agent 能记录和读取笔记
  ✅ 共进行 {agent.turn_count} 轮对话

  关键代码：
  ─────────────────────────────────────
  agent = MultiTurnAgent()
  response = agent.chat("你的问题")
  ─────────────────────────────────────

  恭喜！你已经掌握了 Agent 开发的核心技能 🎉

  完整的交互式对话体验，请运行：
    uv run ai-life-agent
""")


if __name__ == "__main__":
    run_demo()
