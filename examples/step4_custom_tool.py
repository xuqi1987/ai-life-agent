"""
╔══════════════════════════════════════════════════════════════╗
║  第 4 步：写你的第一个工具 —— 从"工具消费者"变成"工具开发者"    ║
╚══════════════════════════════════════════════════════════════╝

学习目标
--------
学会自己编写工具，让 Agent 拥有你想要的任何能力。

工具开发三步法
--------------
  第一步：写 Python 函数（实现功能）
  第二步：注册工具（告诉 AI 这个工具的名称、描述、参数）
  第三步：测试（让 AI 在对话中使用它）

工具设计原则
------------
  ✅ 函数只做一件事（单一职责）
  ✅ 参数尽量少（LLM 更容易理解）
  ✅ 返回字符串或可转为字符串的结果
  ✅ 描述要清晰，让 LLM 知道什么时候用

运行方法
--------
  uv run python examples/step4_custom_tool.py
"""

import random

from ai_life_agent.core.react import ReActExecutor
from ai_life_agent.tools.registry import ToolParameter, ToolRegistry

# ============================================================
# 🛠️  自定义工具 1：随机励志名言
# ============================================================

QUOTES = [
    "成功不是终点，失败也不是终点，勇气才是持续的力量。 —— 丘吉尔",
    "你的时间有限，不要浪费在过别人的生活上。 —— 乔布斯",
    "种一棵树最好的时机是十年前，其次是现在。 —— 中国谚语",
    "生活中最重要的事，不是你所处的位置，而是你所朝的方向。 —— 霍姆斯",
    "坚持下去的理由只需要一个，放弃的理由却可以有一千个。",
]


def get_motivational_quote(topic: str = "通用") -> str:
    """
    这就是工具函数的本体 —— 一个普通的 Python 函数。

    topic: 名言主题，目前返回随机名言
    """
    quote = random.choice(QUOTES)
    return f"【{topic}励志名言】{quote}"


# ============================================================
# 🛠️  自定义工具 2：生成简单的待办事项列表
# ============================================================

TODO_LIST: list[str] = []  # 全局待办列表（演示用，真实项目用数据库）


def add_todo(task: str) -> str:
    """添加一个待办事项。"""
    TODO_LIST.append(task)
    return f"✅ 已添加待办：「{task}」（当前共 {len(TODO_LIST)} 项）"


def list_todos() -> str:
    """查看所有待办事项。"""
    if not TODO_LIST:
        return "📋 当前没有待办事项"
    items = "\n".join(f"  {i+1}. {task}" for i, task in enumerate(TODO_LIST))
    return f"📋 待办列表（共 {len(TODO_LIST)} 项）：\n{items}"


def clear_todos() -> str:
    """清空所有待办事项。"""
    count = len(TODO_LIST)
    TODO_LIST.clear()
    return f"🗑️  已清空 {count} 个待办事项"


# ============================================================
# 🛠️  自定义工具 3：简单的单位换算
# ============================================================

def convert_unit(value: float, from_unit: str, to_unit: str) -> str:
    """
    常用单位换算工具。
    支持：温度（摄氏/华氏/开尔文）、长度（米/英尺/英寸）
    """
    conversions = {
        # 温度
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
        ("celsius", "kelvin"):     lambda v: v + 273.15,
        ("kelvin", "celsius"):     lambda v: v - 273.15,
        # 长度
        ("meter", "feet"):         lambda v: v * 3.28084,
        ("feet", "meter"):         lambda v: v / 3.28084,
        ("meter", "inch"):         lambda v: v * 39.3701,
        ("inch", "meter"):         lambda v: v / 39.3701,
        ("kilometer", "mile"):     lambda v: v * 0.621371,
        ("mile", "kilometer"):     lambda v: v / 0.621371,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        supported = ", ".join(f"{a}→{b}" for a, b in conversions.keys())
        return f"❌ 不支持的换算：{from_unit} → {to_unit}。支持的换算：{supported}"

    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


# ============================================================
# 主程序：注册工具并用 ReAct 测试
# ============================================================

def main():
    print("=" * 60)
    print("  第 4 步：开发你自己的工具")
    print("=" * 60)

    # ----------------------------------------------------------
    # 注册所有自定义工具
    # ----------------------------------------------------------
    registry = ToolRegistry()

    # 工具1：励志名言
    registry.register_tool(
        name="get_motivational_quote",
        description="获取一条随机的励志名言，可以指定主题。适合在用户需要鼓励时调用。",
        parameters=[
            ToolParameter("topic", "名言主题，例如：学习、工作、生活", "string", required=False),
        ],
        func=get_motivational_quote,
    )

    # 工具2：待办事项管理（三个工具）
    registry.register_tool(
        name="add_todo",
        description="添加一个待办事项到列表中。",
        parameters=[
            ToolParameter("task", "要添加的任务描述", "string", required=True),
        ],
        func=add_todo,
    )
    registry.register_tool(
        name="list_todos",
        description="查看当前所有待办事项列表。",
        parameters=[],  # 没有参数
        func=list_todos,
    )
    registry.register_tool(
        name="clear_todos",
        description="清空所有待办事项。",
        parameters=[],
        func=clear_todos,
    )

    # 工具3：单位换算
    registry.register_tool(
        name="convert_unit",
        description=(
            "单位换算工具。支持温度（celsius/fahrenheit/kelvin）"
            "和长度（meter/feet/inch/kilometer/mile）换算。"
        ),
        parameters=[
            ToolParameter("value",     "要换算的数值",                    "number", required=True),
            ToolParameter("from_unit", "原始单位，例如 celsius, meter",   "string", required=True),
            ToolParameter("to_unit",   "目标单位，例如 fahrenheit, feet", "string", required=True),
        ],
        func=convert_unit,
    )

    print(f"\n✅ 已注册 {len(registry)} 个工具: {registry.list_tools()}")

    executor = ReActExecutor(registry=registry, verbose=True, max_iterations=6)

    # ----------------------------------------------------------
    # 测试 1：励志名言
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("【测试 1】励志名言")
    print("=" * 60)
    print("\n你   : 我今天学习遇到困难，给我一句鼓励的话\n")
    r1 = executor.run("我今天学习遇到困难，给我一句鼓励的话")
    print(f"\n最终: {r1.answer}")

    # ----------------------------------------------------------
    # 测试 2：待办事项管理
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("【测试 2】待办事项管理（多步操作）")
    print("=" * 60)
    print("\n你   : 帮我添加三个待办事项：学习Python、看技术书、写笔记，然后显示列表\n")
    r2 = executor.run("帮我添加三个待办事项：学习Python、看技术书、写笔记，然后显示列表")
    print(f"\n最终: {r2.answer}")
    print(f"迭代次数: {r2.total_iterations}")

    # ----------------------------------------------------------
    # 测试 3：单位换算
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("【测试 3】单位换算")
    print("=" * 60)
    print("\n你   : 100摄氏度是多少华氏度？1.75米是多少英尺？\n")
    r3 = executor.run("100摄氏度是多少华氏度？1.75米是多少英尺？")
    print(f"\n最终: {r3.answer}")

    # ----------------------------------------------------------
    # 小结
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  工具开发三步法 总结")
    print("=" * 60)
    print("""
  第一步：写函数（只关注功能逻辑，不用管 AI）
  ─────────────────────────────────────────
  def my_tool(param: str) -> str:
      # 你的业务逻辑
      return "结果"

  第二步：注册（告诉 AI 这个工具的信息）
  ─────────────────────────────────────────
  registry.register_tool(
      name="my_tool",          # 工具名（LLM 用来调用）
      description="工具用途",   # 工具描述（越清晰越好！）
      parameters=[...],        # 参数列表
      func=my_tool,            # 指向你的函数
  )

  第三步：接入 ReAct（AI 自动决定何时调用）
  ─────────────────────────────────────────
  executor = ReActExecutor(registry=registry)
  result = executor.run("用户输入")

  下一步：语音合成演示 → step5_tts.py
""")


if __name__ == "__main__":
    main()
