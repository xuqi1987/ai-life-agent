"""
╔══════════════════════════════════════════════════════════════╗
║  第 3 步：ReAct —— 让 AI 自己"想清楚再行动"                   ║
╚══════════════════════════════════════════════════════════════╝

学习目标
--------
理解 ReAct 模式，以及为什么它是现代 Agent 的核心。

核心概念
--------
问题：step2 中工具调用是"一次性的"，如果需要多步操作怎么办？
      例如：「先查天气，再根据天气推荐穿衣，最后告诉我要不要带伞」

解决：ReAct 循环
  Re = Reasoning（推理）：AI 思考下一步该做什么
  Act = Acting（行动）：AI 调用工具获取信息
  每次 Observation（观察工具结果）后，AI 继续推理，决定是继续行动还是给出答案。

ReAct 循环示意图
----------------
  ┌─────────────────────────────────────────┐
  │            ReAct 循环                    │
  │                                         │
  │  用户输入 → Thought（我该怎么做？）       │
  │               ↓                        │
  │            Action（调用某个工具）        │
  │               ↓                        │
  │          Observation（工具返回结果）      │
  │               ↓                        │
  │         再次 Thought（根据结果继续思考）   │
  │               ↓                        │
  │         ...循环直到...                  │
  │               ↓                        │
  │          Answer（最终回答用户）           │
  └─────────────────────────────────────────┘

运行方法
--------
  uv run python examples/step3_react.py
"""

from ai_life_agent.core.react import ReActExecutor
from ai_life_agent.tools.builtin import register_builtin_tools
from ai_life_agent.tools.registry import ToolParameter, ToolRegistry


def main():
    print("=" * 60)
    print("  第 3 步：ReAct 推理循环")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. 准备工具注册表（和 step2 一样，但这次用内置工具）
    # --------------------------------------------------------
    registry = ToolRegistry()
    register_builtin_tools(registry)  # 注册: calculator, get_current_time, echo

    # 额外加一个"天气"工具（模拟）
    registry.register_tool(
        name="get_weather",
        description="查询指定城市的天气情况。",
        parameters=[
            ToolParameter("city", "城市名，例如：北京、上海", "string", required=True)
        ],
        func=lambda city: {
            "北京": "晴天 22°C，建议穿薄外套",
            "上海": "阴天 19°C，建议带雨伞",
            "深圳": "多云 28°C，穿短袖即可",
        }.get(city, f"{city}: 晴天 25°C"),
    )

    print(f"\n已注册工具: {registry.list_tools()}")

    # --------------------------------------------------------
    # 2. 创建 ReAct 执行器
    # --------------------------------------------------------
    executor = ReActExecutor(
        registry=registry,
        verbose=True,      # 开启详细日志，能看到 AI 的"思考过程"
        max_iterations=5,  # 最多循环 5 次，防止死循环
    )

    # --------------------------------------------------------
    # 【演示 1】需要多步骤的问题
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("【演示 1】多步骤问题：计算 + 获取时间")
    print("=" * 60)
    print("\n你   : 帮我算一下 sqrt(256)，同时告诉我现在几点了\n")

    result1 = executor.run("帮我算一下 sqrt(256)，同时告诉我现在几点了")

    print("\n" + "-" * 40)
    print(f"最终回答: {result1.answer}")
    print(f"共迭代: {result1.total_iterations} 次")

    # --------------------------------------------------------
    # 【演示 2】天气 + 推理
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("【演示 2】查天气然后给建议")
    print("=" * 60)
    print("\n你   : 上海今天天气怎么样？要不要带伞？\n")

    result2 = executor.run("上海今天天气怎么样？要不要带伞？")

    print("\n" + "-" * 40)
    print(f"最终回答: {result2.answer}")
    print(f"共迭代: {result2.total_iterations} 次")

    # --------------------------------------------------------
    # 【演示 3】查看完整推理链路（透明的 AI 思考过程）
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("【演示 3】查看完整推理链路（Thought/Action/Observation）")
    print("=" * 60)
    print("\n你   : 计算 100 * 200 + 300，然后告诉我结果的平方根\n")

    result3 = executor.run("计算 100 * 200 + 300，然后告诉我结果的平方根")

    print("\n--- 完整推理链路 ---")
    print(result3.format_trace())
    print(f"\n最终回答: {result3.answer}")

    # --------------------------------------------------------
    # 小结
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  小结：ReAct 的三个要素")
    print("=" * 60)
    print("""
  1️⃣  Thought（推理）
      AI 思考"我现在知道什么？我还需要什么？下一步该怎么做？"
      对应代码：LLM 返回 thinking block（思维链）

  2️⃣  Action（行动）
      AI 决定调用哪个工具、传什么参数
      对应代码：LLM 返回 tool_use block，我们执行 registry.execute()

  3️⃣  Observation（观察）
      工具执行结果，喂回给 AI 继续思考
      对应代码：构建 tool_result 消息，发回给 LLM

  循环结束条件：AI 不再需要工具（stop_reason = "end_turn"）

  下一步：学习如何写自己的工具 → step4_custom_tool.py
""")


if __name__ == "__main__":
    main()
