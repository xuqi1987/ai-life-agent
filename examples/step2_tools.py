"""
╔══════════════════════════════════════════════════════════════╗
║  第 2 步：给 AI 一把"工具" —— 工具调用（Tool Calling）         ║
╚══════════════════════════════════════════════════════════════╝

学习目标
--------
理解什么是"工具调用"，以及为什么 Agent 需要工具。

核心概念
--------
问题：LLM 的知识有"截止日期"，它不知道实时信息（今天几号？天气如何？）
解决：给 LLM"工具"—— 真实的 Python 函数，LLM 决定何时调用

工具调用的流程
--------------
  ┌──────────┐    ① 发送消息+工具列表    ┌──────────┐
  │   你的   │ ──────────────────────► │  MiniMax  │
  │  Python  │                        │   M2.7   │
  │  程序    │ ◄────────────────────── │          │
  └──────────┘    ② LLM 说"我要调用    └──────────┘
       │            工具 X，参数是 Y"
       │
       ▼
  ③ 你的程序执行工具函数，得到结果
       │
       ▼
  ④ 把结果发回给 LLM
       │
       ▼
  ⑤ LLM 根据结果给出最终回答

运行方法
--------
  uv run python examples/step2_tools.py
"""

from ai_life_agent.llm.client import MiniMaxClient
from ai_life_agent.tools.registry import ToolParameter, ToolRegistry

# ============================================================
# 第一部分：定义工具（普通的 Python 函数）
# ============================================================

def get_weather(city: str) -> str:
    """
    模拟天气查询工具。
    真实场景中，这里可以调用和风天气、OpenWeatherMap 等 API。
    """
    # 模拟数据（实际项目中换成真实 API）
    weather_data = {
        "北京": "晴天，气温 22°C，微风",
        "上海": "多云，气温 25°C，东南风 3级",
        "广州": "阵雨，气温 28°C，湿度 85%",
        "深圳": "晴天，气温 27°C，能见度良好",
    }
    return weather_data.get(city, f"{city}: 暂无数据（未在数据库中）")


def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    BMI 计算工具（体重公斤 ÷ 身高米的平方）。
    """
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "偏瘦"
    elif bmi < 24:
        category = "正常"
    elif bmi < 28:
        category = "超重"
    else:
        category = "肥胖"
    return f"BMI = {bmi:.1f}，体型评估：{category}"


# ============================================================
# 第二部分：把工具"注册"到注册表
# 注册 = 告诉 AI "你有这些工具，以及每个工具怎么用"
# ============================================================

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # 注册天气工具
    registry.register_tool(
        name="get_weather",
        description="查询指定城市的实时天气情况。输入城市名，返回天气描述。",
        parameters=[
            ToolParameter(
                name="city",
                description="要查询天气的城市名，例如：北京、上海、广州",
                type="string",
                required=True,
            )
        ],
        func=get_weather,
    )

    # 注册 BMI 计算工具
    registry.register_tool(
        name="calculate_bmi",
        description="根据身高和体重计算 BMI 指数，并给出体型评估。",
        parameters=[
            ToolParameter(
                name="weight_kg",
                description="体重，单位：千克（kg）",
                type="number",
                required=True,
            ),
            ToolParameter(
                name="height_m",
                description="身高，单位：米（m），例如 1.75",
                type="number",
                required=True,
            ),
        ],
        func=calculate_bmi,
    )

    return registry


def main():
    print("=" * 60)
    print("  第 2 步：工具调用（Tool Calling）")
    print("=" * 60)

    client = MiniMaxClient()
    registry = build_registry()

    # 获取工具的 JSON Schema（这是 LLM 能理解的"工具说明书"）
    tools_schema = registry.get_schemas()

    print(f"\n已注册工具: {registry.list_tools()}")
    print("\n" + "-" * 40)

    # --------------------------------------------------------
    # 【演示 1】询问天气 —— 触发工具调用
    # --------------------------------------------------------
    print("\n【演示 1】询问北京天气\n")

    user_question = "北京今天天气怎么样？"
    print(f"你   : {user_question}")

    # 第一次调用 LLM（带工具列表）
    response = client.chat(
        messages=[{"role": "user", "content": user_question}],
        tools=tools_schema,
    )

    print(f"\n  → LLM 决定: 调用工具？= {response.has_tool_calls}")

    if response.has_tool_calls:
        # LLM 要求调用工具
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_params = tc["input"]
            print(f"  → 工具名称: {tool_name}")
            print(f"  → 工具参数: {tool_params}")

            # 执行工具（调用我们定义的 Python 函数）
            result = registry.execute(tool_name, tool_params)
            print(f"  → 工具结果: {result}")

        # 把工具结果发回给 LLM，让它给出最终回答
        final_response = client.chat(
            messages=[
                {"role": "user", "content": user_question},
                # assistant 的完整消息（含工具调用请求）必须原样回传
                {"role": "assistant", "content": response.raw_content},
                # 工具执行结果
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": response.tool_calls[0]["id"],
                            "content": result,
                        }
                    ],
                },
            ],
            tools=tools_schema,
        )
        print(f"\nAI  : {final_response.text}")
    else:
        print(f"AI  : {response.text}")

    # --------------------------------------------------------
    # 【演示 2】直接回答（不需要工具）
    # --------------------------------------------------------
    print("\n" + "-" * 40)
    print("\n【演示 2】不需要工具的问题\n")

    user_question2 = "Python 和 Java 哪个更适合初学者？"
    print(f"你   : {user_question2}")

    response2 = client.chat(
        messages=[{"role": "user", "content": user_question2}],
        tools=tools_schema,  # 即使提供了工具，LLM 也会判断不需要用
    )

    print(f"  → LLM 决定: 调用工具？= {response2.has_tool_calls}")
    print(f"AI  : {response2.text[:200]}...")

    # --------------------------------------------------------
    # 小结
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  小结")
    print("=" * 60)
    print("""
  ✅ 工具 = 普通的 Python 函数
  ✅ 工具注册 = 把函数"介绍"给 LLM（名称+描述+参数）
  ✅ LLM 自己决定：用不用工具、用哪个工具、传什么参数
  ✅ 你的程序负责执行工具，再把结果发回给 LLM

  下一步：让 AI 自动循环调用工具直到给出完整回答 → step3_react.py
""")


if __name__ == "__main__":
    main()
