# 从零学习 Agent 开发 —— 完整教程

> 面向完全不懂 Agent 的初学者，一步一步教你从 0 到 1 构建智能 Agent。

---

## 目录

1. [什么是 Agent？](#一什么是-agent)
2. [环境准备](#二环境准备)
3. [第 1 步：和大模型说话](#三第-1-步和大模型说话)
4. [第 2 步：给 Agent 工具](#四第-2-步给-agent-工具)
5. [第 3 步：ReAct 推理循环](#五第-3-步react-推理循环)
6. [第 4 步：写自己的工具](#六第-4-步写自己的工具)
7. [第 5 步：语音合成](#七第-5-步语音合成)
8. [第 6 步：完整 Agent](#八第-6-步完整-agent)
9. [架构全景图](#九架构全景图)
10. [常见问题](#十常见问题)

---

## 一、什么是 Agent？

### 1.1 用一个类比理解

想象你雇了一个**全能助理**：

```
你说："帮我安排明天下午 3 点和张总的会议，查一下北京明天天气，
       如果下雨就提醒我带伞，然后把这些都记到我的日历里。"

助理做的事：
  1. 🧠 思考：我需要查天气、创建日历事项、发提醒
  2. 📞 行动：调用天气 API → 得到"明天有雨"
  3. 🧠 思考：有雨，要提醒带伞；继续创建日历
  4. 📅 行动：调用日历 API → 创建会议事项
  5. 📱 行动：发送提醒 → "记得带伞"
  6. ✅ 回答：所有任务完成，告知你结果
```

这个助理就是 **Agent**，它的关键能力是：
- **理解意图**：知道你真正想要什么
- **拆分任务**：把复杂请求分解为可执行的步骤
- **使用工具**：能调用外部 API、数据库、代码等
- **自主决策**：自己决定下一步该做什么

### 1.2 Agent vs 普通 ChatBot 的区别

| 对比项 | 普通 ChatBot | Agent |
|--------|-------------|-------|
| 知识来源 | 训练数据（有截止日期） | 训练数据 + **实时工具** |
| 任务能力 | 聊天、问答 | 聊天 + **执行操作** |
| 实时信息 | ❌ 不知道今天天气 | ✅ 调用天气 API |
| 操作能力 | ❌ 不能发邮件 | ✅ 调用邮件工具 |
| 多步任务 | 只能给出建议 | **自主完成** |

### 1.3 本项目的 Agent 架构

```
┌────────────────────────────────────────┐
│              你的问题                   │
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│         ReAct 推理循环                  │
│  ┌──────────┐  ┌──────────┐           │
│  │ 思考     │→ │ 行动     │           │
│  │(Thought) │  │(Action)  │           │
│  └──────────┘  └────┬─────┘           │
│        ↑            │                 │
│        │            ▼                 │
│  ┌─────────────────────────┐          │
│  │ 观察 (Observation)       │          │
│  │ 工具执行结果              │          │
│  └─────────────────────────┘          │
│            直到给出最终答案             │
└────────────────────────────────────────┘
          ↕                ↕
┌──────────────┐  ┌──────────────────────┐
│  MiniMax M2.7│  │     工具注册表         │
│  大语言模型  │  │ calculator/time/tts  │
│  (负责推理)  │  │ search/notes/...     │
└──────────────┘  └──────────────────────┘
```

---

## 二、环境准备

### 2.1 安装依赖

```bash
# 克隆项目
git clone https://github.com/xuqi1987/ai-life-agent.git
cd ai-life-agent

# 安装（使用 uv 包管理器）
uv sync
```

### 2.2 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 MiniMax API Key
# 获取地址：https://platform.minimaxi.com/user-center/basic-information/interface-key
```

`.env` 文件内容：
```
MINIMAX_API_KEY=sk-你的密钥
```

### 2.3 验证配置

```bash
# 运行第 1 个示例，看到 AI 回答说明配置成功
uv run python examples/step1_hello_llm.py
```

---

## 三、第 1 步：和大模型说话

> **对应示例**：`examples/step1_hello_llm.py`

### 3.1 核心代码（只需 7 行）

```python
from ai_life_agent.llm.client import MiniMaxClient

client = MiniMaxClient()          # 创建客户端（自动读取 API Key）

response = client.chat(
    messages=[{"role": "user", "content": "你好！"}]
)

print(response.text)              # 输出 AI 的回答
```

### 3.2 消息格式详解

```python
messages = [
    # role 只有两种：
    #   "user"      —— 你说的话
    #   "assistant" —— AI 说的话
    
    {"role": "user",      "content": "我叫小明"},
    {"role": "assistant", "content": "你好，小明！"},
    {"role": "user",      "content": "你还记得我叫什么吗？"},
    #                                ↑ AI 会根据历史记住"小明"
]
```

> ⚠️ **重要**：LLM 本身没有记忆！每次请求都要把完整的历史对话一起发送，
> 它才能"记住"上文。

### 3.3 用 system 设定 AI 角色

```python
response = client.chat(
    messages=[{"role": "user", "content": "你好"}],
    system="你是一个专业的厨师助手，说话时总会提到食谱和食材。",
)
# AI 会以厨师助手的身份回答
```

### 3.4 运行示例

```bash
uv run python examples/step1_hello_llm.py
```

**预期输出**：
```
第 1 步：和大模型直接对话

【示例 1】单轮问答

你   : 你好！请用一句话介绍你自己。
AI  : 你好！我是由MiniMax开发的AI助手，专注于提供智能对话服务。

【示例 2】多轮对话（LLM 记住上文）

你   : 我叫小明，我喜欢 Python。
AI  : 你好，小明！Python 是一门很棒的语言。
你   : 你还记得我叫什么名字吗？
AI  : 当然记得，你叫小明，而且你喜欢 Python！
```

---

## 四、第 2 步：给 Agent 工具

> **对应示例**：`examples/step2_tools.py`

### 4.1 为什么需要工具？

```python
# 问题：
response = client.chat([{"role": "user", "content": "今天北京天气怎么样？"}])
# AI 可能会说："我的训练数据截止于某日，无法提供实时天气。"

# 解决：给 AI 一个"天气查询工具"，让它能获取实时数据
```

### 4.2 三步添加工具

**第一步**：写 Python 函数

```python
def get_weather(city: str) -> str:
    """查询天气（这里用模拟数据）。"""
    return f"{city}：晴天，25°C"
```

**第二步**：注册工具（告诉 AI 怎么用）

```python
from ai_life_agent.tools.registry import ToolRegistry, ToolParameter

registry = ToolRegistry()
registry.register_tool(
    name="get_weather",               # 工具名称（AI 调用时用这个名字）
    description="查询指定城市的天气",  # 描述（越清晰 AI 越知道何时用）
    parameters=[
        ToolParameter(
            name="city",              # 参数名
            description="城市名称",   # 参数描述
            type="string",            # 参数类型
            required=True,            # 是否必填
        )
    ],
    func=get_weather,                 # 实际执行的函数
)
```

**第三步**：传给 LLM

```python
tools_schema = registry.get_schemas()   # 转为 AI 能理解的格式

response = client.chat(
    messages=[{"role": "user", "content": "北京天气"}],
    tools=tools_schema,                  # 告诉 AI 有这些工具
)

if response.has_tool_calls:
    # AI 决定调用工具
    result = registry.execute(
        response.tool_calls[0]["name"],   # 工具名
        response.tool_calls[0]["input"],  # 参数
    )
    print(result)  # "北京：晴天，25°C"
```

### 4.3 工具调用的完整流程

```
你的程序                    MiniMax AI
    │                          │
    │── 发送消息 + 工具列表 ──►│
    │                          │ 思考：需要查天气，用 get_weather("北京")
    │◄── 返回工具调用请求 ──── │
    │                          │
    │ 执行: get_weather("北京")
    │ 结果: "晴天 25°C"
    │                          │
    │── 发送工具结果 ─────────►│
    │                          │ 思考：有了天气数据，可以回答了
    │◄── 返回最终回答 ──────── │
```

### 4.4 运行示例

```bash
uv run python examples/step2_tools.py
```

---

## 五、第 3 步：ReAct 推理循环

> **对应示例**：`examples/step3_react.py`

### 5.1 为什么需要 ReAct？

step2 的工具调用是**单次**的，但实际问题经常需要**多步骤**：

```
用户："上海明天天气怎样？如果下雨提醒我带伞，同时帮我算算今天是几月几号"

需要：
  1. 调用天气工具（上海）
  2. 判断是否下雨
  3. 调用时间工具
  4. 综合所有结果给出回答

→ 这需要 AI 自主"循环决策"，这就是 ReAct！
```

### 5.2 ReAct 就 3 行代码

```python
from ai_life_agent.core.react import ReActExecutor
from ai_life_agent.tools.builtin import register_builtin_tools
from ai_life_agent.tools.registry import ToolRegistry

# 1. 准备工具
registry = ToolRegistry()
register_builtin_tools(registry)  # 注册计算器、时间等内置工具

# 2. 创建执行器
executor = ReActExecutor(registry=registry, verbose=True)

# 3. 运行（AI 自动处理所有工具调用）
result = executor.run("计算 sqrt(144) 并告诉我现在几点")

print(result.answer)  # "sqrt(144)=12，现在是 14:30:00"
```

### 5.3 verbose=True 时看到的输出

```
[ReAct 迭代 1/10] 发送消息给 LLM...
  思维链: 用户想要两件事：1)计算sqrt(144) 2)获取当前时间
          这两个可以同时处理...
  执行工具: calculator({"expression": "sqrt(144)"})
  工具结果: 12
  执行工具: get_current_time({})
  工具结果: 2026-03-31 14:30:00

[ReAct 迭代 2/10] 发送消息给 LLM...
  最终回答: sqrt(144) = 12，当前时间是 2026-03-31 14:30:00
```

### 5.4 ReAct 的核心代码（简化版）

```python
messages = [{"role": "user", "content": user_input}]

for i in range(max_iterations):
    response = llm.chat(messages, tools=tools)
    
    if not response.has_tool_calls:
        return response.text    # ✅ 有了最终答案，结束循环
    
    # 执行所有工具
    tool_results = []
    for tc in response.tool_calls:
        result = registry.execute(tc["name"], tc["input"])
        tool_results.append(result)
    
    # 把结果喂回给 AI，继续下一轮
    messages.append({"role": "assistant", "content": response.raw_content})
    messages.append({"role": "user", "content": tool_results_as_messages})
```

---

## 六、第 4 步：写自己的工具

> **对应示例**：`examples/step4_custom_tool.py`

### 6.1 工具开发模板

```python
# ============================================================
# 工具函数（只管功能，不用管 AI）
# ============================================================
def my_awesome_tool(param1: str, param2: int = 10) -> str:
    """
    工具的功能说明（写清楚，LLM 会读这个来理解工具用途）。
    
    Args:
        param1: 参数1的说明
        param2: 参数2的说明（有默认值=非必填）
    """
    # 你的业务逻辑
    result = f"处理了 {param1}，次数={param2}"
    return result  # 返回字符串


# ============================================================
# 注册工具（让 AI 知道这个工具存在）
# ============================================================
from ai_life_agent.tools.registry import ToolRegistry, ToolParameter

registry = ToolRegistry()
registry.register_tool(
    name="my_awesome_tool",
    description="这个工具能做什么（对 LLM 说清楚，它才知道何时用）",
    parameters=[
        ToolParameter("param1", "参数1的描述", "string", required=True),
        ToolParameter("param2", "参数2的描述", "number", required=False),
    ],
    func=my_awesome_tool,
)
```

### 6.2 工具类型最佳实践

| 工具类型 | 例子 | 返回格式建议 |
|---------|------|------------|
| 查询类 | 查天气、查股价 | 字符串描述 |
| 计算类 | 数学计算、单位换算 | "结果: X" |
| 操作类 | 发邮件、保存文件 | "成功/失败" |
| 获取信息 | 读数据库、读文件 | 内容字符串 |

### 6.3 用装饰器快速注册

```python
@registry.register(description="获取股票价格")
def get_stock_price(symbol: str) -> str:
    """查询股票实时价格。"""
    # 调用你的股票 API
    return f"{symbol}: ¥100.50 (+2.3%)"

# 等价于：
registry.register_tool("get_stock_price", "获取股票价格", [...], get_stock_price)
```

---

## 七、第 5 步：语音合成

> **对应示例**：`examples/step5_tts.py`

### 7.1 一行代码转语音

```python
from ai_life_agent.tools.tts import TTS

tts = TTS()
tts.speak_to_file("你好，我是 AI 助手！", "output.mp3")
# 生成 output.mp3，用播放器打开即可收听
```

### 7.2 音色选择

```python
# 男声
tts.speak_to_file("文本", "out.mp3", voice_id="male-qn-qingse")

# 女声
tts.speak_to_file("文本", "out.mp3", voice_id="female-shaonv")

# 播音腔
tts.speak_to_file("文本", "out.mp3", voice_id="presenter_male")
```

### 7.3 语速调节

```python
# 慢速（0.7倍）
tts.speak_to_file("文本", "slow.mp3", speed=0.7)

# 快速（1.5倍）
tts.speak_to_file("文本", "fast.mp3", speed=1.5)
```

---

## 八、第 6 步：完整 Agent

> **对应示例**：`examples/step6_full_agent.py`

### 8.1 多轮对话 + 记忆

```python
from ai_life_agent.core.react import ReActExecutor
from ai_life_agent.memory import Memory

# 记忆存储对话历史
memory = Memory()
executor = ReActExecutor(registry=registry)

def chat(user_input: str) -> str:
    # 取历史记录
    history = memory.get_conversation_history()
    
    # 运行 Agent（带历史）
    result = executor.run(user_input, conversation_history=history)
    
    # 存入历史（供下次使用）
    memory.add_turn("user", user_input)
    memory.add_turn("assistant", result.answer)
    
    return result.answer

# 使用
chat("我叫小明")                   # 第 1 轮
chat("你还记得我叫什么名字吗？")    # 第 2 轮（AI 会说"小明"）
```

### 8.2 使用封装好的 Agent 类

```python
from ai_life_agent.core.agent import Agent

# 最简单的用法：两行代码
agent = Agent()
print(agent.run("现在几点了？"))
```

### 8.3 运行交互式对话

```bash
uv run ai-life-agent
```

---

## 九、架构全景图

```
ai-life-agent/
│
├── 📄 demo.py                ← 一键演示入口（从这里开始！）
├── 📄 .env                   ← API Key 配置（不提交 git！）
│
├── 📁 examples/              ← 6 步学习示例
│   ├── step1_hello_llm.py    ← LLM 对话
│   ├── step2_tools.py        ← 工具调用
│   ├── step3_react.py        ← ReAct 循环
│   ├── step4_custom_tool.py  ← 自定义工具
│   ├── step5_tts.py          ← 语音合成
│   └── step6_full_agent.py   ← 完整 Agent
│
├── 📁 src/ai_life_agent/
│   ├── config.py             ← API Key 和模型配置
│   ├── llm/
│   │   └── client.py         ← MiniMax API 客户端
│   ├── core/
│   │   ├── agent.py          ← Agent 主类（入口）
│   │   └── react.py          ← ReAct 推理循环（核心！）
│   ├── tools/
│   │   ├── registry.py       ← 工具注册表
│   │   ├── builtin.py        ← 内置工具（计算器/时间/echo）
│   │   └── tts.py            ← 语音合成工具
│   └── memory/
│       └── __init__.py       ← 对话记忆管理
│
└── 📁 docs/
    ├── TUTORIAL.md           ← 本教程
    ├── QUICK_START.md        ← 快速上手（5分钟）
    ├── ARCHITECTURE.md       ← 架构设计
    ├── REACT.md              ← ReAct 原理
    └── TOOLS.md              ← 工具系统
```

---

## 十、常见问题

### Q1: 运行时报 `MINIMAX_API_KEY 未配置`

```bash
# 检查 .env 文件是否存在
cat .env

# 如果没有，创建它
cp .env.example .env
# 然后编辑 .env，填入你的 API Key
```

### Q2: 报错 `socksio` 或代理相关错误

```bash
# 安装 SOCKS 代理支持
uv add "httpx[socks]"
```

### Q3: TTS 报错 `not support model`

检查你的 MiniMax 账户套餐，Plus 套餐支持 `speech-2.8-hd`。

在 `.env` 中配置：
```
TTS_MODEL=speech-2.8-hd
```

### Q4: AI 不使用工具，直接回答

这是正常行为。AI 会自主判断是否需要工具，如果问题能直接回答，就不会调用工具。
如果你希望强制使用工具，可以在 system prompt 中明确说明。

### Q5: ReAct 循环超过 10 次仍无答案

可能是 AI 陷入了无效循环，增加工具描述的清晰度，或调整 system prompt。

### Q6: 如何查看 AI 的思考过程？

```python
executor = ReActExecutor(registry=registry, verbose=True)
# 或者
result = executor.run("你的问题")
print(result.format_trace())  # 打印完整推理链路
```

---

## 下一步学习

| 阶段 | 目标 | 参考 |
|-----|------|------|
| ✅ 已完成 | LLM 对话、工具调用、ReAct | 本教程 |
| 🔜 v0.4 | SQLite 持久化记忆、自动摘要 | SPEC.md |
| 🔜 v0.5 | 多 Agent 协作（Supervisor 模式） | SPEC.md |
| 🔜 v0.6 | Skill 插件系统 | SPEC.md |
| 🔜 v0.7 | 语音输入（ASR） + 人脸识别 | SPEC.md |

---

*本教程由 AI Life Agent 项目提供，使用 MiniMax M2.7 驱动。*
