# AI Life Agent — 架构设计文档

> 本文档介绍 AI Life Agent 的整体架构设计，适合想要理解或扩展系统的开发者阅读。

---

## 一、总体架构

```
┌──────────────────────────────────────────────────────────┐
│                    用户输入（文本）                         │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                      Agent                               │
│  - 维护对话历史（message list）                            │
│  - 管理工具注册表（ToolRegistry）                          │
│  - 驱动 ReAct 推理循环                                    │
│  - 优雅降级（无 API Key 时返回提示）                        │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                   ReActExecutor                          │
│                                                          │
│  ┌─────────┐    ┌──────────┐    ┌─────────────────────┐ │
│  │ Thought │ →  │  Action  │ →  │    Observation      │ │
│  │ (LLM)   │    │ (工具调用) │    │ (工具执行结果)       │ │
│  └─────────┘    └──────────┘    └──────────┬──────────┘ │
│       ↑                                     │           │
│       └─────────────────────────────────────┘           │
│                   （循环直到最终答案）                      │
└──────────┬───────────────────────┬───────────────────────┘
           │                       │
           ▼                       ▼
┌─────────────────┐    ┌──────────────────────────────────┐
│  MiniMaxClient  │    │        ToolRegistry               │
│  (LLM 推理)     │    │  calculator / get_current_time   │
│  Anthropic SDK  │    │  echo / tts_speak / ...          │
└─────────────────┘    └──────────────────────────────────┘
```

---

## 二、模块说明

### 2.1 `config.py` — 配置管理

使用 `pydantic-settings` 从环境变量或 `.env` 文件加载配置。

```python
from ai_life_agent.config import settings

print(settings.minimax_api_key)   # MINIMAX_API_KEY
print(settings.chat_model)        # MiniMax-M2.7
print(settings.tts_model)         # speech-2.8-hd
```

**配置优先级**（高 → 低）：系统环境变量 → `.env` 文件 → 代码默认值

---

### 2.2 `llm/client.py` — LLM 客户端

通过 **Anthropic SDK** 调用 MiniMax M2.7，只需将 `base_url` 指向 MiniMax 的兼容端点：

```python
client = anthropic.Anthropic(
    api_key=api_key,
    base_url="https://api.minimaxi.com/anthropic",
)
```

**为什么用 Anthropic SDK？**

MiniMax 完整支持 Anthropic API 格式，包括：
- 工具调用（`tool_use` / `tool_result`）
- 思维链（`thinking` block）
- 流式输出（`stream=True`）

这让我们无需维护自定义 HTTP 客户端，直接复用成熟的 SDK。

**多轮工具调用注意事项：**

MiniMax 要求将完整的 `response.content`（含 thinking + tool_use 所有 block）
回传到下一轮请求，以保持思维链的连续性：

```python
# ✅ 正确：回传完整 content 列表
messages.append({"role": "assistant", "content": response.raw_content})

# ❌ 错误：只回传文本，丢失 thinking block
messages.append({"role": "assistant", "content": response.text})
```

---

### 2.3 `tools/registry.py` — 工具注册表

工具注册的核心抽象：**工具 = 带元数据的 Python 函数**。

```
ToolRegistry
├── register(decorator)      # 装饰器注册
├── register_tool(explicit)  # 显式注册
├── execute(name, params)    # 执行工具
├── get_schemas()            # 生成 Anthropic Schema
├── enable/disable(name)     # 动态启用/禁用
└── list_tools()             # 列出所有工具
```

**Schema 生成**：`get_schemas()` 自动将 `ToolDefinition` 转换为 Anthropic 格式，
可直接传入 `MiniMaxClient.chat(tools=...)` 参数：

```json
{
    "name": "calculator",
    "description": "计算数学表达式",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "..."}
        },
        "required": ["expression"]
    }
}
```

---

### 2.4 `core/react.py` — ReAct 推理循环

实现 **Reasoning + Acting** 循环，详见 [REACT.md](./REACT.md)。

---

### 2.5 `core/agent.py` — Agent 主类

统一入口，整合所有模块。Agent 支持两种运行模式：

| 模式 | 条件 | 行为 |
|-----|------|-----|
| 完整模式 | MINIMAX_API_KEY 已配置 | 使用 ReAct + 工具调用 |
| 降级模式 | 无 API Key | 返回提示信息，功能可测试 |

---

## 三、数据流

```
用户输入: "计算 sqrt(144)"
    │
    ▼
Agent.run("计算 sqrt(144)")
    │
    ▼
ReActExecutor.run(user_input, history)
    │
    ├─── [迭代 1] ──────────────────────────────────
    │    LLM 调用: messages=[{role:user, content:...}]
    │    LLM 响应: tool_use(calculator, {expression:"sqrt(144)"})
    │    工具执行: calculator("sqrt(144)") → "12"
    │    更新 messages: += [assistant(tool_use), user(tool_result:"12")]
    │
    ├─── [迭代 2] ──────────────────────────────────
    │    LLM 调用: messages（含工具结果）
    │    LLM 响应: text("sqrt(144) = 12")
    │    停止原因: end_turn
    │
    ▼
ReActResult(answer="sqrt(144) = 12", steps=[...])
    │
    ▼
Agent.messages += [user(...), assistant("sqrt(144) = 12")]
    │
    ▼
返回 "sqrt(144) = 12"
```

---

## 四、扩展指南

### 添加新工具

```python
from ai_life_agent.tools.registry import ToolRegistry, ToolParameter

registry = ToolRegistry()

@registry.register(description="查询城市天气")
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的实时天气。
    
    Args:
        city: 城市名称，例如 "北京"、"上海"
        unit: 温度单位，"celsius" 或 "fahrenheit"
    """
    # 调用天气 API
    return f"{city}: 晴，25°C"
```

### 自定义系统提示词

```python
from ai_life_agent.core.agent import Agent

agent = Agent(
    system_prompt="你是一个专业的代码助手，擅长 Python 编程。请用代码示例回答问题。"
)
```

### 关闭工具系统

```python
agent = Agent(enable_tools=False)
# Agent 将直接对话，不调用任何工具
```

---

## 五、目录结构

```
src/ai_life_agent/
├── config.py              # 配置管理（pydantic-settings）
├── __init__.py            # 版本号
├── cli.py                 # 命令行入口
├── core/
│   ├── agent.py           # Agent 主类（统一入口）
│   ├── planner.py         # 任务规划（v0.4 扩展）
│   ├── executor.py        # 工具执行（低层，被 ReAct 使用）
│   └── react.py           # ReAct 推理循环（v0.3）
├── llm/
│   ├── __init__.py
│   └── client.py          # MiniMax LLM 客户端
├── tools/
│   ├── __init__.py
│   ├── registry.py        # 工具注册表
│   ├── builtin.py         # 内置工具（calculator/time/echo）
│   ├── tts.py             # TTS 工具
│   ├── asr.py             # ASR 工具（待实现）
│   └── vision.py          # 视觉工具（待实现）
└── memory/
    └── __init__.py        # 记忆系统（v0.4 完整实现）
```
