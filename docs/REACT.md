# ReAct 推理循环 — 原理与实现

> ReAct = **Re**asoning + **Act**ing：让 LLM 交替进行推理和行动的范式。

原论文：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

---

## 一、什么是 ReAct？

传统 LLM 直接从输入生成输出，对于需要**多步骤推理**或**实时信息**的问题力不从心。
ReAct 通过让 LLM 交替输出**思维**（Thought）和**行动**（Action），并将**行动结果**（Observation）
反馈给模型，形成迭代循环，直到给出最终答案。

```
用户输入
    │
    ▼
┌──────────────────────────────────────┐
│           ReAct 循环                  │
│                                      │
│  ┌──────────┐                        │
│  │ Thought  │  LLM 思考：我该做什么？  │
│  └────┬─────┘                        │
│       │                              │
│       ▼                              │
│  ┌──────────┐                        │
│  │  Action  │  决定调用哪个工具？       │
│  └────┬─────┘                        │
│       │                              │
│       ▼                              │
│  ┌──────────────┐                    │
│  │ Observation  │  执行工具，获取结果  │
│  └──────┬───────┘                    │
│         │                            │
│         └── 结果喂回 LLM ──► 继续    │
│                                      │
│  直到：stop_reason = "end_turn"       │
└──────────────────────────────────────┘
    │
    ▼
最终答案（Answer）
```

---

## 二、本项目的实现

### 2.1 核心实现文件

`src/ai_life_agent/core/react.py` → `ReActExecutor`

### 2.2 消息格式（Anthropic SDK）

本项目使用 Anthropic SDK 格式实现工具调用，每轮迭代的消息结构：

```
[迭代 1 - 工具调用]

用户消息:
  {"role": "user", "content": "计算 sqrt(144)"}

模型响应（tool_use）:
  {
    "role": "assistant",
    "content": [
      {"type": "thinking", "thinking": "我需要用 calculator 工具..."},
      {"type": "tool_use", "id": "t1", "name": "calculator",
       "input": {"expression": "sqrt(144)"}}
    ]
  }

工具执行结果（tool_result）:
  {
    "role": "user",
    "content": [
      {"type": "tool_result", "tool_use_id": "t1", "content": "12"}
    ]
  }

[迭代 2 - 最终答案]

模型响应（end_turn）:
  {"role": "assistant", "content": [{"type": "text", "text": "sqrt(144) = 12"}]}
```

### 2.3 关键设计决策

#### 决策 1：完整回传 assistant 消息

MiniMax 要求在多轮工具调用中，必须将完整的 assistant 消息（包含 thinking block）
回传给模型，以保持思维链的连续性：

```python
# ✅ 正确：回传 raw_content（包含 thinking + tool_use 所有 block）
messages.append({"role": "assistant", "content": response.raw_content})
```

如果只回传文本部分，模型会失去上下文，影响推理质量。

#### 决策 2：工具执行错误不崩溃

工具执行失败时，将错误信息作为 Observation 返回，让模型自行决策（如重试或换策略）：

```python
try:
    observation_str = self.registry.execute(tool_name, tool_input)
except Exception as e:
    observation_str = f"工具执行失败: {e}"
    # 不 raise，继续循环
```

#### 决策 3：最大迭代次数防护

设置 `max_iterations`（默认 10）防止工具循环调用导致无限循环：

```python
for iteration in range(1, self.max_iterations + 1):
    ...
else:
    # 超出限制，强制停止
    result.stopped_by_limit = True
```

---

## 三、一个完整的执行示例

**用户输入**：「帮我计算 sqrt(144)，并告诉我现在的时间」

```
[迭代 1]
  思维链: 用户想要计算 sqrt(144) 并获取当前时间。
          这两个操作是独立的，可以同时调用。

  调用工具: calculator({"expression": "sqrt(144)"})
  工具结果: 12

  调用工具: get_current_time({})
  工具结果: 2026-03-31 22:23:52

[迭代 2]
  思维链: 两个结果都已返回，可以给出最终答案了。

  最终回答: 计算结果和当前时间如下：
    1. sqrt(144) = 12
    2. 当前时间：2026-03-31 22:23:52
```

---

## 四、ReAct vs 简单问答的对比

| 场景 | 简单 LLM | ReAct |
|-----|---------|-------|
| "北京今天天气" | 知识截止，无法回答 | 调用天气 API 获取实时数据 |
| "2^100 + 3^50 = ?" | 可能算错（大数运算） | 调用计算器精确计算 |
| "帮我订明天的会议" | 无法操作外部系统 | 调用日历 API 创建事件 |
| "你好" | 直接回答 | 无需工具，直接回答（0 次工具调用） |

---

## 五、如何扩展 ReAct

### 添加新工具

```python
from ai_life_agent.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.register(description="搜索网页")
def web_search(query: str) -> str:
    """使用搜索引擎搜索信息。
    
    Args:
        query: 搜索关键词
    """
    # 调用搜索 API
    return f"搜索结果: ..."

# 创建带新工具的 Executor
from ai_life_agent.core.react import ReActExecutor
executor = ReActExecutor(registry=registry)
```

### 自定义 System Prompt

```python
executor = ReActExecutor(
    registry=registry,
    system_prompt="你是专业的旅游顾问，擅长为用户推荐旅游目的地和行程规划。",
)
```

### 查看完整推理链路

```python
result = executor.run("我要去日本旅游，有什么推荐？")
print(result.format_trace())  # 输出完整的 Thought/Action/Observation 记录
```

---

## 六、学习检验

完成 v0.3 后，你应该能回答以下问题：

1. ReAct 中的 "Thought" 和 "Action" 分别由谁决定？
   > Thought 由 LLM 自主生成（通过 thinking block）；Action 是 LLM 输出的工具调用（tool_use block）；Observation 是工具执行的实际结果。

2. 为什么要将完整的 assistant 消息（含 thinking）回传给模型？
   > 保持思维链的连续性。MiniMax M2.7 的推理能力依赖于上下文中的 thinking 内容，如果丢失会影响后续推理质量。

3. ReAct 循环的终止条件是什么？
   > `stop_reason == "end_turn"` 且无工具调用请求，或达到最大迭代次数。

4. 工具调用失败时 ReAct 如何处理？
   > 将错误信息作为 Observation 返回给模型，让模型决定是否重试或改变策略，不强制中断循环。
