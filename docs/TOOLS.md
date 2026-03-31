# 工具系统设计文档

> 本文档介绍 AI Life Agent 的工具系统架构、使用方式和扩展方法。

---

## 一、工具系统概述

工具系统基于"**工具即函数**"的设计理念：

- 每个工具对应一个 Python 函数
- 工具元数据（名称、描述、参数）通过装饰器声明
- 工具可在运行时动态注册/注销
- 工具 Schema 自动转换为 LLM（Anthropic）格式

```
Python 函数
    │
    │ @registry.register(description="...")
    ▼
ToolDefinition（含 to_anthropic_schema()）
    │
    │ get_schemas()
    ▼
[{name, description, input_schema}, ...]
    │
    │ chat(tools=schemas)
    ▼
MiniMax LLM（决定调用哪个工具）
```

---

## 二、注册工具

### 方式一：装饰器注册（推荐）

```python
from ai_life_agent.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.register(description="将摄氏度转换为华氏度")
def celsius_to_fahrenheit(celsius: float) -> float:
    """温度单位转换。
    
    Args:
        celsius: 摄氏度数值
    """
    return celsius * 9 / 5 + 32

# 测试
result = registry.execute("celsius_to_fahrenheit", {"celsius": 100})
# => 212.0
```

### 方式二：显式注册

```python
from ai_life_agent.tools.registry import ToolRegistry, ToolParameter

registry = ToolRegistry()
registry.register_tool(
    name="get_weather",
    description="查询指定城市的实时天气",
    parameters=[
        ToolParameter("city", "城市名称，例如 '北京'", "string", required=True),
        ToolParameter("unit", "温度单位", "string", required=False, enum=["celsius", "fahrenheit"]),
    ],
    func=lambda city, unit="celsius": f"{city}: 晴，25°C",
)
```

---

## 三、内置工具

注册方式：`register_builtin_tools(registry)`

### calculator — 数学计算器

```python
registry.execute("calculator", {"expression": "sqrt(16)"})  # => "4"
registry.execute("calculator", {"expression": "2 ** 10"})   # => "1024"
registry.execute("calculator", {"expression": "sin(pi/2)"}) # => "1.0"
```

支持的运算：`+ - * / ** %`
支持的函数：`sqrt, sin, cos, tan, log, log2, log10, exp, abs, round, ceil, floor, factorial, pow`
支持的常量：`pi, e`

**安全机制**：禁止双下划线（`__import__`、`__builtins__` 等），防止代码注入。

### get_current_time — 获取当前时间

```python
registry.execute("get_current_time", {})
# => "2026-03-31 22:23:52"

registry.execute("get_current_time", {"format": "%Y/%m/%d"})
# => "2026/03/31"
```

### echo — 原样返回（测试用）

```python
registry.execute("echo", {"text": "Hello World"})
# => "Hello World"
```

---

## 四、动态管理工具

```python
# 禁用工具（LLM 不可见）
registry.disable("echo")

# 启用工具
registry.enable("echo")

# 注销工具
registry.unregister("echo")

# 检查工具是否存在
"calculator" in registry  # True

# 列出所有工具
registry.list_tools()  # ["calculator", "get_current_time", "echo"]

# 获取 Anthropic Schema（传给 LLM）
schemas = registry.get_schemas()
```

---

## 五、参数类型映射

Python 类型注解会自动映射到 JSON Schema 类型：

| Python 类型 | JSON Schema 类型 |
|------------|-----------------|
| `str`      | `string`        |
| `int`      | `number`        |
| `float`    | `number`        |
| `bool`     | `boolean`       |
| `list`     | `array`         |
| `dict`     | `object`        |

有默认值的参数自动标记为**非必填**（不出现在 `required` 列表中）。

---

## 六、TTS 工具

```python
from ai_life_agent.tools.tts import TTS

tts = TTS()

# 合成为字节流
audio_bytes = tts.speak("你好，我是 AI 助手！")

# 直接保存到文件
tts.speak_to_file("你好！", "output.mp3", voice_id="male-qn-qingse")
```

**可用音色（部分）**：

| voice_id | 描述 |
|---------|------|
| `male-qn-qingse` | 青涩青年男声 |
| `female-shaonv` | 少女音 |
| `female-yujie` | 御姐音 |
| `female-tianmei` | 甜美女声 |
| `presenter_male` | 男性播音员 |

**支持模型**（根据订阅方案）：
- `speech-2.8-hd` — 新一代 HD，精准还原真实语气（Plus 方案支持）

---

## 七、工具最佳实践

1. **描述要清晰**：工具描述决定了 LLM 何时调用，描述越准确，调用越精准
2. **参数越少越好**：参数多会增加 LLM 的理解负担，优先使用有意义的默认值
3. **返回字符串**：工具应返回字符串（或可转为字符串的结果），便于 LLM 理解
4. **错误信息友好**：工具出错时返回描述性错误信息，而非抛出异常（让 LLM 决策）
5. **单一职责**：每个工具只做一件事，复杂操作拆分为多个工具

```python
# ✅ 好的工具设计
@registry.register(description="将文本翻译为指定语言")
def translate(text: str, target_lang: str = "英语") -> str:
    """翻译文本。"""
    ...

# ❌ 不好：职责过多
@registry.register(description="翻译、摘要并发邮件")
def translate_summarize_email(text: str, email: str) -> str:
    ...  # 太复杂，应拆分
```
