# AI Life Agent — 需求规格说明书

> 本项目作为 Agent 开发的学习工程，通过渐进式迭代掌握 Agent 的核心能力与设计模式。

---

## 一、项目愿景

**目标：** 从零构建一个多模态智能 Agent，理解并掌握现代 LLM Agent 的核心原理与工程实践。

**学习路径：**
1. 单 Agent 基本架构
2. 工具调用系统（Tool Calling）
3. 记忆系统（Memory）
4. 规划能力（Planning / ReAct）
5. 多 Agent 协作
6. Skill / Plugin 机制
7. 生产级 Agent 架构

---

## 二、技术选型

| 组件 | 技术 | 说明 |
|-----|------|-----|
| 核心模型 | MiniMax M2.7 / OpenAI GPT-4o | LLM 推理引擎 |
| 编程语言 | Python 3.10+ | 主力开发语言 |
| 包管理 | uv | 现代化 Python 包管理 |
| 项目规范 | ruff / mypy / pytest | 代码质量工具链 |
| 语音合成 | MiniMax TTS API | 多音色 TTS |
| 语音识别 | MiniMax ASR API | 语音转文字 |
| 人脸识别 | OpenCV + face_recognition | 本地离线方案 |
| HTTP 客户端 | httpx / requests | API 调用 |
| 配置管理 | pydantic-settings / .env | 类型安全配置 |

---

## 三、Agent 核心能力模型

### 3.1 单 Agent 架构

```
┌─────────────────────────────────────────────────────────┐
│                        USER                             │
└─────────────────────┬───────────────────────────────────┘
                      │ 自然语言 / 语音输入
                      ▼
┌─────────────────────────────────────────────────────────┐
│                     INPUT LAYER                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐        │
│  │  Text    │  │  Audio   │  │   Image      │        │
│  │  Parser  │  │  ASR     │  │   Vision     │        │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘        │
└───────┼─────────────┼───────────────┼──────────────────┘
        │             │               │
        ▼             ▼               ▼
┌─────────────────────────────────────────────────────────┐
│                   COGNITION LAYER                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │              PLANNER (ReAct Loop)                 │   │
│  │                                                   │   │
│  │   Thought → Action → Observation → Answer       │   │
│  │                                                   │   │
│  │   ┌─────────────┐  ┌─────────────────────┐     │   │
│  │   │  Intention   │  │   Task             │     │   │
│  │   │  Recognition │  │   Decomposition    │     │   │
│  │   └─────────────┘  └─────────────────────┘     │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    TOOL LAYER                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  TTS     │  │  Search  │  │  Custom  │             │
│  │  Tool    │  │  Tool    │  │  Tools   │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   MEMORY LAYER                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐     │
│  │ Short-term │  │ Long-term  │  │  Vector      │     │
│  │ (Context)  │  │ (Facts)    │  │  Search      │     │
│  └────────────┘  └────────────┘  └──────────────┘     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  Text   │  │  TTS     │  │  Action  │             │
│  │  Reply  │  │  Audio   │  │  Execute │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心能力模块

#### 能力 1：ReAct 推理循环（必学基础）

**原理：** Thought → Action → Observation → Answer 循环，让模型自己决定下一步行动。

```python
# 伪代码
while not done:
    thought = llm.think(messages + context)   # 模型思考：我应该做什么？
    if thought.is_answer:
        return thought.answer                  # 直接回答
    result = tool.execute(thought.action)      # 调用工具
    messages.append(observation: result)       # 把结果喂回去
```

**学习目标：** 理解 Agent 的"自我决策"能力，而非固定流程。

#### 能力 2：工具调用系统（Tool Calling）

**原理：** LLM 输出结构化 JSON，指定调用的工具和参数，Agent 执行后返回结果。

```
LLM 输出格式：
{
  "tool": "tts_speak",
  "args": {"text": "你好", "voice": "female_tianmei"}
}
```

**工具注册机制：**
```python
@tool_registry.register(name="tts_speak", description="将文字转为语音")
def tts_speak(text: str, voice: str = "default") -> AudioBytes:
    ...
```

**学习目标：** 掌握"工具即能力"的设计思想，工具可插拔。

#### 能力 3：记忆系统

**三层记忆架构：**

| 层级 | 容量 | 持久化 | 用途 |
|-----|------|--------|-----|
| Working Memory | 单次对话 | ❌ | 当前任务上下文 |
| Short-term | 最近 N 条 | ✅ (SQLite) | 最近对话历史 |
| Long-term | 全部 | ✅ (SQLite) | 事实、偏好、技能 |

**记忆写入时机：**
- 每轮对话结束自动摘要
- 检测到重要事实时立即存储
- 用户明确告知的信息立即存储

**学习目标：** 理解 Agent 如何积累经验、实现个性化。

#### 能力 4：任务规划（Task Planning）

**单任务分解：**
```python
# 用户："帮我查上海天气，然后告诉要不要带伞"
# 拆解为：
#   Step 1: call weather_api(city="上海")
#   Step 2: decide umbrella(weather_data)
#   Step 3: speak_to_user(decision)
```

**树状规划（复杂任务）：**
```
Root Task: 筹备生日派对
├── Task 1: 确定日期和人数
├── Task 2: 预订场地
├── Task 3: 邀请宾客
│   ├── Subtask: 发邀请消息
│   └── Subtask: 收集回复
└── Task 4: 订蛋糕
```

**学习目标：** 理解 Agent 如何处理复杂、多步骤任务。

#### 能力 5：多 Agent 协作

**两种协作模式：**

**模式 A：Supervisor（主管-执行者）**
```
用户 → Supervisor Agent
            ├── 拆解任务
            ├── 分发给专业 Agent
            │   ├── Web Search Agent
            │   ├── Code Agent
            │   └── TTS Agent
            └── 整合结果返回
```

**模式 B：Collaborative（对等协作）**
```
Agent A ←→ Agent B
  ↓              ↓
  共享 Memory / 状态
```

**通信协议：**
```python
@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str | None      # None = broadcast
    content: str
    msg_type: MessageType     # REQUEST / RESPONSE / OBSERVATION
    reply_to: str | None      # 关联消息 ID
```

**学习目标：** 理解多 Agent 如何分工、协作、传递信息。

#### 能力 6：Skill 机制（技能插件）

**Skill 定义：**
```python
@dataclass
class Skill:
    name: str
    description: str
    parameters: list[Parameter]
    handler: Callable
    enabled: bool = True
```

**Skill 生命周期：**
```
注册 → 发现 → 匹配 → 调用 → 结果返回
```

**Skill 示例（预约技能）：**
```python
@skill(name="schedule_meeting",
       description="安排会议",
       params=[p("title"), p("time"), p("attendees")])
def schedule_meeting(title: str, time: str, attendees: list[str]):
    # 调用日历 API
    # 发送邀请
    return {"meeting_id": "xxx", "status": "created"}
```

**学习目标：** 掌握可扩展的 Skill 注册与匹配机制。

---

## 四、功能需求列表

### 4.1 第一阶段：单 Agent 骨架（v0.1）

- [x] 项目结构搭建（uv + pytest + ruff）
- [x] Agent 核心类（Agent / Planner / Executor）
- [x] Memory 基础实现
- [ ] 单元测试覆盖

### 4.2 第二阶段：工具系统（v0.2）

- [ ] 工具注册表（ToolRegistry）
- [ ] MiniMax Chat API 集成
- [ ] TTS 工具实现
- [ ] ASR 工具实现
- [ ] 通用 Search 工具

### 4.3 第三阶段：ReAct 循环（v0.3）

- [ ] ReAct Executor
- [ ] 工具调用解析（JSON output parsing）
- [ ] 自我反思机制（Self-reflection）
- [ ] 错误重试逻辑

### 4.4 第四阶段：记忆增强（v0.4）

- [ ] SQLite 持久化
- [ ] 记忆自动摘要
- [ ] 重要信息提取（事实/偏好）
- [ ] 向量搜索（可选 later）

### 4.5 第五阶段：多 Agent 协作（v0.5）

- [ ] Agent 通信协议
- [ ] Supervisor Agent
- [ ] 协作任务队列
- [ ] Agent 状态管理

### 4.6 第六阶段：Skill 系统（v0.6）

- [ ] Skill 定义与注册
- [ ] Skill 匹配器（意图识别 → Skill）
- [ ] 内置 Skills（TTS / ASR / Weather / Search）
- [ ] Skill 热加载机制

### 4.7 第七阶段：多模态（v0.7）

- [ ] 人脸识别工具
- [ ] 图像描述工具
- [ ] 语音对话完整链路

---

## 五、学习成果检验

每阶段完成后，应能回答：

| 阶段 | 核心问题 |
|-----|---------|
| v0.1 | Agent 的基本循环是什么？Planner 和 Executor 的区别？ |
| v0.2 | 工具注册和普通函数调用有什么区别？ |
| v0.3 | ReAct 中的"Thought"和"Action"分别由谁决定？ |
| v0.4 | 三层记忆的区别？什么时候该写入长期记忆？ |
| v0.5 | Supervisor 模式和Collaborative 模式各适合什么场景？ |
| v0.6 | Skill 和 Tool 的边界是什么？ |
| v0.7 | 多模态输入如何影响 Agent 的决策？ |

---

## 六、项目规范

### 代码规范
- 所有代码通过 ruff 检查（`uv run ruff check .`）
- 所有新增代码有类型注解
- 所有公共函数有 docstring
- 提交前必须跑过 `uv run pytest`

### Git 规范
- `feat/` — 新功能
- `fix/` — Bug 修复
- `docs/` — 文档更新
- `refactor/` — 重构（不改变功能）
- `test/` — 测试相关

### 目录结构

```
ai-life-agent/
├── src/ai_life_agent/
│   ├── core/               # 核心组件
│   │   ├── agent.py        # Agent 主循环
│   │   ├── planner.py      # 任务规划
│   │   ├── executor.py     # 工具执行
│   │   └── react.py        # ReAct 实现
│   ├── tools/              # 工具实现
│   │   ├── registry.py     # 工具注册表
│   │   ├── tts.py          # TTS
│   │   ├── asr.py          # ASR
│   │   ├── search.py       # 搜索
│   │   └── vision.py       # 人脸/图像
│   ├── memory/             # 记忆系统
│   │   ├── short_term.py   # 短期记忆
│   │   ├── long_term.py    # 长期记忆
│   │   └── summarizer.py   # 记忆摘要
│   ├── skills/             # 技能系统
│   │   ├── skill.py         # Skill 定义
│   │   └── matcher.py       # Skill 匹配
│   ├── multi_agent/        # 多 Agent
│   │   ├── supervisor.py    # 主管 Agent
│   │   ├── messenger.py     # Agent 间通信
│   │   └── registry.py     # Agent 注册
│   ├── llm/                # LLM 接口
│   │   └── client.py       # MiniMax API 客户端
│   └── cli.py              # CLI 入口
├── tests/                  # 测试
│   ├── test_core.py
│   ├── test_tools.py
│   └── test_memory.py
├── docs/                   # 学习文档
│   ├── ARCHITECTURE.md     # 架构文档
│   ├── REACT.md            # ReAct 原理
│   └── MULTI_AGENT.md      # 多 Agent 设计
├── pyproject.toml
└── SPEC.md
```

---

## 七、版本计划

| 版本 | 目标 | 关键交付 |
|-----|------|---------|
| v0.1 | 可运行骨架 | Agent / Planner / Executor + 测试 |
| v0.2 | 工具调用 | MiniMax API + TTS/ASR 工具 |
| v0.3 | ReAct 循环 | 完整 ReAct 实现 |
| v0.4 | 记忆系统 | SQLite + 自动摘要 |
| v0.5 | 多 Agent | Supervisor + 通信协议 |
| v0.6 | Skill 系统 | 可插拔 Skill 机制 |
| v0.7 | 多模态 | 人脸识别 + 语音对话 |

---

## 八、参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Agent 核心架构设计 — 字节最佳实践](https://www.example.com)
- [LangChain Agent 设计](https://python.langchain.com/docs/concepts/agents/)
- [MiniMax API 文档](https://www.minimax.chat/document)
