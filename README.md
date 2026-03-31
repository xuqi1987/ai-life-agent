# AI Life Agent

> 从零学习 Agent 开发 —— 用 MiniMax M2.7 驱动的完整 AI 智能体框架

**适合人群**：完全不懂 Agent 开发的初学者，也适合想快速搭建生产级 Agent 的开发者。

---

## ✨ 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 🤖 **LLM 对话** | ✅ 已实现 | MiniMax M2.7，支持多轮对话 |
| 🛠️ **工具系统** | ✅ 已实现 | 可扩展的工具注册表，支持装饰器注册 |
| 🔄 **ReAct 循环** | ✅ 已实现 | 自主推理 → 工具调用 → 观察 → 答案 |
| 🔊 **语音合成** | ✅ 已实现 | MiniMax TTS，多音色/语速可调 |
| 💾 **对话记忆** | ✅ 已实现 | 跨轮次记忆，支持笔记保存 |
| 🎙️ **语音输入** | 🔜 规划中 | MiniMax ASR |
| 👤 **人脸识别** | 🔜 规划中 | OpenCV + 云 API |

---

## 🚀 快速上手（5 分钟）

### 1. 安装

```bash
git clone https://github.com/xuqi1987/ai-life-agent.git
cd ai-life-agent
uv sync    # 需要 Python 3.11+
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 MiniMax API Key
# 获取地址：https://platform.minimaxi.com/user-center/basic-information/interface-key
```

### 3. 运行演示

```bash
uv run python demo.py    # 一键体验 6 步演示
```

### 4. 交互式对话

```bash
uv run ai-life-agent     # 启动命令行 Agent
```

详细指引见 **[docs/QUICK_START.md](docs/QUICK_START.md)**

---

## 📚 学习路径

本项目提供 **6 步循序渐进的教程**，从零掌握 Agent 开发：

```
examples/
├── step1_hello_llm.py       ← 【第1步】和大模型对话（10行代码）
├── step2_tools.py           ← 【第2步】给 AI 使用工具
├── step3_react.py           ← 【第3步】ReAct 自主推理循环
├── step4_custom_tool.py     ← 【第4步】开发自己的工具
├── step5_tts.py             ← 【第5步】文字转语音
└── step6_full_agent.py      ← 【第6步】完整多轮 Agent
```

**逐步学习**：

```bash
uv run python examples/step1_hello_llm.py    # 从第 1 步开始
uv run python examples/step2_tools.py
# ... 依次运行
```

完整教程：**[docs/TUTORIAL.md](docs/TUTORIAL.md)**

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────┐
│              你的问题（自然语言）              │
└─────────────────────┬───────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│           ReAct 推理循环（core/react.py）     │
│   思考 → 工具调用 → 观察 → 思考 → ... → 答案  │
└──────────┬──────────────────────┬───────────┘
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────┐
│  MiniMax M2.7    │   │   工具注册表           │
│  LLM 推理引擎    │   │ calculator/time/tts  │
│  (llm/client.py) │   │ notes/search/...     │
└──────────────────┘   └──────────────────────┘
```

```
src/ai_life_agent/
├── config.py           ← 配置管理（API Key、模型选择）
├── llm/client.py       ← MiniMax API 客户端
├── core/
│   ├── agent.py        ← Agent 主类（高层接口）
│   └── react.py        ← ReAct 推理循环（核心）
├── tools/
│   ├── registry.py     ← 工具注册表
│   ├── builtin.py      ← 内置工具（计算器、时间）
│   └── tts.py          ← 语音合成
└── memory/             ← 对话记忆管理
```

---

## 🔧 开发命令

```bash
uv run pytest -q          # 运行测试（当前 58 个测试）
uv run ruff check .        # 代码检查
uv run ruff check . --fix  # 自动修复
```

---

## 📖 文档

| 文档 | 说明 |
|------|------|
| [QUICK_START.md](docs/QUICK_START.md) | 5 分钟快速上手 |
| [TUTORIAL.md](docs/TUTORIAL.md) | 完整初学者教程 |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构设计说明 |
| [REACT.md](docs/REACT.md) | ReAct 原理详解 |
| [TOOLS.md](docs/TOOLS.md) | 工具系统文档 |
| [SPEC.md](SPEC.md) | 完整功能规格 |

---

## 版本历史

| 版本 | 内容 |
|------|------|
| v0.4 | 完整初学者教程 + 6 步示例脚本 + 一键演示菜单 |
| v0.3 | ReAct 推理循环 + 完整测试套件 |
| v0.2 | MiniMax API 接入 + 工具系统 + TTS |
| v0.1 | 项目骨架（规划中） |

---

## License

Apache-2.0
