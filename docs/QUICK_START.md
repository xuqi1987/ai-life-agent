# 5 分钟快速上手

> 从安装到运行第一个 AI Agent，只需 5 分钟。

---

## 第 1 步：安装（1 分钟）

```bash
# 需要 Python 3.11+，确认版本
python --version

# 克隆项目
git clone https://github.com/xuqi1987/ai-life-agent.git
cd ai-life-agent

# 安装依赖（使用 uv）
pip install uv        # 如果没有 uv，先安装
uv sync               # 安装所有依赖
```

---

## 第 2 步：配置 API Key（1 分钟）

```bash
# 复制配置文件
cp .env.example .env
```

用任意编辑器打开 `.env`，填入你的 API Key：

```
MINIMAX_API_KEY=sk-你的密钥在这里
```

> **获取 API Key**：访问 [MiniMax 控制台](https://platform.minimaxi.com/user-center/basic-information/interface-key)

---

## 第 3 步：运行演示（1 分钟）

```bash
# 启动一键演示菜单
uv run python demo.py
```

选择 `[1]` 看第一个演示（和大模型对话），选 `[q]` 退出。

---

## 第 4 步：交互式对话（2 分钟）

```bash
# 启动命令行聊天
uv run ai-life-agent
```

直接输入问题，试试：
- `"现在几点了？"`
- `"帮我算一下 123 * 456"`
- `"给我讲个笑话"`

输入 `exit` 退出。

---

## 常用命令速查

| 命令 | 说明 |
|------|------|
| `uv run python demo.py` | 一键演示菜单 |
| `uv run python examples/step1_hello_llm.py` | 第 1 步：对话 |
| `uv run python examples/step2_tools.py` | 第 2 步：工具 |
| `uv run python examples/step3_react.py` | 第 3 步：ReAct |
| `uv run python examples/step4_custom_tool.py` | 第 4 步：自定义工具 |
| `uv run python examples/step5_tts.py` | 第 5 步：语音 |
| `uv run python examples/step6_full_agent.py` | 第 6 步：完整 Agent |
| `uv run ai-life-agent` | 交互式对话 |
| `uv run pytest -q` | 运行测试 |

---

## 遇到问题？

- 📖 完整教程：[docs/TUTORIAL.md](TUTORIAL.md)
- 🏗️ 架构说明：[docs/ARCHITECTURE.md](ARCHITECTURE.md)
- ❓ 常见问题：见 [TUTORIAL.md 第十节](TUTORIAL.md#十常见问题)
