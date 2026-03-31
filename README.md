# ai-life-agent

Multi-modal Agent powered by MiniMax M2.7

## Features

- 🤖 **LLM Integration** — MiniMax M2.7 for reasoning and planning
- 🔊 **TTS** — MiniMax TTS API for speech synthesis
- 👤 **Face Recognition** — Local/OpenCV + Cloud face API support
- 🎙️ **Voice Input** — Speech to text via MiniMax ASR
- 🛠️ **Tool System** — Extensible tool registry
- 💾 **Memory** — Conversation history and fact accumulation

## Architecture

```
User Input
    ↓
Planning Agent (M2.7)
    ↓
Tool Executor
    ↓
Memory Layer
    ↓
Response (Text / TTS)
```

## Quick Start

```bash
# Clone
git clone https://github.com/xuqi1987/ai-life-agent.git
cd ai-life-agent

# Install
uv sync

# Configure
cp .env.example .env
# Edit .env with your MiniMax API key

# Run
uv run ai-life-agent
```

## Project Structure

```
ai_life_agent/
├── core/           # Core agent logic
│   ├── agent.py    # Main agent loop
│   ├── planner.py  # Task planning
│   └── executor.py # Tool execution
├── tools/          # Tool implementations
│   ├── tts.py      # Text-to-speech
│   ├── asr.py      # Speech-to-text
│   └── vision.py   # Face recognition
├── memory/         # Memory management
└── cli.py          # CLI entry point
```

## Development

```bash
uv sync --extra dev     # Install dev dependencies
uv run pytest            # Run tests
uv run ruff check .      # Lint
uv run mypy src/         # Type check
```

## License

Apache-2.0
