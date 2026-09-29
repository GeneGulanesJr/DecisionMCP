# LayaMCP

HTTP MCP server that wraps the [Laya](https://github.com/NandhaKishorM/laya) decision engine for use with [Pi](https://pi.dev) (and any other MCP client).

## What it does

Exposes 13 tools to your agent — 5 upstream-preset, 6 coding-specific (custom question schemas) and 2 maintenance tools:

| Tool                  | Purpose                                              | Latency  | Source     |
| --------------------- | ---------------------------------------------------- | -------- | ---------- |
| `laya_guard`          | Prompt-injection / jailbreak detection               | ~33 ms   | preset     |
| `laya_route`          | Decide "small model" vs "frontier model" per prompt  | ~33 ms   | preset     |
| `laya_moderate`       | Toxicity / harassment / threats                      | ~33 ms   | preset     |
| `laya_triage`         | Intent / urgency / churn / refund / frustration (support) | ~33 ms   | preset     |
| `laya_email`          | Email team / urgency / needs-reply / spam / phishing | ~33 ms   | preset     |
| `laya_review_tone`    | Code review tone + priority                           | ~33 ms   | custom     |
| `laya_bug_severity`   | Bug severity + area                                  | ~33 ms   | custom     |
| `laya_commit_classify`| Commit message type / scope / risk                    | ~33 ms   | custom     |
| `laya_test_priority`  | Test run priority + reason                            | ~33 ms   | custom     |
| `laya_secret_risk`    | Detect leaked credentials in text                    | ~33 ms   | custom     |
| `laya_diff_intent`    | PR diff intent / scope / risk                         | ~33 ms   | custom     |
| `laya_update`         | Check for a newer Laya release / model weights / new models (opt-in `apply`) | —        | admin      |
| `laya_usage`          | Usage + session stats, JSONL export for training      | —        | admin      |

All inference tools run in-process via Laya's encoder checkpoints (ModernBERT-large / mmBERT-base). One HTTP endpoint, single model load per process.

## Install

```bash
cd LayaMCP
uv venv .venv && source .venv/bin/activate     # or python -m venv
uv pip install -e ".[dev]"                     # or pip install -e ".[dev]"
```

### Models

Models live **inside the project** in `./models` (~2.2 GB, gitignored): the three Laya checkpoints (`english`, `multilingual`, `typed-decisions`) from [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya). LayaMCP points the Hugging Face cache there automatically (an explicit `HF_HUB_CACHE` / `HF_HOME` still wins). To download them once:

```bash
HF_HUB_CACHE="$PWD/models" python -c "from laya import Router; Router().preload()"
```

Keep them current with the `laya_update` tool (see [docs/TOOLS.md](docs/TOOLS.md#laya_update)).

## Run

```bash
# Default: 127.0.0.1:8765
layamcp

# Or override via env:
LAYAMCP_PORT=9000 layamcp
```

The MCP endpoint uses the SSE transport at `http://127.0.0.1:8765/sse` (`GET /health` for a health check).

Every prediction is logged to `./data/usage.db` with its MCP session, for training data. Input text is **not** stored unless you set `LAYAMCP_USAGE_STORE_TEXT=true` (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)).

## Configure Pi

Pi (v0.87.x) has no built-in MCP client; it attaches MCP servers through the
[`pi-mcp-extension`](https://www.npmjs.com/package/pi-mcp-extension) bridge:

```bash
pi install npm:pi-mcp-extension
```

Then register this server in `~/.pi/agent/mcp.json` (global) or `.pi/mcp.json` (per-project):

```json
{
  "settings": {
    "toolPrefix": "mcp",
    "requestTimeoutMs": 30000,
    "maxRetries": 5
  },
  "mcpServers": {
    "layamcp": {
      "transport": "sse",
      "url": "http://127.0.0.1:8765/sse",
      "lifecycle": "eager",
      "healthCheckIntervalMs": 60000
    }
  }
}
```

Start the server first (`layamcp`), then start Pi — all 13 tools appear as
`mcp_layamcp_<tool>` (e.g. `mcp_layamcp_laya_guard`, argument is `prompt`).
Use `/mcp` inside Pi to check connection status; with `lifecycle: "eager"` Pi
connects at session start (5 retries), with `"lazy"` you start it manually via
`/mcp:start`. The server speaks MCP over SSE at `/sse` (`GET /health` for a
health check).

## Documentation

- **[AGENTS.md](AGENTS.md)** — instructions for AI coding agents working in this repo
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — layer diagram, data flow, performance
- **[docs/TOOLS.md](docs/TOOLS.md)** — per-tool behaviour, when to use, limitations
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — every env var, .env format, prod checklist
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — adding tools, code style, release checklist
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — systemd, Docker, nginx reverse proxy
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — common errors and fixes

## Adding a new tool

The architecture is plugin-style. To add a tool:

**1. Create `laya_mcp/tools/mytool.py`:**

```python
from pydantic import BaseModel, Field

from .base import Tool


class MyInput(BaseModel):
    text: str = Field(..., description="Input text.")


class MyOutput(BaseModel):
    label: str
    confidence: float


class MyTool(Tool):
    name = "laya_mytool"
    description = "One-line description shown to the agent."
    input_schema = MyInput
    output_schema = MyOutput

    async def run(self, bridge, text: str) -> MyOutput:
        raw = bridge.predict(text, preset="...")  # or build your own questions
        return MyOutput(label=..., confidence=...)
```

**2. Register it in `laya_mcp/tools/__init__.py`:**

```python
from .mytool import MyTool

TOOLS: list[Tool] = [
    GuardTool(),
    RouteTool(),
    ModerateTool(),
    TriageTool(),
    EmailTool(),
    MyTool(),  # ← new
]
```

That's it. The MCP wire schema, FastAPI route, dispatch table — all derive from the class. No other files need to change.

### Adding a custom preset (not from upstream)

If you want a tool that uses questions you defined yourself (not from `laya.presets`):

```python
from ._helpers import choice_of, require_dict

MY_QUESTIONS = {
    "answer": {
        "type": "noul",  # yes/no; answer is P(yes)
        "instructions": "Is the `text` a question?",
    },
    "priority": {
        "type": "choice",
        "instructions": "How urgent is the `text`?",
        "criteria": {"low": "can wait", "medium": "this week", "high": "now"},
    },
}

class MyTool(Tool):
    ...
    async def run(self, bridge, text: str) -> MyOutput:
        # "text" is the field name your instructions refer to (`text` above).
        raw = bridge.predict_custom(text, questions=MY_QUESTIONS, state_key="text")
        require_dict(raw, self.name)
        priority, confidence = choice_of(raw, "priority", self.name)
        ...
```

`LayaBridge.predict_custom()` already exists in `bridge.py`.

## Tests

```bash
pytest                                              # mocked, no models needed, fast
LAYAMCP_INTEGRATION=1 pytest tests/test_integration.py   # real models (needs ./models)
```

## License

Apache-2.0