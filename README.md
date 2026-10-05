# DecisionMCP

Engine-agnostic HTTP MCP server for decision-model inferences, for use with [Pi](https://pi.dev) (and any other MCP client). Ships with the [Laya](https://github.com/NandhaKishorM/laya) decision engine behind a pluggable `DecisionEngine` protocol — other engines plug in without touching tools.

## What it does

Exposes 13 tools to your agent — 5 engine-preset, 6 coding-specific (custom question schemas) and 2 maintenance tools:

| Tool                  | Purpose                                              | Latency  | Source     |
| --------------------- | ---------------------------------------------------- | -------- | ---------- |
| `decision_guard`      | Prompt-injection / jailbreak detection               | ~33 ms   | preset     |
| `decision_route`      | Decide "small model" vs "frontier model" per prompt  | ~33 ms   | preset     |
| `decision_moderate`   | Toxicity / harassment / threats                      | ~33 ms   | preset     |
| `decision_triage`     | Intent / urgency / churn / refund / frustration (support) | ~33 ms   | preset     |
| `decision_email`      | Email team / urgency / needs-reply / spam / phishing | ~33 ms   | preset     |
| `decision_review_tone`| Code review tone + priority                           | ~33 ms   | custom     |
| `decision_bug_severity`| Bug severity + area                                  | ~33 ms   | custom     |
| `decision_commit_classify`| Commit message type / scope / risk                    | ~33 ms   | custom     |
| `decision_test_priority`| Test run priority + reason                            | ~33 ms   | custom     |
| `decision_secret_risk`| Detect leaked credentials in text                    | ~33 ms   | custom     |
| `decision_diff_intent`| PR diff intent / scope / risk                         | ~33 ms   | custom     |
| `decision_update`     | Check for a newer engine release / model weights / new models (opt-in `apply`) | —        | admin      |
| `decision_usage`      | Usage + session stats, JSONL export for training      | —        | admin      |

With the default Laya engine, inference runs in-process over its encoder checkpoints (ModernBERT-large / mmBERT-base). One HTTP endpoint, single engine load per process.

## Install

```bash
cd DecisionMCP
uv venv .venv && source .venv/bin/activate     # or python -m venv
uv pip install -e ".[dev]"                     # or pip install -e ".[dev]"
uv pip install -e ".[dev,laya]"                # + the Laya engine (heavy: torch)
```

### Models

Models live **inside the project** in `./models` (~2.2 GB, gitignored): the three Laya checkpoints (`english`, `multilingual`, `typed-decisions`) from [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya). DecisionMCP points the Hugging Face cache there automatically (an explicit `HF_HUB_CACHE` / `HF_HOME` still wins). To download them once:

```bash
HF_HUB_CACHE="$PWD/models" python -c "from laya import Router; Router().preload()"
```

Keep them current with the `decision_update` tool (see [docs/TOOLS.md](docs/TOOLS.md#decision_update)).

## Run

```bash
# Default: 127.0.0.1:8765
decisionmcp

# Or override via env:
DECISIONMCP_PORT=9000 decisionmcp
```

The MCP endpoint uses the SSE transport at `http://127.0.0.1:8765/sse` (`GET /health` for a health check).

Every prediction is logged to `./data/usage.db` with its MCP session, for training data. Input text is **not** stored unless you set `DECISIONMCP_USAGE_STORE_TEXT=true` (see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)).

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
    "decisionmcp": {
      "transport": "sse",
      "url": "http://127.0.0.1:8765/sse",
      "lifecycle": "eager",
      "healthCheckIntervalMs": 60000
    }
  }
}
```

Start the server first (`decisionmcp`), then start Pi — all 13 tools appear as
`mcp_decisionmcp_<tool>` (e.g. `mcp_decisionmcp_decision_guard`, argument is `prompt`).
Use `/mcp` inside Pi to check connection status; with `lifecycle: "eager"` Pi
connects at session start (5 retries), with `"lazy"` you start it manually via
`/mcp:start`. The server speaks MCP over SSE at `/sse` (`GET /health` for a
health check).

> **Migrating from LayaMCP?** Tools were renamed `laya_*` → `decision_*`, so
> prompts that call `mcp_layamcp_laya_guard` must switch to
> `mcp_decisionmcp_decision_guard`. See the [CHANGELOG](CHANGELOG.md).

## Adding a new engine

Implement the `DecisionEngine` protocol (`preload`, `version`, `presets`, `predict`) in `decision_mcp/engines/myengine.py`, export it, and pass it to `DecisionBridge(engine=MyEngine(), ...)` in `server.py`. All 13 tools — including the custom-question ones — work unchanged on any engine that speaks the normalized answer shape. Details in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#adding-a-new-engine).

## Documentation

- **[AGENTS.md](AGENTS.md)** — instructions for AI coding agents working in this repo
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — layer diagram, data flow, performance
- **[docs/TOOLS.md](docs/TOOLS.md)** — per-tool behaviour, when to use, limitations
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — every env var, .env format, prod checklist
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — adding tools & engines, code style, release checklist
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** — launchd, systemd, Docker, nginx reverse proxy
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — common errors and fixes

## Adding a new tool

The architecture is plugin-style. To add a tool:

**1. Create `decision_mcp/tools/mytool.py`:**

```python
from pydantic import BaseModel, Field

from ._helpers import choice_of, require_dict
from .base import Tool


class MyInput(BaseModel):
    text: str = Field(..., description="Input text.")


class MyOutput(BaseModel):
    label: str
    confidence: float


class MyTool(Tool):
    name = "decision_mytool"
    description = "One-line description shown to the agent."
    input_schema = MyInput
    output_schema = MyOutput

    async def run(self, bridge, text: str) -> MyOutput:
        raw = bridge.predict(text, preset="...")  # or build your own questions
        require_dict(raw, self.name)
        label, confidence = choice_of(raw, "label_question", self.name)
        return MyOutput(label=label, confidence=confidence)
```

**2. Register it in `decision_mcp/tools/__init__.py`:**

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

### Adding a custom question schema (not from an engine preset)

If you want a tool that uses questions you defined yourself:

```python
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

`DecisionBridge.predict_custom()` works with every engine.

## Tests

```bash
pytest                                              # mocked + FakeEngine, no models needed, fast
DECISIONMCP_INTEGRATION=1 pytest tests/test_integration.py   # real models (needs .[laya] + ./models)
```

## License

Apache-2.0
