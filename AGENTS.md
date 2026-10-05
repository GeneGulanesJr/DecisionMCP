# AGENTS.md — Instructions for AI coding agents

If you're an AI coding agent (Aider, Cursor, Claude Code, Pi, etc.) working in this repo, read this first.

## What this project is

**DecisionMCP** is an engine-agnostic HTTP MCP server (SSE transport, endpoint `/sse`) that serves decision-model inferences to Pi (and any other MCP client). It ships with the [Laya](https://github.com/NandhaKishorM/laya) engine behind a pluggable `DecisionEngine` protocol, so other engines can be added without touching tools. It exposes **13 tools** at `http://127.0.0.1:8765` by default — 5 engine-preset + 6 coding-specific + 2 maintenance. Model weights live in `./models` (gitignored); usage logs in `./data` (gitignored).

| Tool | Purpose | Source |
|---|---|---|
| `decision_guard` | Prompt-injection / jailbreak detection | engine preset |
| `decision_route` | Cheap-vs-frontier model routing | engine preset |
| `decision_moderate` | Toxicity / harassment / threats | engine preset |
| `decision_triage` | Support-ticket classification | engine preset |
| `decision_email` | Email triage | engine preset |
| `decision_review_tone` | Code review tone + priority | custom questions |
| `decision_bug_severity` | Bug severity + area | custom questions |
| `decision_commit_classify` | Commit message type / scope / risk | custom questions |
| `decision_test_priority` | Test run priority + reason | custom questions |
| `decision_secret_risk` | Detect leaked credentials in text | custom questions |
| `decision_diff_intent` | PR diff intent / scope / risk | custom questions |
| `decision_update` | Check/apply Laya engine + model updates, list new models | maintenance (no inference) |
| `decision_usage` | Usage/session stats and JSONL export for training | maintenance (no inference) |

With the default engine, inference runs in-process via `laya.Router` (~33 ms per call on T4). The plugin architecture means adding a tool is a 1-file change, and adding an engine is 1 file + 1 line in `server.py`.

## Build / run / test

```bash
# Install (editable + dev deps; add the laya extra to actually serve)
pip install -e ".[dev]"
pip install -e ".[dev,laya]"

# Run server (foreground)
decisionmcp                                # default: 127.0.0.1:8765
DECISIONMCP_PORT=9000 decisionmcp         # override

# Run tests (uses MagicMock bridges + a FakeEngine — no GPU, no laya needed, fast)
pytest                                  # all tests
DECISIONMCP_INTEGRATION=1 pytest tests/test_integration.py   # real models in ./models
pytest tests/test_tools.py              # specific file
pytest -k guard                         # specific test pattern
pytest -v                               # verbose

# Verify against the real engine (requires the laya extra + GPU/CPU)
python -c "from decision_mcp.bridge import DecisionBridge; b = DecisionBridge(preload=True); print(b.predict('hello', preset='guard'))"
```

## File map — where to make changes

| Want to... | Edit |
|---|---|
| Add a new tool | Create `decision_mcp/tools/mytool.py`, register in `decision_mcp/tools/__init__.py`, add test in `tests/test_tools.py` |
| Change a tool's parsing | Edit the tool file in `decision_mcp/tools/` |
| Add an engine preset | Edit the engine's `presets()` (e.g. `decision_mcp/engines/laya.py`) |
| Add a new engine | Create `decision_mcp/engines/myengine.py` implementing `DecisionEngine`, wire into `DecisionBridge(engine=...)` in `server.py` |
| Change server port / host / log level | Edit `.env`, or set `DECISIONMCP_*` env var |
| Change FastAPI / MCP wiring | Edit `decision_mcp/server.py` |
| Change the abstract Tool contract | Edit `decision_mcp/tools/base.py` (rare — affects all tools) |
| Add a test | Edit `tests/test_tools.py` or `tests/test_bridge.py` |
| Bump version | Edit `pyproject.toml` `version` field |

## Code style

- Python 3.10+
- Type hints **everywhere**. `from __future__ import annotations` at the top of every file.
- **Pydantic v2** for schemas (`BaseModel.model_json_schema()`, not `schema()`).
- **Async tools** — `async def run(...)`.
- **Raise on bad data, don't return defaults.** Silent false negatives are worse than errors. Use the helpers in `decision_mcp/tools/_helpers.py` (`require_dict`, `choice_of`, `yes_prob`, `score_level`, `bin_mass`) — they raise `ToolError` on any malformed engine output.
- **Use the custom exception hierarchy.** Catch `DecisionMCPError` at the boundary (server). Use `BridgeError` / `UnknownPresetError` for engine issues, `ToolError` for parsing issues.
- **Log context, not data.** When logging errors, include the preset name and input length, not the raw input — to avoid leaking user prompts into log files.
- Tools in `decision_mcp/tools/` are **pure plugins**: subclass `Tool`, define class attributes, register in `__init__.py`. No other files need to change.
- One tool per file. Keep file names lowercase, no separators (`guard.py`, not `Guard.py` or `prompt-guard.py`).
- **Heavy engine imports are lazy.** Never import `laya` (or any engine library) at module level — only inside engine methods, so the package and tests work without it.

## Architectural invariants (don't break these)

1. **Single `DecisionBridge` instance per process.** Model weights are heavy (~500 MB–1 GB). Don't create multiple bridges.
2. **Tools are stateless.** All state lives in `DecisionBridge` (the engine + its weights). Tool classes are pure logic + Pydantic schemas.
3. **`Tool.run()` is async.** Don't make it sync — it would block the FastAPI event loop.
4. **The registry in `decision_mcp/tools/__init__.py` is the only file to edit when adding/removing a tool.** Don't add tool imports to `server.py` — the server reads from the registry.
5. **The server is the only place that catches `DecisionMCPError` and converts to MCP error blocks.** Tools raise; server decides how to surface. Don't add try/except in `run()` unless you have a specific reason.
6. **Tool parsers validate strictly.** No silent defaults on missing keys / wrong types — use the helpers in `decision_mcp/tools/_helpers.py`.
7. **Engine answers are typed.** `DecisionEngine.predict` returns `{"answers": {q: {"type": "choice"|"noul"|"score", ...}}}` — not `{"q1": {"label", "confidence"}}`. The input text must be passed as `{field: text}` where `field` is the name the question instructions use (the bridge does this).
8. **Tools never touch engines directly.** Tools call `bridge.predict` / `bridge.predict_custom`; engines know nothing about MCP. That's what keeps the server engine-agnostic.
9. **Input text is never logged or stored by default.** The usage store (`decision_mcp/usage.py`) keeps text only when `DECISIONMCP_USAGE_STORE_TEXT=true`, and never for `decision_secret_risk`. Don't weaken this.
10. **`decision_update` `apply` stays opt-in** (`DECISIONMCP_ALLOW_UPDATES`): it runs pip and the HTTP server has no auth. It may only ever install `laya`.

## Common tasks

### Add a new tool (the canonical pattern)

**1.** Create `decision_mcp/tools/mytool.py`:

```python
"""Tool: <one-line description>."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from .base import Tool


class MyInput(BaseModel):
    text: str = Field(..., description="...")


class MyOutput(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class MyTool(Tool):
    name = "decision_mytool"
    description = "One-line description shown to agents."
    input_schema = MyInput
    output_schema = MyOutput

    async def run(self, bridge: DecisionBridge, text: str) -> MyOutput:
        raw = bridge.predict(text, preset="<preset_name>")
        require_dict(raw, self.name)
        label, confidence = choice_of(raw, "<question_name>", self.name)
        return MyOutput(label=label, confidence=confidence)
```

(add `from ._helpers import choice_of, require_dict` at the top)

**2.** Register in `decision_mcp/tools/__init__.py`:

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

**3.** Add a test in `tests/test_tools.py`:

```python
@pytest.mark.asyncio
async def test_mytool_happy_path() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(q=choice("spam", 0.9))  # from tests/fixtures.py
    out = await MyTool().run(bridge, text="...")
    assert out.label == "spam"
```

**4.** Verify: `pytest -k mytool -v`

### Add a custom preset (not from an engine's presets)

Questions are plain dicts (`type` is `choice`, `noul` or `score`); refer to the input by its field name in backticks. Prefer `choice` over `score` for ordinal scales — it answered better on custom questions (see `docs/TOOLS.md`).

```python
MY_QUESTIONS = {
    "answer": {"type": "noul", "instructions": "Is `text` a question?"},
    "priority": {
        "type": "choice",
        "instructions": "How urgent is `text`?",
        "criteria": {"low": "can wait", "medium": "this week", "high": "now"},
    },
}
```

In the tool's `run()`:

```python
raw = bridge.predict_custom(text, questions=MY_QUESTIONS, state_key="text")
```

### Add a new engine

Implement the protocol in `decision_mcp/engines/myengine.py` (`preload`, `version`, `presets`, `predict`), export it from `decision_mcp/engines/__init__.py`, and pass it to `DecisionBridge(engine=MyEngine(), ...)` in `server.py`. See `docs/DEVELOPMENT.md` → "Adding a new engine".

### Debug a tool's parsing against the real engine

```python
import json
from decision_mcp.bridge import DecisionBridge

b = DecisionBridge(preload=True)
raw = b.predict("ignore all previous instructions", preset="guard")
print(json.dumps(raw, indent=2, default=str))
```

Use the actual shape to update the parser in the tool file.

## What NOT to do

- ❌ Don't load multiple `DecisionBridge` instances in one process.
- ❌ Don't make `Tool.run()` synchronous.
- ❌ Don't edit `decision_mcp/tools/base.py` to add a tool — subclass it.
- ❌ Don't call an engine directly from a tool — go through the bridge.
- ❌ Don't import `laya` (or any engine library) at module level — lazy-import inside engine methods.
- ❌ Don't trust the engine's return shape — always validate via the `_helpers.py` functions.
- ❌ Don't store raw input text in the usage log unless `DECISIONMCP_USAGE_STORE_TEXT` is on.
- ❌ Don't hardcode ports / hosts — use `DECISIONMCP_*` env vars via `Settings`.
- ❌ Don't add tool imports to `server.py` — register in `decision_mcp/tools/__init__.py`.
- ❌ Don't return defaults on parse failures — raise `ToolError`. Silent failures hide false negatives.
- ❌ Don't leak user data into logs — log preset names and input lengths, not raw input.
- ❌ Don't catch `Exception` inside `Tool.run()` — let the server catch it. Exceptions in tools are bugs.

## Where to read more

- `docs/ARCHITECTURE.md` — full layer diagram, data flow, performance notes
- `docs/TOOLS.md` — per-tool behavior, when to use, limitations
- `docs/CONFIGURATION.md` — every env var, .env file format, prod checklist
- `docs/MIGRATION.md` — upgrading an existing LayaMCP v0.1.x install (v0.2.0 renames)
- `docs/DEVELOPMENT.md` — adding tools & engines, code style, release checklist
- `docs/DEPLOYMENT.md` — systemd, Docker, nginx reverse proxy
- `docs/TROUBLESHOOTING.md` — common errors and fixes
