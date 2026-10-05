# Architecture

## Layer diagram

```
┌──────────────────────────────────────────────┐
│  MCP clients                                 │
│  • Pi (TUI / chat)                          │
│  • Future web UI / IDE / Slack bot / etc.   │
└────────────────┬─────────────────────────────┘
                 │ HTTP MCP (JSON-RPC 2.0)
                 ▼
┌──────────────────────────────────────────────┐
│  FastAPI app  (decision_mcp.server:app)      │
│  ┌────────────────────────────────────────┐  │
│  │  POST/GET/DELETE /mcp  (streamable     │  │
│  │  HTTP — pi's built-in client, current  │  │
│  │  MCP spec)                             │  │
│  │  - initialize                        │  │
│  │  - tools/list                        │  │
│  │  - tools/call                        │  │
│  │  GET /sse  (legacy MCP over SSE)      │  │
│  └────────────────────────────────────────┘  │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  Tool registry  (decision_mcp.tools.TOOLS)   │
│  • GuardTool        • ReviewToneTool         │
│  • RouteTool        • BugSeverityTool        │
│  • TriageTool       • ... (13 total)         │
└────────────────┬─────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────┐
│  DecisionBridge  (decision_mcp.bridge)       │
│  • Single engine instance per process        │
│  • Preset registry (from the engine)         │
│  • predict(state, preset)        → dict      │
│  • predict_custom(state, qs)     → dict      │
│  • Usage recording + error normalization     │
└────────────────┬─────────────────────────────┘
                 │ DecisionEngine protocol
                 ▼
┌──────────────────────────────────────────────┐
│  Engines  (decision_mcp.engines.*)           │
│  ┌────────────────────────────────────────┐  │
│  │ LayaEngine (default)                   │  │
│  │  • wraps laya.Router                   │  │
│  │  • English / Multilingual /            │  │
│  │    typed-decisions via script detect   │  │
│  │  • 5 preset question schemas           │  │
│  │  • ~33 ms / call on T4                 │  │
│  └────────────────────────────────────────┘  │
│  (future engines implement the same          │
│   protocol and plug in here)                 │
└──────────────────────────────────────────────┘
```

## Component responsibilities

### `decision_mcp/server.py` — HTTP transport

- Owns the single `DecisionBridge` instance (loaded at import time).
- Mounts MCP routes via the `mcp[server]` SDK.
- Translates JSON-RPC 2.0 requests to Python calls.
- Returns tool results as MCP `content` blocks.

### `decision_mcp/bridge.py` — engine-agnostic bridge

- Owns the active `DecisionEngine` (which owns all loaded weights).
- Exposes `predict(state, preset)` for engine-registered presets.
- Exposes `predict_custom(state, questions)` for custom question schemas.
- Centralizes the model lifecycle so weights load **exactly once per process**.
- Records every call to the `UsageStore` (with engine-version provenance).
- Normalizes engine failures into `BridgeError` subclasses.

### `decision_mcp/engines/base.py` — the engine contract

- `DecisionEngine` protocol: `preload()`, `version()`, `presets()`, `predict(state, questions)`.
- `PresetSpec`: a named question schema + the state key its instructions reference.
- Engines know nothing about MCP, tools, or usage logging — that's the bridge's job.

### `decision_mcp/engines/laya.py` — default engine

- `LayaEngine` wraps `laya.Router`; `laya` is imported lazily so the package
  (and the mocked test suite) work without the heavy library installed.
- Installed via the `laya` extra: `pip install ".[laya]"`.

### `decision_mcp/config.py` — settings

- `Settings` via `pydantic-settings`.
- Env-overridable via `DECISIONMCP_*` prefix.
- Defaults: `127.0.0.1:8765`, `preload=True`, `INFO` logs.

### `decision_mcp/tools/base.py` — `Tool` ABC

- Defines the contract for any tool:
  - `name` (str)
  - `description` (str)
  - `input_schema` (Pydantic model)
  - `output_schema` (Pydantic model)
  - `async run(bridge, **kwargs)`
  - `to_mcp_schema()`

### `decision_mcp/tools/*.py` — tool implementations

- One file per tool.
- Each parses the engine's normalized dict output into its own `*Output` Pydantic schema.
- Pure plugins — easy to add, easy to remove, easy to test in isolation.

### `decision_mcp/tools/__init__.py` — registry

- A single `TOOLS: list[Tool]` of instantiated tool objects.
- Order = display order in MCP `tools/list`.
- **The only file to edit when adding or removing a tool.**

## Data flow per tool call

1. **Pi** POSTs a `tools/call` JSON-RPC request to `/mcp` (streamable HTTP;
   the legacy `/sse` endpoint behaves equivalently) → FastAPI route.
2. **Server** dispatches to `Tool.run(bridge, **arguments)`.
3. **Tool** calls `bridge.predict(state, preset)` or `bridge.predict_custom(state, questions)`.
4. **Bridge** looks up the preset (or uses the custom questions) and forwards
   `{state_key: text}` + questions to `engine.predict(...)`.
5. **Engine** runs a forward pass (~33 ms) and returns the normalized decision shape:
   ```python
   {
       "answers": {
           "<question_key>": {"type": "choice"|"noul"|"score", ..., "answer_confidence": 0.x},
           ...
       },
       "usage": {...},
       "routing": {...},
   }
   ```
6. **Tool** parses the dict into its `*Output` Pydantic model (via `tools/_helpers.py`).
7. **Bridge** records the call (tool, latency, answers, engine version) to the usage log.
8. **Server** serializes the result back as MCP `content` blocks.

## Adding new tools

See `AGENTS.md` → "Common tasks → Add a new tool" for the canonical pattern.

The pattern is:
- **One file** in `decision_mcp/tools/`.
- **Two lines** in `decision_mcp/tools/__init__.py` (import + register).
- **One test** in `tests/test_tools.py`.

No other changes. Schema, route, and dispatch all derive from the class.

## Adding a new engine

Implement the `DecisionEngine` protocol in `decision_mcp/engines/myengine.py`
(`preload`, `version`, `presets`, `predict`), export it from
`decision_mcp/engines/__init__.py`, and pass it to `DecisionBridge(engine=...)`
in `server.py`. Tools and presets that exist on the new engine work unchanged;
custom-question tools work with any engine from day one because they bring
their own question schemas.

## Performance characteristics

| Operation                | Latency (T4) | Notes                                  |
| ------------------------ | ------------ | -------------------------------------- |
| First call (cold start)  | 2–5 s        | Model load from disk                   |
| Subsequent calls         | ~33 ms       | Single forward pass (LayaEngine)       |
| Multilingual input       | ~33 ms       | Router picks `mmBERT-base` checkpoint  |
| Typed-decisions input    | ~33 ms       | Router picks `typed-decisions` fine-tune |
| Concurrent requests      | serialized   | Single forward pass at a time (MPS/CUDA queue) |

The model load happens **once per process** at `DecisionBridge(preload=True)` in `server.py`. To defer load until first call, set `DECISIONMCP_PRELOAD_MODELS=false`.

To cap memory if multiple checkpoints would exceed VRAM (LayaEngine):

```python
# decision_mcp/engines/laya.py
self._router = Router(preload=False, max_loaded=1)  # keep one loaded at a time
```

## Security

- **No built-in auth.** Don't bind to `0.0.0.0` without putting a reverse proxy with auth in front (see `docs/DEPLOYMENT.md`).
- **No rate limiting.** Add at the reverse-proxy layer.
- **Tool inputs are arbitrary text.** Engine prompts can be adversarial. The `decision_guard` tool is meant to detect this — consider running it on any text that originated from outside your system.

## Extensibility points

| To add... | Touch this file |
|---|---|
| A new tool | `decision_mcp/tools/mytool.py` + register in `__init__.py` |
| A new engine | `decision_mcp/engines/myengine.py` (implement `DecisionEngine`) |
| A new engine preset | the engine's `presets()` (e.g. `engines/laya.py`) |
| A custom question schema (not from a preset) | `bridge.predict_custom(state, questions)` |
| A new FastAPI route outside MCP | `decision_mcp/server.py` |
| New settings | `decision_mcp/config.py` |
| New tool categories (e.g., group "guardrails" vs "routing") | Refactor `tools/` into subpackages |

The plugin pattern means most extensions are **1 file + 2 lines**.
