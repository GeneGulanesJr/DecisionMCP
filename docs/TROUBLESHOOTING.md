# Troubleshooting

## Server won't start

### `Address already in use`

Port 8765 is taken. Either:
- Stop the conflicting process, or
- `DECISIONMCP_PORT=9000 decisionmcp`

Find what's using the port:

```bash
# Linux / macOS
lsof -i :8765
ss -tlnp | grep 8765

# Windows
netstat -ano | findstr :8765
```

### `ModuleNotFoundError: No module named 'laya'`

The default engine isn't installed. The `laya` extra is opt-in since the
engine-agnostic refactor:

```bash
pip install -e ".[laya]"
```

### `ModuleNotFoundError: No module named 'mcp'`

The MCP SDK isn't installed:

```bash
pip install "mcp>=1.9,<2"
```

### Model fails to download on first run

First run downloads model weights from HuggingFace (~500 MB–1 GB). Requires:
- Network access to `huggingface.co`
- ~2 GB free disk space in your HF cache (the project redirects it to `./models` unless `HF_HUB_CACHE`/`HF_HOME` is set)

If behind a firewall, pre-download on a connected machine:

```bash
pip install huggingface_hub
HF_HUB_CACHE="$PWD/models" huggingface-cli download convaiinnovations/laya
HF_HUB_CACHE="$PWD/models" huggingface-cli download convaiinnovations/laya-multilingual
HF_HUB_CACHE="$PWD/models" huggingface-cli download convaiinnovations/laya-typed-decisions
```

### `ImportError` inside `decision_mcp/server.py`

The `mcp` SDK API may have changed between versions. Check your installed version:

```bash
pip show mcp
```

Then check the actual exports:

```python
python -c "import mcp.server.sse; print(dir(mcp.server.sse))"
```

Adjust the imports in `decision_mcp/server.py` to match.

## Tool returns wrong shape

The engine's return format may have changed (or differ from what your parser expects). Run a real call and inspect:

```python
import json
from decision_mcp.bridge import DecisionBridge

b = DecisionBridge(preload=True)
raw = b.predict("test input", preset="guard")
print(json.dumps(raw, indent=2, default=str))
```

Then adjust the parser in the relevant tool file (e.g. `decision_mcp/tools/guard.py`).

Tool parsers are intentionally strict: they raise `ToolError` on unexpected shapes instead of returning defaults, so a drifted schema surfaces as a loud MCP error, never as silent wrong results.

## Tests fail

### `pytest` says no tests collected

Make sure you're in the project root and `tests/` is reachable. Pytest config in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Import errors in tests

Reinstall in editable mode:

```bash
pip install -e ".[dev]"
```

The mocked suite runs without `laya` installed (bridge tests use the
`FakeEngine` double in `tests/conftest.py`). If you see import errors about
`laya`, something bypassed the lazy import — engine code belongs inside
methods in `decision_mcp/engines/laya.py`, not at module level.

### `RuntimeWarning: coroutine '...' was never awaited`

You forgot `await` on an async tool call. Tool `run()` methods are async — always `await tool.run(...)`.

## Pi can't connect

### Check the server is up

```bash
curl http://127.0.0.1:8765/health
```

Should return `{"status": "ok", "tools": 13}`.

### Check Pi's MCP config

`~/.pi/agent/mcp.json` (global) or `.pi/mcp.json` (per project) — pi ≥ 1.0's
built-in client speaks streamable HTTP (the legacy `pi-mcp-extension`
bridge is gone; its `transport`/`lifecycle` keys are not valid here):

```json
{
  "mcpServers": {
    "decisionmcp": {
      "url": "http://127.0.0.1:8765/mcp",
      "description": "Decision-model tools over the Laya engine"
    }
  }
}
```

Common mistakes:
- Still registered under the old `layamcp` key, or still pointing at the
  SSE endpoint (`/sse`) — the built-in client needs `/mcp` and rejects SSE
- Still running the removed `pi-mcp-extension` (delete it with
  `pi remove npm:pi-mcp-extension`) — it collides with the built-in `/mcp`
- Wrong port
- Config not picked up — run `/reload` inside Pi, or restart the session

Then verify outside Pi:

```bash
pi mcp list   # expect: decisionmcp: connected, 13 tools
```

### Check the firewall

On localhost binds (`127.0.0.1`), firewall usually isn't an issue. If you bind to `0.0.0.0`:

```bash
# Linux (ufw)
sudo ufw allow 8765/tcp

# Linux (firewalld)
sudo firewall-cmd --permanent --add-port=8765/tcp
sudo firewall-cmd --reload
```

## Slow first call

First call loads model weights from disk (~2–5 seconds). Subsequent calls are ~33 ms.

To pre-load at boot:

```bash
DECISIONMCP_PRELOAD_MODELS=true decisionmcp    # default
```

If pre-loading is slow, you may be on a slow disk or the model cache isn't warm. Check:

```bash
du -sh ./models/
```

## High memory usage

Each loaded checkpoint is ~500 MB–1 GB. The engine holds them in VRAM. To cap (LayaEngine):

```python
# decision_mcp/engines/laya.py — keep only one loaded at a time
self._router = Router(preload=False, max_loaded=1)
```

Or run with `DECISIONMCP_PRELOAD_MODELS=false` so weights load on demand and can be evicted.

## OOM / CUDA out of memory

The checkpoint is too big for your GPU. Options:
- Use a smaller checkpoint (English-only, not multilingual)
- Reduce `max_loaded`
- Run on CPU (`CUDA_VISIBLE_DEVICES="" decisionmcp`) — slower but no VRAM cap

## Tool registry empty / new tool not appearing

If you added a tool but it's not in `tools/list`:

1. **Did you register it?** Check `decision_mcp/tools/__init__.py`:
   ```python
   TOOLS = [..., MyTool()]
   ```
2. **Did you restart the server?** The registry is read at import time, not per-request.
3. **Are there import errors?** Check the server logs. A broken tool file can fail the whole `__init__.py` import, leaving the registry broken.

```bash
DECISIONMCP_LOG_LEVEL=DEBUG decisionmcp
```

Look for tracebacks on startup.

## Getting help

- Check the upstream [Laya repo](https://github.com/NandhaKishorM/laya) for engine API questions.
- Check the [MCP spec](https://modelcontextprotocol.io) for protocol questions.
- File an issue in this repo for project-specific questions.

---

## Tool returns an error instead of a result

Tools raise `ToolError` on any malformed engine output. The server returns these as MCP error content blocks. To diagnose:

### 1. Read the error message

The error includes context:

```
[decision_guard] Missing expected key 'q1' in engine answers. Got keys: ['injection_check']
```

This tells you the engine's actual question key is `injection_check`, not the assumed `q1`.

### 2. Inspect the raw engine output

```python
import json
from decision_mcp.bridge import DecisionBridge

b = DecisionBridge(preload=True)
raw = b.predict("test input", preset="guard")
print(json.dumps(raw, indent=2, default=str))
```

### 3. Fix the parser

Either:
- Update the key name in the tool file (e.g., change `"q1"` to `"injection_check"` in `decision_mcp/tools/guard.py`).
- Or accept multiple shapes with a fallback:

  ```python
  try:
      first = choice_of(raw, "q1", self.name)
  except ToolError:
      first = choice_of(raw, "injection_check", self.name)
  ```

### 4. Add a regression test

Add the new key name to `tests/test_tools.py` so the fix sticks:

```python
@pytest.mark.asyncio
async def test_guard_with_alternate_key():
    bridge = MagicMock()
    bridge.predict.return_value = result(
        injection_check=noul(0.9), prompt_injection=noul(0.1)
    )
    out = await GuardTool().run(bridge, prompt="...")
    assert out.is_injection is True
```

---

## Server keeps running but every tool call fails

If all tool calls return errors but the server itself is up:

### 1. Check the engine loaded successfully

Look at startup logs for `ModelLoadError`:

```
ModelLoadError: Failed to initialize decision engine (preload=true): ...
```

If you see this, the model weights couldn't load — see "Model fails to download on first run" above.

### 2. Run with `DEBUG` logging

```bash
DECISIONMCP_LOG_LEVEL=DEBUG decisionmcp
```

Look for tracebacks on the failed calls.

### 3. Verify the engine works outside the server

```python
from decision_mcp.bridge import DecisionBridge
b = DecisionBridge(preload=True)
print(b.predict("test", preset="guard"))
```

If this fails, the problem is upstream (the engine itself), not the server.

---

## Pydantic ValidationError on tool call

Means the inputs you sent don't match the tool's input schema. The error message tells you which field is wrong:

```
Invalid input for decision_guard: 1 validation error for GuardInput
prompt
  Input should be a valid string [type=string_type, input_value=None, input_type=None]
```

Fix the client (Pi) to pass the right argument name and type. Each tool's input schema is in `decision_mcp/tools/<tool>.py`:

- `decision_guard`: `{"prompt": str}`
- `decision_route`: `{"prompt": str}`
- `decision_triage`: `{"text": str}`
- `decision_moderate`: `{"text": str}`
- `decision_email`: `{"body": str}`

---

## Server logs are too noisy / too quiet

Adjust `DECISIONMCP_LOG_LEVEL`:

- `DEBUG` — verbose, includes every event
- `INFO` — startup, tool calls (default)
- `WARNING` — only invalid input, unknown tools
- `ERROR` — only failures (recommended for production)

```bash
DECISIONMCP_LOG_LEVEL=ERROR decisionmcp
```
