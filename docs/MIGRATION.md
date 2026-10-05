# Migrating from LayaMCP to DecisionMCP (v0.1.x → v0.2.0)

v0.2.0 renames the project to **DecisionMCP** and moves Laya behind a
pluggable `DecisionEngine` protocol, so the tools are no longer coupled to a
single model vendor. Laya is still the default engine — functionally nothing
about the 13 decisions changes — but every externally visible name does. This
guide walks through upgrading an existing LayaMCP install.

## 1. Pull and reinstall

```bash
git pull
pip install -e ".[dev,laya]"     # laya is now an optional extra
```

The importable package changed from `laya_mcp` to `decision_mcp`, and the
console script from `layamcp` to `decisionmcp`. The old editable install
shadows nothing, but re-installing refreshes the entry point.

If you only run the mocked tests or develop tools, `.[dev]` alone is enough —
`laya` (and its torch dependency) is imported lazily and only needed to serve.

## 2. Rename the env vars

The settings prefix changed from `LAYAMCP_` to `DECISIONMCP_`. Old names are
**silently ignored** (`extra="ignore"`), so a missed rename shows up as
"default seems to be on" rather than an error — check each one:

| Old | New |
|---|---|
| `LAYAMCP_HOST` | `DECISIONMCP_HOST` |
| `LAYAMCP_PORT` | `DECISIONMCP_PORT` |
| `LAYAMCP_PRELOAD_MODELS` | `DECISIONMCP_PRELOAD_MODELS` |
| `LAYAMCP_LOG_LEVEL` | `DECISIONMCP_LOG_LEVEL` |
| `LAYAMCP_DATA_DIR` | `DECISIONMCP_DATA_DIR` |
| `LAYAMCP_USAGE_ENABLED` | `DECISIONMCP_USAGE_ENABLED` |
| `LAYAMCP_USAGE_STORE_TEXT` | `DECISIONMCP_USAGE_STORE_TEXT` |
| `LAYAMCP_ALLOW_UPDATES` | `DECISIONMCP_ALLOW_UPDATES` |

Update `.env`, and any service definitions that inject env vars (systemd
units, `docker-compose.yml`, NSSM `AppEnvironmentExtra`, launchd plists —
see §6).

## 3. Usage database — automatic

On first open, the store renames `calls.laya_version` → `engine_version` in
place (SQLite `ALTER TABLE ... RENAME COLUMN`). Data is preserved; no action
needed. Rows whose provenance was captured before that column existed keep
`engine_version = NULL`.

Already-exported JSONL files still say `laya_version` — that's historical
fact, not rewritten. New exports carry `engine_version`.

## 4. Re-register the server in Pi

In `~/.pi/agent/mcp.json` (global) or `.pi/mcp.json` (per project), rename
the server entry — the key determines the tool prefix Pi exposes:

```json
{
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

- Tools are now `mcp_decisionmcp_decision_guard`, `mcp_decisionmcp_decision_route`, …
  (was `mcp_layamcp_laya_guard`, …).
- **Prompts, skills, and agent instructions that name tools must be updated**
  (`laya_guard` → `decision_guard`, `laya_usage` → `decision_usage`, etc.).
- Start the server with `decisionmcp` before starting Pi, as before.

## 5. Relink the Pi lifecycle extension

The extension file was renamed:

```bash
rm ~/.pi/agent/extensions/layamcp-lifecycle.ts
ln -sf "$PWD/integrations/pi/decisionmcp-lifecycle.ts" ~/.pi/agent/extensions/decisionmcp-lifecycle.ts
```

Its refcount/lock/pid files under `data/` are renamed too
(`layamcp-refs.json` → `decisionmcp-refs.json`, etc.). Stale files from the
old names are simply orphaned — delete them whenever.

## 6. Swap the launchd agents (macOS)

```bash
# stop the old agents
launchctl bootout gui/$(id -u)/com.gulaneskorp.layamcp
launchctl bootout gui/$(id -u)/com.gulaneskorp.layamcp-backup
rm ~/Library/LaunchAgents/com.gulaneskorp.layamcp*.plist

# install the new ones
cp deploy/launchd/com.gulaneskorp.decisionmcp.plist ~/Library/LaunchAgents/
cp deploy/launchd/com.gulaneskorp.decisionmcp-backup.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gulaneskorp.decisionmcp.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gulaneskorp.decisionmcp-backup.plist
```

The shipped plists assume the checkout lives at
`~/Documents/GulanesKorp/DecisionMCP` — if your repo is elsewhere, edit the
paths inside the plists (and the `REPO` constant in the lifecycle extension)
before bootstrapping.

## 7. Code that imported the old package

If you have scripts or integrations importing LayaMCP directly:

| Old | New |
|---|---|
| `from laya_mcp.bridge import LayaBridge` | `from decision_mcp.bridge import DecisionBridge` |
| `LayaBridge(preload=...)` | `DecisionBridge(preload=...)` — or `DecisionBridge(engine=MyEngine(), ...)` |
| `LayaBridge.PRESETS` / `PRESET_STATE_KEYS` | `bridge.presets` (dict of `PresetSpec` with `.questions` / `.state_key`) |
| `from laya_mcp.errors import LayaMCPError` | `from decision_mcp.errors import DecisionMCPError` |
| `bridge.router` | `bridge.engine` (a `DecisionEngine`, not a laya `Router`) |
| `store.record(...)` (no version) | `store.record(..., engine_version=...)` |
| `uvicorn laya_mcp.server:app` | `uvicorn decision_mcp.server:app` |

The repo's own integrations were already converted — see
`integrations/lapis-memory-save-classified.ts` (env vars are now
`LAPIS_DECISIONMCP_*`) and `integrations/pi/`.

## FAQ

**Can I keep the old `laya_*` tool names for a transition period?**
Yes — tool wire names are just the `name` class attribute in each file in
`decision_mcp/tools/`. Change them back (and nothing else) and the old names
come back; mixing old and new names also works since the registry only
requires uniqueness.

**Does the engine abstraction change any tool output?**
No. Tool inputs, outputs, and thresholds are untouched; only names and the
backend seam changed.

**What if I don't install the `laya` extra?**
The server fails fast at startup with `ModelLoadError: ... No module named
'laya'`. Everything else (tools list building, tests, docs tooling) works
without it.

**Where's the architectural rationale?**
`docs/ARCHITECTURE.md` (layer diagram + "Adding a new engine") and
`docs/DEVELOPMENT.md` → "Adding a new engine".
