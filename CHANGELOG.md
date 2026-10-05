# Changelog

## [Unreleased]

### Added: streamable HTTP transport (`/mcp`) — pi's built-in MCP client

The server now serves the current MCP transport, **streamable HTTP**, at
`/mcp` (POST/GET/DELETE) alongside legacy SSE (`/sse` kept for older
clients). This is what pi ≥ 1.0's **built-in MCP client** speaks — the
community `pi-mcp-extension` bridge is no longer needed.

- Pi config: `~/.pi/agent/mcp.json` uses the built-in format —
  `{"mcpServers": {"decisionmcp": {"url": "http://127.0.0.1:8765/mcp",
  "description": "…"}}}`. The bridge's `transport`/`lifecycle`/`settings`
  keys are gone.
- Tool naming: bridge `mcp_decisionmcp_<tool>` → built-in
  `mcp__decisionmcp__<tool>`.
- Usage log: streamable-HTTP requests are attributed via the
  `mcp-session-id` header — one usage-log session per MCP session;
  client `DELETE` (session termination) closes it.
- Starlette 1.7 quirk: `Mount("/mcp")` 307-redirects the exact `/mcp` path
  under real ASGI servers (TestClient masks it), so the transport is served
  by a root-mounted fallback guard; `tests/test_server_http.py` covers it.
- Lifecycle extension: fixed stale `REPO` path (pointed at a
  never-created `GulanesKorp/DecisionMCP` directory — the rename stayed
  in-tree), so `decisionmcp-lifecycle.ts` again spawns/stops the server
  with Pi sessions; relink the extension symlink if you had the old
  `layamcp-lifecycle.ts` one.
- Deps: `mcp[server]<2` → `mcp>=1.9,<2` (the `server` extra no longer
  exists in current SDK releases; no other dependency changes).

## [0.2.0] - 2026-10-05

### Breaking: DecisionMCP — the server is now engine-agnostic

LayaMCP is renamed **DecisionMCP** so decision tools are no longer coupled to
the Laya library. Laya moves behind a pluggable engine contract; Laya itself
remains the default engine.

**Migration guide:** step-by-step instructions live in
[docs/MIGRATION.md](docs/MIGRATION.md). Summary:

- Package: `laya_mcp` → `decision_mcp`; console script `layamcp` → `decisionmcp`.
- Env vars: `LAYAMCP_*` → `DECISIONMCP_*` (rename them in `.env` and any
  service definitions; old names are ignored).
- Tool wire names: `laya_*` → `decision_*` (e.g. `laya_guard` → `decision_guard`).
  Pi exposes them as `mcp_decisionmcp_<tool>` now — update `~/.pi/agent/mcp.json`
  (server key `layamcp` → `decisionmcp`) and any prompts that name tools.
- Install: `laya` is now an optional extra (`pip install ".[laya]"`). The code
  and mocked test suite run without it; serving needs it.
- Exception base: `LayaMCPError` → `DecisionMCPError`.
- Usage DB: the `calls.laya_version` column is renamed to `engine_version`
  in place on first open (data preserved); `stats()` reports
  `by_engine_version` and JSONL exports carry `engine_version`.
- Pi lifecycle extension: `integrations/pi/layamcp-lifecycle.ts` →
  `decisionmcp-lifecycle.ts` (relink the symlink; refcount files under `data/`
  are renamed too).
- launchd agents: `com.gulaneskorp.layamcp*` → `com.gulaneskorp.decisionmcp*`
  (bootout the old labels, bootstrap the new plists).

### Added
- `decision_mcp/engines/` package: `DecisionEngine` protocol (`preload`,
  `version`, `presets`, `predict`) + `PresetSpec` — the seam for pluggable
  decision models
- `LayaEngine` adapter (`decision_mcp/engines/laya.py`); `laya` is imported
  lazily so the package and tests work without it installed
- `DecisionBridge` accepts an injected engine (`DecisionBridge(engine=...)`)
- `tests/conftest.py`: `FakeEngine` double — the whole mocked suite runs
  without `laya`/torch; `pytest.importorskip("laya")` gates the one
  default-engine test
- `tests/test_usage.py`: migration test for LayaMCP-era `laya_version` DBs

### Changed
- `LayaBridge` → `DecisionBridge`; presets come from the engine
  (`bridge.presets`), not a class-level dict
- Per-row provenance in the usage log is engine-generic (`engine_version`)
- Tool descriptions and error messages no longer name Laya
  ("Expected dict from engine", "Engine predict failed (...)")
- `pyproject.toml`: project `decisionmcp` v0.2.0; `laya>=0.3.21` moved to a
  `[laya]` extra

### Removed
- Module-level `from laya import Router` import in the bridge package

## [0.1.x]

### Added
- `laya_update` tool: checks PyPI for a newer `laya`, the Hub for newer model commits and for new Laya models; opt-in `apply` (`LAYAMCP_ALLOW_UPDATES=true`)
- `laya_usage` tool + usage store: SQLite log of every prediction and MCP session (tool, model, latency, tokens, the engine's answers), stats and JSONL export for training. Input text is stored only with `LAYAMCP_USAGE_STORE_TEXT=true`, and never for `laya_secret_risk`
- Models are stored in `./models` (Hugging Face cache redirected there; `HF_HUB_CACHE` / `HF_HOME` still override)
- `tests/test_integration.py`: opt-in (`LAYAMCP_INTEGRATION=1`) checks of all inference tools against the real models
- Pi lifecycle extension + launchd agents (server + nightly corpus backup)
- Per-row laya version provenance; labeled-only training export

### Fixed
- Tools parse the real typed answer shapes (`answers` → `choice` / `noul` / `score`) with strict validation instead of silent defaults
- Server: `/messages/` mounted as a raw ASGI app and `/sse` returns an empty `Response`, fixing "response already completed" errors that broke every client

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of LayaMCP — HTTP MCP server wrapping the Laya decision engine
- Five tools: `laya_guard`, `laya_route`, `laya_moderate`, `laya_triage`, `laya_email`
- Plugin-style architecture: adding a tool = 1 file + 2 lines in the registry
- Custom exception hierarchy with defensive parsing (raises on malformed engine output)
- Configuration via `LAYAMCP_*` env vars (HOST, PORT, PRELOAD_MODELS, LOG_LEVEL)
- Pytest test suite (mocked bridges — no GPU needed)
- AGENTS.md for AI coding agent instructions
- docs/ folder: ARCHITECTURE, TOOLS, CONFIGURATION, DEVELOPMENT, DEPLOYMENT, TROUBLESHOOTING
- integrations/lapis-memory-save-classified.ts — spec for LaPis-side integration
