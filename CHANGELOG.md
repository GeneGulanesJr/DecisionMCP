# Changelog

## [Unreleased]

### Added
- `laya_update` tool: checks PyPI for a newer `laya`, the Hub for newer model commits and for new Laya models; opt-in `apply` (`LAYAMCP_ALLOW_UPDATES=true`)
- `laya_usage` tool + `laya_mcp/usage.py`: SQLite log of every prediction and MCP session (tool, model, latency, tokens, Laya's answers), stats and JSONL export for training. Input text is stored only with `LAYAMCP_USAGE_STORE_TEXT=true`, and never for `laya_secret_risk`
- Models are stored in `./models` (Hugging Face cache redirected there; `HF_HUB_CACHE` / `HF_HOME` still override)
- `tests/test_integration.py`: opt-in (`LAYAMCP_INTEGRATION=1`) checks of all inference tools against the real models

### Fixed
- Tools now parse Laya's real output (`answers` → `choice` / `noul` / `score`) instead of an assumed `{"q1": {"label", "confidence"}}` shape, and use the real preset question names
- Text is passed to Laya as `{field: text}` (the field each question's instructions name) instead of a bare string
- `LayaBridge` calls `Router.predict` directly (the `Router.agents` attribute it used doesn't exist), which also restores multilingual routing
- Custom question sets for the six coding tools are plain Laya question dicts (`render_options(key=...)` doesn't exist); ordinal scales use `choice`, which answered better than `score`
- Server: `/messages/` mounted as a raw ASGI app and `/sse` returns an empty `Response`, fixing "response already completed" errors that broke every client; `tools/list` returns `mcp.types.Tool` and tool results/errors are `CallToolResult` (`isError` is now a real MCP error flag)
- `ToolError(cause=...)` now sets `__cause__`

### Changed
- Output schemas: `laya_triage` returns `is_urgent` / `churn_risk` / `refund_requested` as booleans and `frustration` as Laya's legend text; `laya_email` returns `category` (was `intent`) plus `is_spam` / `is_phishing`
- Requires `laya>=0.3.21`; declares `huggingface-hub` and `packaging`

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of LayaMCP — HTTP MCP server wrapping the Laya decision engine
- Five tools: `laya_guard`, `laya_route`, `laya_moderate`, `laya_triage`, `laya_email`
- Plugin-style architecture: adding a tool = 1 file + 2 lines in the registry
- Custom exception hierarchy with defensive parsing (raises on malformed Laya output)
- Configuration via `LAYAMCP_*` env vars (HOST, PORT, PRELOAD_MODELS, LOG_LEVEL)
- Pytest test suite (33+ tests, mocked bridges — no GPU needed)
- AGENTS.md for AI coding agent instructions
- docs/ folder: ARCHITECTURE, TOOLS, CONFIGURATION, DEVELOPMENT, DEPLOYMENT, TROUBLESHOOTING
- integrations/lapis-memory-save-classified.ts — spec for LaPis-side integration
