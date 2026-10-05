# Pi integration

## decisionmcp-lifecycle.ts — start/stop the server with Pi

Starts DecisionMCP with the **first** Pi session, keeps it while **any** Pi
session is open, closes it with the **last** one. Multiple concurrent Pi
instances share one server via a refcount file (`data/decisionmcp-refs.json`).

Install (symlink so repo edits are picked up by `/reload`):

```bash
ln -sf "$PWD/decisionmcp-lifecycle.ts" ~/.pi/agent/extensions/decisionmcp-lifecycle.ts
```

If an old `layamcp-lifecycle.ts` symlink exists (pre-rename), remove it —
it dangles after the `decision_mcp` rename:

```bash
rm -f ~/.pi/agent/extensions/layamcp-lifecycle.ts
```

## MCP client config (pi ≥ 1.0 built-in)

`~/.pi/agent/mcp.json` — the built-in client speaks **streamable HTTP**:

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

Verify with `pi mcp list` (expect `decisionmcp: connected, 13 tools`). Tools
are `mcp__decisionmcp__<tool>`. The legacy `pi-mcp-extension` bridge (SSE at
`/sse`) is no longer needed; the server still serves `/sse` for other old
clients.

Notes:

- Cold start (first Pi of the day) waits for model preload (~10–40 s) so the
  MCP bridge connects cleanly; warm starts are instant.
- A server you started manually has no pidfile from us and is never killed.
- On last close the extension runs `scripts/backup_usage.py` once.
- This mode replaces the launchd *server* agent (`com.gulaneskorp.layamcp`,
  disabled); the nightly *backup* agent can stay as a safety net.
