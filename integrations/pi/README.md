# Pi integration

## decisionmcp-lifecycle.ts — start/stop the server with Pi

Starts DecisionMCP with the **first** Pi session, keeps it while **any** Pi
session is open, closes it with the **last** one. Multiple concurrent Pi
instances share one server via a refcount file (`data/decisionmcp-refs.json`).

Install (symlink so repo edits are picked up by `/reload`):

```bash
ln -sf "$PWD/decisionmcp-lifecycle.ts" ~/.pi/agent/extensions/decisionmcp-lifecycle.ts
```

Notes:

- Cold start (first Pi of the day) waits for model preload (~10–40 s) so the
  MCP bridge connects cleanly; warm starts are instant.
- A server you started manually has no pidfile from us and is never killed.
- On last close the extension runs `scripts/backup_usage.py` once.
- This mode replaces the launchd *server* agent (`com.gulaneskorp.decisionmcp`);
  the nightly *backup* agent can stay as a safety net.
