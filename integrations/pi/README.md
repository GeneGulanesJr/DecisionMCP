# Pi integration

## layamcp-lifecycle.ts — start/stop the server with Pi

Starts LayaMCP with the **first** Pi session, keeps it while **any** Pi
session is open, closes it with the **last** one. Multiple concurrent Pi
instances share one server via a refcount file (`data/layamcp-refs.json`).

Install (symlink so repo edits are picked up by `/reload`):

```bash
ln -sf "$PWD/layamcp-lifecycle.ts" ~/.pi/agent/extensions/layamcp-lifecycle.ts
```

Notes:

- Cold start (first Pi of the day) waits for model preload (~10–40 s) so the
  MCP bridge connects cleanly; warm starts are instant.
- A server you started manually has no pidfile from us and is never killed.
- On last close the extension runs `scripts/backup_usage.py` once.
- This mode replaces the launchd *server* agent (`com.gulaneskorp.layamcp`);
  the nightly *backup* agent can stay as a safety net.
