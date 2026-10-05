# Deployment

## Local dev

```bash
pip install -e ".[dev,laya]"
decisionmcp
```

Runs on `http://127.0.0.1:8765` (default).

## macOS — launchd (optional) + Pi lifecycle extension

Two ways to run the server on macOS; pick one for the server itself. The
nightly backup agent is independent of both.

**Mode A — start/stop with Pi (default, active).** The
[`integrations/pi/decisionmcp-lifecycle.ts`](../integrations/pi/decisionmcp-lifecycle.ts)
extension starts the server with the first Pi session, shares one instance
across concurrent Pi sessions (refcount file + lockfile under `data/`), and
stops it — after a final backup — when the last session closes. Install:

```bash
ln -sf "$PWD/integrations/pi/decisionmcp-lifecycle.ts" ~/.pi/agent/extensions/decisionmcp-lifecycle.ts
```

Cold start (first Pi of the day) waits ~10–40 s for model preload; warm starts
are instant. See `integrations/pi/README.md`.

**Mode B — always-on via launchd.** For a server that runs regardless of Pi:

| Agent | What it does |
| --- | --- |
| `com.gulaneskorp.decisionmcp` | Runs `.venv/bin/decisionmcp` on login, restarts it if it crashes (`KeepAlive`), logs to `data/logs/decisionmcp.log` |

```bash
cp deploy/launchd/com.gulaneskorp.decisionmcp.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gulaneskorp.decisionmcp.plist
```

If you later switch to Mode A: `launchctl bootout gui/$(id -u)/com.gulaneskorp.decisionmcp`
(KeepAlive would otherwise resurrect the server after the lifecycle extension
stops it). Running both modes at once is harmless but redundant.

**Nightly backup agent (keep either way).** At 03:00, runs
`scripts/backup_usage.py`: WAL-safe SQLite snapshot to `data/backups/usage-<ts>.db`,
keeps the last 14; a missing DB (fresh install) is a clean no-op.

```bash
cp deploy/launchd/com.gulaneskorp.decisionmcp-backup.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gulaneskorp.decisionmcp-backup.plist
```

Operate:

```bash
launchctl kickstart gui/$(id -u)/com.gulaneskorp.decisionmcp-backup      # backup now
launchctl print gui/$(id -u)/com.gulaneskorp.decisionmcp-backup | head   # state
```

Manual backups any time: `.venv/bin/python scripts/backup_usage.py --keep 14`.

## Linux — systemd

`/etc/systemd/system/decisionmcp.service`:

```ini
[Unit]
Description=DecisionMCP server
After=network.target

[Service]
Type=simple
User=decisionmcp
WorkingDirectory=/opt/DecisionMCP
Environment="DECISIONMCP_HOST=127.0.0.1"
Environment="DECISIONMCP_PORT=8765"
Environment="DECISIONMCP_LOG_LEVEL=WARNING"
ExecStart=/opt/DecisionMCP/.venv/bin/decisionmcp
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now decisionmcp
sudo systemctl status decisionmcp
sudo journalctl -u decisionmcp -f    # tail logs
```

## Windows — NSSM (Non-Sucking Service Manager)

```cmd
nssm install decisionmcp "C:\Users\you\DecisionMCP\.venv\Scripts\decisionmcp.exe"
nssm set decisionmcp AppDirectory "C:\Users\you\DecisionMCP"
nssm set decisionmcp AppEnvironmentExtra DECISIONMCP_PORT=8765 DECISIONMCP_LOG_LEVEL=WARNING
nssm start decisionmcp
nssm status decisionmcp
```

Or run as a scheduled task that restarts on exit.

## Docker

`Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install DecisionMCP + the Laya engine. The engine pulls in torch +
# transformers + huggingface_hub, so this image is large (~2 GB).
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[laya]" \
    && pip install "mcp[server]>=1.0"

COPY decision_mcp/ ./decision_mcp/

# HuggingFace cache — pre-bake model weights here if you want.
# Otherwise they're downloaded on first run inside the container.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8765

CMD ["python", "-m", "decision_mcp.server"]
```

`docker-compose.yml`:

```yaml
services:
  decisionmcp:
    build: .
    ports:
      - "127.0.0.1:8765:8765"
    environment:
      - DECISIONMCP_HOST=0.0.0.0
      - DECISIONMCP_LOG_LEVEL=WARNING
    volumes:
      - hf-cache:/root/.cache/huggingface
    restart: unless-stopped

volumes:
  hf-cache:
```

```bash
docker compose up -d
docker compose logs -f decisionmcp
```

## Reverse proxy with auth (if exposing externally)

nginx:

```nginx
server {
    listen 443 ssl;
    server_name decisionmcp.example.com;

    ssl_certificate /etc/letsencrypt/live/decisionmcp.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/decisionmcp.example.com/privkey.pem;

    # Token-based auth — clients send `Authorization: Bearer <token>`
    # Use a tool like oauth2-proxy or a simple Lua script.
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Optional: rate limit
        limit_req zone=decisionmcp burst=20 nodelay;
    }
}

# Rate limit zone (define at http level)
# limit_req_zone $binary_remote_addr zone=decisionmcp:10m rate=10r/s;
```

Caddy (simpler):

```caddy
decisionmcp.example.com {
    reverse_proxy 127.0.0.1:8765
    basicauth {
        admin $2a$14$<bcrypt-hash>
    }
}
```

⚠️ **The MCP server has no built-in auth.** Don't expose to the internet without putting a reverse proxy with auth in front.

## Health check

The FastAPI app exposes `/health`:

```bash
curl http://127.0.0.1:8765/health
```

Add to monitoring / load balancer health checks.

## Logs

Logs go to stdout by default. Configure via `DECISIONMCP_LOG_LEVEL`:

```bash
DECISIONMCP_LOG_LEVEL=DEBUG decisionmcp 2>&1 | tee /var/log/decisionmcp.log
```

For production, send to a log aggregator:
- **Vector** → Loki / Elasticsearch / S3
- **journald** (systemd) → Loki / journalbeat
- **Docker** → driver to your aggregator

## Resource sizing

| Resource    | Minimum      | Recommended (production) |
| ----------- | ------------ | ------------------------ |
| CPU         | 1 core       | 2+ cores                 |
| RAM         | 4 GB         | 8+ GB                    |
| Disk        | 5 GB         | 10+ GB                   |
| GPU         | none (CPU)   | T4 / L4 / equivalent     |
| Network     | localhost    | 100 Mbps+                |

**Why so much disk?** Model weights are ~500 MB–1 GB. HuggingFace cache holds them.

## First-run model download

On first run, the Laya engine downloads model checkpoints from HuggingFace to
the project's `./models` cache. Requires:

- Network access to `huggingface.co`
- ~2 GB free disk space

If behind a firewall, pre-download:

```bash
HF_HUB_CACHE="$PWD/models" huggingface-cli download convaiinnovations/laya
HF_HUB_CACHE="$PWD/models" huggingface-cli download convaiinnovations/laya-multilingual
HF_HUB_CACHE="$PWD/models" huggingface-cli download convaiinnovations/laya-typed-decisions
```
