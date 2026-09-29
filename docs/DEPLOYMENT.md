# Deployment

## Local dev

```bash
pip install -e ".[dev]"
layamcp
```

Runs on `http://127.0.0.1:8765` (default).

## macOS — launchd (optional) + Pi lifecycle extension

Two ways to run the server on macOS; pick one for the server itself. The
nightly backup agent is independent of both.

**Mode A — start/stop with Pi (default, active).** The
[`integrations/pi/layamcp-lifecycle.ts`](../integrations/pi/layamcp-lifecycle.ts)
extension starts the server with the first Pi session, shares one instance
across concurrent Pi sessions (refcount file + lockfile under `data/`), and
stops it — after a final backup — when the last session closes. Install:

```bash
ln -sf "$PWD/integrations/pi/layamcp-lifecycle.ts" ~/.pi/agent/extensions/layamcp-lifecycle.ts
```

Cold start (first Pi of the day) waits ~10–40 s for model preload; warm starts
are instant. See `integrations/pi/README.md`.

**Mode B — always-on via launchd.** For a server that runs regardless of Pi:

| Agent | What it does |
| --- | --- |
| `com.gulaneskorp.layamcp` | Runs `.venv/bin/layamcp` on login, restarts it if it crashes (`KeepAlive`), logs to `data/logs/layamcp.log` |

```bash
cp deploy/launchd/com.gulaneskorp.layamcp.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gulaneskorp.layamcp.plist
```

If you later switch to Mode A: `launchctl bootout gui/$(id -u)/com.gulaneskorp.layamcp`
(KeepAlive would otherwise resurrect the server after the lifecycle extension
stops it). Running both modes at once is harmless but redundant.

**Nightly backup agent (keep either way).** At 03:00, runs
`scripts/backup_usage.py`: WAL-safe SQLite snapshot to `data/backups/usage-<ts>.db`,
keeps the last 14; a missing DB (fresh install) is a clean no-op.

```bash
cp deploy/launchd/com.gulaneskorp.layamcp-backup.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.gulaneskorp.layamcp-backup.plist
```

Operate:

```bash
launchctl kickstart gui/$(id -u)/com.gulaneskorp.layamcp-backup      # backup now
launchctl print gui/$(id -u)/com.gulaneskorp.layamcp-backup | head   # state
```

Manual backups any time: `.venv/bin/python scripts/backup_usage.py --keep 14`.

## Linux — systemd

`/etc/systemd/system/layamcp.service`:

```ini
[Unit]
Description=LayaMCP server
After=network.target

[Service]
Type=simple
User=layamcp
WorkingDirectory=/opt/LayaMCP
Environment="LAYAMCP_HOST=127.0.0.1"
Environment="LAYAMCP_PORT=8765"
Environment="LAYAMCP_LOG_LEVEL=WARNING"
ExecStart=/opt/LayaMCP/.venv/bin/layamcp
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now layamcp
sudo systemctl status layamcp
sudo journalctl -u layamcp -f    # tail logs
```

## Windows — NSSM (Non-Sucking Service Manager)

```cmd
nssm install layamcp "C:\Users\you\LayaMCP\.venv\Scripts\layamcp.exe"
nssm set layamcp AppDirectory "C:\Users\you\LayaMCP"
nssm set layamcp AppEnvironmentExtra LAYAMCP_PORT=8765 LAYAMCP_LOG_LEVEL=WARNING
nssm start layamcp
nssm status layamcp
```

Or run as a scheduled task that restarts on exit.

## Docker

`Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Laya + deps. Laya pulls in torch + transformers + huggingface_hub,
# so this image is large (~2 GB).
COPY pyproject.toml .
RUN pip install --no-cache-dir . \
    && pip install "mcp[server]>=1.0"

COPY laya_mcp/ ./laya_mcp/

# HuggingFace cache — pre-bake model weights here if you want.
# Otherwise they're downloaded on first run inside the container.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8765

CMD ["python", "-m", "laya_mcp.server"]
```

`docker-compose.yml`:

```yaml
services:
  layamcp:
    build: .
    ports:
      - "127.0.0.1:8765:8765"
    environment:
      - LAYAMCP_HOST=0.0.0.0
      - LAYAMCP_LOG_LEVEL=WARNING
    volumes:
      - hf-cache:/root/.cache/huggingface
    restart: unless-stopped

volumes:
  hf-cache:
```

```bash
docker compose up -d
docker compose logs -f layamcp
```

## Reverse proxy with auth (if exposing externally)

nginx:

```nginx
server {
    listen 443 ssl;
    server_name layamcp.example.com;

    ssl_certificate /etc/letsencrypt/live/layamcp.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/layamcp.example.com/privkey.pem;

    # Token-based auth — clients send `Authorization: Bearer <token>`
    # Use a tool like oauth2-proxy or a simple Lua script.
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Optional: rate limit
        limit_req zone=layamcp burst=20 nodelay;
    }
}

# Rate limit zone (define at http level)
# limit_req_zone $binary_remote_addr zone=layamcp:10m rate=10r/s;
```

Caddy (simpler):

```caddy
layamcp.example.com {
    reverse_proxy 127.0.0.1:8765
    basicauth {
        admin $2a$14$<bcrypt-hash>
    }
}
```

⚠️ **The MCP server has no built-in auth.** Don't expose to the internet without putting a reverse proxy with auth in front.

## Health check

The FastAPI app exposes `/health` via the `mcp[server]` SDK:

```bash
curl http://127.0.0.1:8765/health
```

Add to monitoring / load balancer health checks.

## Logs

Logs go to stdout by default. Configure via `LAYAMCP_LOG_LEVEL`:

```bash
LAYAMCP_LOG_LEVEL=DEBUG layamcp 2>&1 | tee /var/log/layamcp.log
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

On first run, Laya downloads model checkpoints from HuggingFace to `~/.cache/huggingface/`. Requires:

- Network access to `huggingface.co`
- ~2 GB free disk space

If behind a firewall, pre-download:

```bash
huggingface-cli download convaiinnovations/laya
huggingface-cli download convaiinnovations/laya-multilingual
huggingface-cli download convaiinnovations/laya-typed-decisions
```

Then copy `~/.cache/huggingface/` to the target machine.