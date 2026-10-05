# Configuration

All settings are env-overridable via the `DECISIONMCP_` prefix. Defaults are defined in `decision_mcp/config.py`.

## Settings reference

| Setting                    | Env var                           | Default       | Description                                                                                  |
| -------------------------- | --------------------------------- | ------------- | -------------------------------------------------------------------------------------------- |
| `host`                     | `DECISIONMCP_HOST`                | `127.0.0.1`   | Bind address. Use `0.0.0.0` for network access (behind a reverse proxy with auth).          |
| `port`                     | `DECISIONMCP_PORT`                | `8765`        | HTTP port. Avoid conflicts with common services (5432 Postgres, 6379 Redis, 8080 alt-http). |
| `preload_models`           | `DECISIONMCP_PRELOAD_MODELS`      | `true`        | Load model weights at startup. `false` defers to first call (slower first request).         |
| `log_level`                | `DECISIONMCP_LOG_LEVEL`           | `INFO`        | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                 |
| `data_dir`                 | `DECISIONMCP_DATA_DIR`            | `<project>/data` | Where the usage database (`usage.db`) and exports (`exports/`) live. Gitignored.        |
| `usage_enabled`            | `DECISIONMCP_USAGE_ENABLED`       | `true`        | Log every prediction + MCP session to SQLite (for training data). `false` disables it.      |
| `usage_store_text`         | `DECISIONMCP_USAGE_STORE_TEXT`    | `false`       | Also store the raw input text. Off by default: inputs can contain prompts, code and secrets. Never applied to `decision_secret_risk`. |
| `allow_updates`            | `DECISIONMCP_ALLOW_UPDATES`       | `false`       | Lets `decision_update` `apply` run `pip install -U laya` and re-download models. The server has no auth, so keep it off unless you need it. |

## Model location

Model weights are stored in `<project>/models` (gitignored). `decision_mcp/__init__.py` sets `HF_HUB_CACHE` to that folder before Hugging Face is imported, unless you already set `HF_HUB_CACHE` or `HF_HOME` yourself (then yours wins). This is not a `DECISIONMCP_*` setting because it has to take effect before any imports.

Set `HF_HUB_OFFLINE=1` to guarantee the server never reaches the network once the models are downloaded (note that `decision_update` then can't check for updates).

## `.env` file

Create `.env` in the project root:

```bash
DECISIONMCP_HOST=127.0.0.1
DECISIONMCP_PORT=8765
DECISIONMCP_PRELOAD_MODELS=true
DECISIONMCP_LOG_LEVEL=INFO
```

Pydantic-settings auto-loads it. The `.env` file is gitignored — safe for local overrides.

See `.env.example` for the canonical template.

> **Migrating from LayaMCP:** the env prefix changed from `LAYAMCP_` to
> `DECISIONMCP_`. Rename the variables in your `.env` / service definitions —
> unknown `LAYAMCP_*` variables are ignored (`extra="ignore"`).

## Override at runtime

```bash
DECISIONMCP_PORT=9000 decisionmcp
DECISIONMCP_LOG_LEVEL=DEBUG decisionmcp
DECISIONMCP_HOST=0.0.0.0 DECISIONMCP_PORT=8765 decisionmcp
```

## Network access (multi-client)

By default, the server binds to `127.0.0.1` (localhost only). To expose to other machines on your network:

```bash
DECISIONMCP_HOST=0.0.0.0 decisionmcp
```

⚠️ **No built-in auth.** Don't expose to the public internet without putting a reverse proxy with auth in front (see `docs/DEPLOYMENT.md`).

## Pre-loading vs lazy-loading models

| `DECISIONMCP_PRELOAD_MODELS` | Behaviour                                                              | Trade-off                                                                                    |
| ------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `true` (default)         | Model weights load at process start                                    | First user request is fast (~33 ms). Process startup is slow (~2–5 s).                      |
| `false`                  | Model weights load on first tool call                                  | Process startup is fast. First tool call is slow (~2–5 s). Better for dev / batch jobs.    |

## Production checklist

- [ ] Bind to `127.0.0.1` unless you need network access.
- [ ] Put behind reverse proxy with auth if exposing externally.
- [ ] Set `DECISIONMCP_LOG_LEVEL=WARNING` (or `ERROR`) for production.
- [ ] Use a process manager (systemd / Docker / supervisord) so it restarts on crash.
- [ ] Pre-load models (`DECISIONMCP_PRELOAD_MODELS=true`) so the first user request is fast.
- [ ] Monitor `/health` endpoint (see `docs/DEPLOYMENT.md`).
- [ ] Send logs to a log aggregator (Vector, Loki, Datadog).

## Adding new settings

1. Add a field to `Settings` in `decision_mcp/config.py`.
2. Use it via `settings.<field>` anywhere.
3. Document it here.

Pydantic-settings will pick up the matching `DECISIONMCP_<FIELD_UPPER>` env var automatically.
