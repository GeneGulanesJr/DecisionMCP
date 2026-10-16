# Configuration

All settings are env-overridable via the `DECISIONMCP_` prefix. Defaults are defined in `decision_mcp/config.py`.

## Settings reference

| Setting                    | Env var                           | Default       | Description                                                                                  |
| -------------------------- | --------------------------------- | ------------- | -------------------------------------------------------------------------------------------- |
| `host`                     | `DECISIONMCP_HOST`                | `127.0.0.1`   | Bind address. Use `0.0.0.0` for network access (behind a reverse proxy with auth).          |
| `port`                     | `DECISIONMCP_PORT`                | `8765`        | HTTP port. Avoid conflicts with common services (5432 Postgres, 6379 Redis, 8080 alt-http). |
| `preload_models`           | `DECISIONMCP_PRELOAD_MODELS`      | `true`        | Load model weights at startup. `false` defers to first call (slower first request).         |
| `engine`                   | `DECISIONMCP_ENGINE`              | `laya`        | Which decision engine serves inference: `laya` (default), `d1` ([LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) via `transformers`, install `.[d1]`) or `d1-omni` ([d1-omni-600M-MLX-8bit](https://huggingface.co/LiquidAI/d1-omni-600M-MLX-8bit) via `mlx-vlm`, Apple Silicon, install `.[d1-omni]`). See [Engines](#engines). |
| `log_level`                | `DECISIONMCP_LOG_LEVEL`           | `INFO`        | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                 |
| `data_dir`                 | `DECISIONMCP_DATA_DIR`            | `<project>/data` | Where the usage database (`usage.db`) and exports (`exports/`) live. Gitignored.        |
| `usage_enabled`            | `DECISIONMCP_USAGE_ENABLED`       | `true`        | Log every prediction + MCP session to SQLite (for training data). `false` disables it.      |
| `usage_store_text`         | `DECISIONMCP_USAGE_STORE_TEXT`    | `false`       | Also store the raw input text. Off by default: inputs can contain prompts, code and secrets. Never applied to `decision_secret_risk`. |
| `allow_updates`            | `DECISIONMCP_ALLOW_UPDATES`       | `false`       | Lets `decision_update` `apply` run `pip install -U laya` and re-download models. The server has no auth, so keep it off unless you need it. |
| `cors_origins`             | `DECISIONMCP_CORS_ORIGINS`        | *(empty)*     | Comma-separated origins whose pages may call `/mcp` from JavaScript, e.g. `http://localhost:8000`. Empty sends no CORS headers. Wildcards are refused. See [Browser access](#browser-access-cors). |

## Engines

`DECISIONMCP_ENGINE` picks which registered engine serves inference (registry: `decision_mcp/engines/__init__.py`):

| Engine | Extra     | Model                                                                  | Notes                                                                                       |
| ------ | --------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `laya` | `.[laya]` | `convaiinnovations/laya` (3 checkpoints in `./models`)                 | Default. Picks the checkpoint per call by script detection.                                 |
| `d1`   | `.[d1]`   | [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) (3.1B, ~6 GB)  | One multimodal decision model via `transformers>=5.14` (`trust_remote_code`). cuda → mps → cpu. |
| `d1-omni` | `.[d1-omni]` | [LiquidAI/d1-omni-600M-MLX-8bit](https://huggingface.co/LiquidAI/d1-omni-600M-MLX-8bit) (600M, 0.73 GB) | 8-bit MLX export via `mlx-vlm>=0.7.8`. Apple Silicon only. Same presets as `d1`. |

Both ship the same five presets (guard / route / moderate / triage / email), so the 5 preset tools behave identically; the custom-question tools work with any engine by construction. An unknown name fails at startup with the list of valid engines.

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

## Browser access (CORS)

A web page can only read replies from this server if the server names the page's origin in its CORS headers. By default it names none, so no page can call it, whatever site the user has open. That default matters because the server has no auth.

To let a page you serve yourself call the tools (for example a local demo on `http://localhost:8000`), list its exact origin:

```bash
DECISIONMCP_CORS_ORIGINS=http://localhost:8000 decisionmcp
DECISIONMCP_CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000 decisionmcp
```

- An origin is scheme, host and port, with no path and no trailing slash. `http://localhost:8000` and `http://127.0.0.1:8000` are different origins.
- `*` and any other wildcard are refused at start-up.
- Allowed methods: `GET`, `POST`, `DELETE`, `OPTIONS`. Allowed request headers: `content-type`, `accept`, `mcp-session-id`, `mcp-protocol-version`, `last-event-id`. The `mcp-session-id` response header is exposed so page scripts can read it. Credentials (cookies) are not allowed.
- CORS does not stop non-browser clients. Anything that can reach the port can still call the tools, so keep the bind address at `127.0.0.1`.

From the page, speak streamable HTTP to `/mcp`: `POST` `initialize`, read the `mcp-session-id` response header, `POST` `notifications/initialized`, then `POST` `tools/call` with that header. Every `POST` needs `Content-Type: application/json` and `Accept: application/json, text/event-stream`. A tool result arrives as `result.content[0].text`, a JSON string.

## Pre-loading vs lazy-loading models

| `DECISIONMCP_PRELOAD_MODELS` | Behaviour                                                              | Trade-off                                                                                    |
| ------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `true` (default)         | Model weights load at process start                                    | First user request is fast (~33 ms). Process startup is slow (~2–5 s).                      |
| `false`                  | Model weights load on first tool call                                  | Process startup is fast. First tool call is slow (~2–5 s). Better for dev / batch jobs.    |

## Production checklist

- [ ] Bind to `127.0.0.1` unless you need network access.
- [ ] Put behind reverse proxy with auth if exposing externally.
- [ ] Leave `DECISIONMCP_CORS_ORIGINS` empty unless a page you control must call the server; list exact origins only.
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
