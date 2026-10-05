# Documentation index

| Doc | What's in it |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layer diagram, component responsibilities, data flow per tool call, performance characteristics, security notes, extensibility points |
| [TOOLS.md](TOOLS.md) | Every tool documented: input/output schemas, when to use, costs, limitations, label-normalisation rules, parser locations |
| [CONFIGURATION.md](CONFIGURATION.md) | All `DECISIONMCP_*` env vars, `.env` file format, runtime overrides, pre-load vs lazy-load, production checklist |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Setup, project layout, the canonical "add a tool" pattern, custom presets, real-engine verification, code style, release checklist |
| [DEPLOYMENT.md](DEPLOYMENT.md) | systemd, Windows NSSM, Dockerfile + docker-compose, nginx/Caddy reverse proxy, health check, log shipping, resource sizing, first-run model download |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Server-won't-start, wrong tool shape, tests failing, Pi can't connect, slow first call, high memory / CUDA OOM |

Start with **[ARCHITECTURE.md](ARCHITECTURE.md)** if you're new to the project, or **[DEVELOPMENT.md](DEVELOPMENT.md)** if you're here to add a tool.
