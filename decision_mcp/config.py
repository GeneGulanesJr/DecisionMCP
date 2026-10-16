"""Settings for DecisionMCP, env-overridable via DECISIONMCP_* prefix."""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Override via env or .env file.

    Examples:
        DECISIONMCP_PORT=9000 decisionmcp
        DECISIONMCP_PRELOAD_MODELS=false decisionmcp
    """

    model_config = SettingsConfigDict(
        env_prefix="DECISIONMCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8765
    preload_models: bool = True
    log_level: str = "INFO"

    # Which decision engine serves inference: a key from
    # decision_mcp.engines.ENGINES ("laya", "d1" or "d1-omni").
    engine: str = "laya"

    # Usage / session log (SQLite) used to collect data for training.
    data_dir: Path = _PROJECT_ROOT / "data"
    usage_enabled: bool = True
    # Off by default: inputs can contain prompts, code and secrets. Turn on to make
    # the exported JSONL usable as (input, answers) training pairs.
    usage_store_text: bool = False

    # Lets the decision_update tool run `pip install -U laya` and refresh model
    # weights. Off by default because the HTTP server has no auth.
    allow_updates: bool = False

    # Browser access. Comma-separated origins (scheme://host[:port]) whose pages
    # may call /mcp from JavaScript. Empty (the default) sends no CORS headers at
    # all, so browsers block every cross-origin page. "*" is refused: the HTTP
    # server has no auth, and a wildcard would let any site the user visits call it.
    cors_origins: str = ""

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard_origin(cls, v: str) -> str:
        for origin in (o.strip() for o in v.split(",")):
            if not origin:
                continue
            if "*" in origin:
                raise ValueError(
                    "DECISIONMCP_CORS_ORIGINS must list exact origins; wildcards are refused "
                    "because the server has no auth"
                )
            if not origin.startswith(("http://", "https://")) or origin.endswith("/"):
                raise ValueError(
                    f"DECISIONMCP_CORS_ORIGINS entry {origin!r} must look like "
                    "http://localhost:8000 (scheme and host, no trailing slash)"
                )
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        """``cors_origins`` as a list, blanks dropped."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
