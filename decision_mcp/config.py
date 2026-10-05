"""Settings for DecisionMCP, env-overridable via DECISIONMCP_* prefix."""
from __future__ import annotations

from pathlib import Path

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

    # Usage / session log (SQLite) used to collect data for training.
    data_dir: Path = _PROJECT_ROOT / "data"
    usage_enabled: bool = True
    # Off by default: inputs can contain prompts, code and secrets. Turn on to make
    # the exported JSONL usable as (input, answers) training pairs.
    usage_store_text: bool = False

    # Lets the decision_update tool run `pip install -U laya` and refresh model
    # weights. Off by default because the HTTP server has no auth.
    allow_updates: bool = False


settings = Settings()
