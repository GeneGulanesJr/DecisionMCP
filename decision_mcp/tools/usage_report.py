"""Tool: usage statistics and training-data export from the usage log."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ..errors import ToolError
from .base import Tool


class UsageInput(BaseModel):
    action: Literal["stats", "export"] = Field(
        "stats",
        description=(
            "'stats' summarises logged calls. 'export' writes successful calls to a JSONL "
            "file under the data dir (input is null unless DECISIONMCP_USAGE_STORE_TEXT=true)."
        ),
    )
    since_hours: float | None = Field(
        None, gt=0, description="Only include calls from the last N hours."
    )
    labeled_only: bool = Field(
        False,
        description=(
            "Export only rows that carry input text (ready training pairs). "
            "Requires DECISIONMCP_USAGE_STORE_TEXT=true at call time. Export only."
        ),
    )


class UsageOutput(BaseModel):
    stats: dict[str, Any]
    export_path: str | None = None
    exported_rows: int | None = None


class UsageTool(Tool):
    name = "decision_usage"
    description = (
        "Summarise logged decision calls (per tool, model, session, latency) or export them "
        "as JSONL for training. Text is only kept when DECISIONMCP_USAGE_STORE_TEXT=true."
    )
    input_schema = UsageInput
    output_schema = UsageOutput

    async def run(
        self,
        bridge: DecisionBridge,
        action: str = "stats",
        since_hours: float | None = None,
        labeled_only: bool = False,
    ) -> UsageOutput:
        store = bridge.usage
        if store is None:
            raise ToolError(self.name, "Usage logging is disabled (DECISIONMCP_USAGE_ENABLED=false).")
        since = time.time() - since_hours * 3600 if since_hours else None
        stats = store.stats(since)
        if action != "export":
            return UsageOutput(stats=stats)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "-labeled" if labeled_only else ""
        path = store.path.parent / "exports" / f"usage{suffix}-{stamp}.jsonl"
        rows = store.export_jsonl(path, since, labeled_only=labeled_only)
        return UsageOutput(stats=stats, export_path=str(path), exported_rows=rows)
