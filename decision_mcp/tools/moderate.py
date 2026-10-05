"""Tool: content moderation (toxicity / harassment / threats)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import require_dict, yes_prob
from .base import Tool


class ModerateInput(BaseModel):
    text: str = Field(..., description="Text to moderate.")


class ModerateOutput(BaseModel):
    is_toxic: bool
    is_harassment: bool
    is_threat: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class ModerateTool(Tool):
    name = "decision_moderate"
    description = "Check text for toxicity, harassment, and threats."
    input_schema = ModerateInput
    output_schema = ModerateOutput

    async def run(self, bridge: DecisionBridge, text: str) -> ModerateOutput:
        raw = bridge.predict(text, preset="moderate")
        require_dict(raw, self.name)
        probs = [yes_prob(raw, k, self.name) for k in ("toxic", "harassment", "threat")]
        toxic, harassment, threat = (p >= 0.5 for p in probs)
        return ModerateOutput(
            is_toxic=toxic,
            is_harassment=harassment,
            is_threat=threat,
            # Weakest of the three verdicts: all three are at least this sure.
            confidence=min(max(p, 1.0 - p) for p in probs),
            details=raw,
        )
