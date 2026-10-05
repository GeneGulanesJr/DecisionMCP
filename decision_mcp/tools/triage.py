"""Tool: support ticket triage (intent / urgency / churn / frustration)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import choice_of, require_dict, score_level, yes_prob
from .base import Tool


class TriageInput(BaseModel):
    text: str = Field(..., description="Ticket text to triage.")


class TriageOutput(BaseModel):
    intent: str
    is_urgent: bool
    churn_risk: bool
    refund_requested: bool
    frustration: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class TriageTool(Tool):
    name = "decision_triage"
    description = (
        "Classify a support ticket's intent, urgency, churn risk, refund request, "
        "and frustration level."
    )
    input_schema = TriageInput
    output_schema = TriageOutput

    async def run(self, bridge: DecisionBridge, text: str) -> TriageOutput:
        raw = bridge.predict(text, preset="triage")
        require_dict(raw, self.name)
        intent, confidence = choice_of(raw, "intent", self.name)  # primary signal
        _, frustration, _ = score_level(raw, "frustration", self.name)
        return TriageOutput(
            intent=intent,
            is_urgent=yes_prob(raw, "is_urgent", self.name) >= 0.5,
            churn_risk=yes_prob(raw, "churn_risk", self.name) >= 0.5,
            refund_requested=yes_prob(raw, "refund_requested", self.name) >= 0.5,
            frustration=frustration,
            confidence=confidence,
            details=raw,
        )
