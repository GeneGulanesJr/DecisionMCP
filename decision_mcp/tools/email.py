"""Tool: email-specific triage."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import choice_of, require_dict, score_level, yes_prob
from .base import Tool


class EmailInput(BaseModel):
    body: str = Field(..., description="Raw email body to triage.")


class EmailOutput(BaseModel):
    category: str = Field(..., description="Team that should handle it (billing / technical / sales / ...).")
    urgency: str
    needs_reply: bool
    is_spam: bool
    is_phishing: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class EmailTool(Tool):
    name = "decision_email"
    description = (
        "Triage an email: owning team (category), urgency, whether it needs a reply, "
        "and whether it is spam or phishing."
    )
    input_schema = EmailInput
    output_schema = EmailOutput

    async def run(self, bridge: DecisionBridge, body: str) -> EmailOutput:
        raw = bridge.predict(body, preset="email")
        require_dict(raw, self.name)
        category, confidence = choice_of(raw, "category", self.name)  # primary signal
        _, urgency, _ = score_level(raw, "urgency", self.name)
        return EmailOutput(
            category=category,
            urgency=urgency,
            needs_reply=yes_prob(raw, "needs_reply", self.name) >= 0.5,
            is_spam=yes_prob(raw, "is_spam", self.name) >= 0.5,
            is_phishing=yes_prob(raw, "is_phishing", self.name) >= 0.5,
            confidence=confidence,
            details=raw,
        )
