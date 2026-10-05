"""Tool: bug report severity + area classifier."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import choice_of, require_dict
from .base import Tool

_STATE_KEY = "report"

_BUG_QUESTIONS = {
    "severity": {
        "type": "choice",
        "instructions": "How severe is the bug described in `report`?",
        "criteria": {
            "S0_critical": "outage, data loss, security hole or everyone is blocked",
            "S1_high": "a major feature is broken for many users, no good workaround",
            "S2_medium": "a feature misbehaves for some users, workaround exists",
            "S3_low": "cosmetic issue or typo, easy workaround",
        },
    },
    "area": {
        "type": "choice",
        "instructions": "Which part of the system does the bug in `report` belong to?",
        "criteria": {
            "frontend": "UI, browser, styling, client-side code",
            "backend": "server logic, APIs, databases, business rules",
            "infra": "deployment, CI/CD, networking, hosting, performance at scale",
            "docs": "documentation, README, comments, typos in text",
            "tests": "test suite, flaky tests, test tooling",
            "deps": "third-party dependencies, version conflicts, packaging",
            "auth": "login, permissions, sessions, tokens, access control",
            "unknown": "cannot tell from the report",
        },
    },
}


class BugSeverityInput(BaseModel):
    text: str = Field(..., description="A bug report or issue text to classify.")


class BugSeverityOutput(BaseModel):
    severity: str  # S0_critical | S1_high | S2_medium | S3_low
    area: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class BugSeverityTool(Tool):
    name = "decision_bug_severity"
    description = (
        "Classify a bug report's severity (S0_critical / S1_high / S2_medium / S3_low) "
        "and area (frontend / backend / infra / docs / tests / deps / auth / unknown)."
    )
    input_schema = BugSeverityInput
    output_schema = BugSeverityOutput

    async def run(self, bridge: DecisionBridge, text: str) -> BugSeverityOutput:
        raw = bridge.predict_custom(text, questions=_BUG_QUESTIONS, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        severity, confidence = choice_of(raw, "severity", self.name)
        area, _ = choice_of(raw, "area", self.name)
        return BugSeverityOutput(
            severity=severity,
            area=area,
            confidence=confidence,
            details=raw,
        )
