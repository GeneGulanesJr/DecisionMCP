"""Tool: test priority classifier — which tests to run first."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import choice_of, require_dict
from .base import Tool

_STATE_KEY = "test"

_TEST_QUESTIONS = {
    "priority": {
        "type": "choice",
        "instructions": "How early should the test described in `test` run in a CI pipeline?",
        "criteria": {
            "skip": "redundant or obsolete, not worth running",
            "low": "rarely useful, run last",
            "medium": "normal coverage",
            "high": "covers important or recently changed behaviour",
            "critical": "guards a security fix, a regression or core functionality, run first",
        },
    },
    "reason": {
        "type": "choice",
        "instructions": "Why does the test described in `test` have this priority?",
        "criteria": {
            "covers_new_code": "exercises newly added code",
            "covers_bug_fix": "verifies a recent bug or security fix",
            "covers_regression": "guards against a previously seen regression",
            "smoke_test": "quick sanity check of core functionality",
            "redundant": "duplicates another test",
            "flaky": "is unreliable or intermittently failing",
        },
    },
}


class TestPriorityInput(BaseModel):
    description: str = Field(
        ...,
        description="A test name + description (or full test file content) to prioritize.",
    )


class TestPriorityOutput(BaseModel):
    priority: str  # skip | low | medium | high | critical
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class TestPriorityTool(Tool):
    name = "decision_test_priority"
    description = (
        "Classify a test's run priority (skip / low / medium / high / critical) and the "
        "reason (covers_new_code / covers_bug_fix / covers_regression / smoke_test / "
        "redundant / flaky)."
    )
    input_schema = TestPriorityInput
    output_schema = TestPriorityOutput

    async def run(self, bridge: DecisionBridge, description: str) -> TestPriorityOutput:
        raw = bridge.predict_custom(description, questions=_TEST_QUESTIONS, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        priority, confidence = choice_of(raw, "priority", self.name)
        reason, _ = choice_of(raw, "reason", self.name)
        return TestPriorityOutput(
            priority=priority,
            reason=reason,
            confidence=confidence,
            details=raw,
        )
