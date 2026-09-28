"""Tool: PR diff intent classifier — what is this diff trying to do?"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import LayaBridge
from ._helpers import choice_of, require_dict
from .base import Tool

_STATE_KEY = "diff"

_DIFF_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the `diff` trying to achieve?",
        "criteria": {
            "add_feature": "adds new functionality",
            "fix_bug": "fixes incorrect behaviour",
            "refactor": "restructures code without changing behaviour",
            "perf": "makes something faster or lighter",
            "docs": "changes documentation or comments only",
            "test": "adds or changes tests only",
            "build": "changes build, packaging or dependencies",
            "chore": "maintenance or housekeeping",
            "revert": "undoes an earlier change",
        },
    },
    "scope": {
        "type": "choice",
        "instructions": "How widely does the `diff` reach across the codebase?",
        "criteria": {
            "single_file": "confined to one file",
            "module": "several files in one module or feature",
            "cross_cutting": "touches many modules or shared infrastructure",
        },
    },
    "risk": {
        "type": "choice",
        "instructions": "How likely is the `diff` to break something in production?",
        "criteria": {
            "low": "additive, isolated or non-functional change",
            "medium": "changes existing behaviour in one area",
            "high": "sweeping, security-sensitive or data-affecting change",
        },
    },
}


class DiffIntentInput(BaseModel):
    diff: str = Field(..., description="A unified diff to classify.")


class DiffIntentOutput(BaseModel):
    intent: str
    scope: str
    risk: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class DiffIntentTool(Tool):
    name = "laya_diff_intent"
    description = (
        "Classify a PR diff: intent (add_feature / fix_bug / refactor / perf / docs / "
        "test / build / chore / revert), scope (single_file / module / cross_cutting), "
        "and risk (low / medium / high)."
    )
    input_schema = DiffIntentInput
    output_schema = DiffIntentOutput

    async def run(self, bridge: LayaBridge, diff: str) -> DiffIntentOutput:
        raw = bridge.predict_custom(diff, questions=_DIFF_QUESTIONS, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        intent, confidence = choice_of(raw, "intent", self.name)
        scope, _ = choice_of(raw, "scope", self.name)
        risk, _ = choice_of(raw, "risk", self.name)
        return DiffIntentOutput(
            intent=intent,
            scope=scope,
            risk=risk,
            confidence=confidence,
            details=raw,
        )
