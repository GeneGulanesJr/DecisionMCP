"""Tool: commit message classifier — type, scope, risk."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ._helpers import choice_of, require_dict
from .base import Tool

_STATE_KEY = "message"

_COMMIT_QUESTIONS = {
    "type": {
        "type": "choice",
        "instructions": "What kind of change does the commit `message` describe?",
        "criteria": {
            "feat": "a new feature or capability",
            "fix": "a bug fix",
            "refactor": "restructuring code without changing behaviour",
            "chore": "maintenance, housekeeping, version bumps",
            "docs": "documentation only",
            "test": "adding or changing tests only",
            "perf": "performance improvement",
            "build": "build system or packaging",
            "ci": "continuous integration configuration",
            "revert": "reverts an earlier commit",
        },
    },
    "scope": {
        "type": "choice",
        "instructions": "Which area does the commit `message` touch?",
        "criteria": {
            "api": "HTTP/RPC endpoints, public interfaces",
            "ui": "user interface, frontend",
            "db": "database schema, queries, migrations",
            "infra": "deployment, hosting, CI, tooling",
            "deps": "dependencies",
            "auth": "authentication, authorization, security",
            "none": "no specific area, or cannot tell",
        },
    },
    "risk": {
        "type": "choice",
        "instructions": "How risky is the change described in the commit `message`?",
        "criteria": {
            "low": "docs, tests, formatting or an isolated tweak",
            "medium": "changes behaviour in one area",
            "high": "touches security, data, migrations or many areas, or could break production",
        },
    },
}


class CommitClassifyInput(BaseModel):
    message: str = Field(..., description="A commit message to classify.")


class CommitClassifyOutput(BaseModel):
    type: str
    scope: str
    risk: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class CommitClassifyTool(Tool):
    name = "decision_commit_classify"
    description = (
        "Classify a commit message: type (feat / fix / refactor / chore / docs / test / "
        "perf / build / ci / revert), scope (api / ui / db / infra / deps / auth / none), "
        "and risk (low / medium / high)."
    )
    input_schema = CommitClassifyInput
    output_schema = CommitClassifyOutput

    async def run(self, bridge: DecisionBridge, message: str) -> CommitClassifyOutput:
        raw = bridge.predict_custom(message, questions=_COMMIT_QUESTIONS, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        commit_type, confidence = choice_of(raw, "type", self.name)
        scope, _ = choice_of(raw, "scope", self.name)
        risk, _ = choice_of(raw, "risk", self.name)
        return CommitClassifyOutput(
            type=commit_type,
            scope=scope,
            risk=risk,
            confidence=confidence,
            details=raw,
        )
