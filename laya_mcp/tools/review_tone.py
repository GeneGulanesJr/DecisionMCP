"""Tool: code review comment tone classifier.

Uses a custom Laya question schema (not from upstream presets) to
classify review comments along two dimensions:

- tone: nit / suggestion / blocking / praise / question / off_topic
- priority: low / medium / high
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..bridge import LayaBridge
from ._helpers import choice_of, require_dict
from .base import Tool

_STATE_KEY = "comment"

_REVIEW_QUESTIONS = {
    "tone": {
        "type": "choice",
        "instructions": "What is the tone of the code review `comment`?",
        "criteria": {
            "nit": "trivial style or formatting remark that does not affect behaviour",
            "suggestion": "optional improvement the author may take or leave",
            "blocking": "a problem that must be fixed before merging, such as a bug or a risk",
            "praise": "positive feedback",
            "question": "asks the author to explain or clarify something",
            "off_topic": "unrelated to the code under review",
        },
    },
    "priority": {
        "type": "choice",
        "instructions": "How urgently does the author need to act on the code review `comment`?",
        "criteria": {
            "low": "no action needed, or cosmetic",
            "medium": "should be addressed in this change",
            "high": "must be addressed before merging",
        },
    },
}


class ReviewToneInput(BaseModel):
    comment: str = Field(..., description="A code review comment to classify.")


class ReviewToneOutput(BaseModel):
    tone: str
    priority: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class ReviewToneTool(Tool):
    name = "laya_review_tone"
    description = (
        "Classify a code review comment's tone (nit / suggestion / blocking / "
        "praise / question / off-topic) and priority (low / medium / high)."
    )
    input_schema = ReviewToneInput
    output_schema = ReviewToneOutput

    async def run(self, bridge: LayaBridge, comment: str) -> ReviewToneOutput:
        raw = bridge.predict_custom(comment, questions=_REVIEW_QUESTIONS, state_key=_STATE_KEY)
        require_dict(raw, self.name)
        tone, confidence = choice_of(raw, "tone", self.name)
        priority, _ = choice_of(raw, "priority", self.name)
        return ReviewToneOutput(
            tone=tone,
            priority=priority,
            confidence=confidence,
            details=raw,
        )
