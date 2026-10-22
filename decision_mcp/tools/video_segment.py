"""Tool: candidate clip acceptance (vision).

After AutoVid proposes a candidate clip (start/end, description, evidence
frames), this tool decides — with bounded, reviewable verdicts — whether the
segment is accepted into the catalog, needs more evidence, is rejected, or is
flagged for human review.

What the model sees
-------------------

Up to four evidence frames as images plus a single ``summary`` text (what
happens, duration, motion, audio presence). Client-side bounds such as
``min_duration_s`` never reach the model; they only tighten the final verdict
in code (a too-short "highlight" is demoted to ``needs_evidence``).

Verdict semantics
-----------------

- ``accept``: the model chose accept (or a confident highlight with complete
  action) and the code-side duration check passes.
- ``needs_evidence``: interesting but under-evidenced — more frames or a
  better description required before it enters the catalog.
- ``reject``: not meaningful content (dead time, duplicates, noise).
- ``flag_review``: contradictions between the answers, or the model itself
  chose flag_review; a human should look before the segment is used.

Requires an engine with ``supports_images``; refuses to judge a clip blind.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ..errors import ToolError
from ._helpers import choice_of, require_dict, score_level, yes_prob
from .base import Tool

_STATE_KEY = "summary"

VERDICTS = ("accept", "needs_evidence", "reject", "flag_review")

_SEGMENT_QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "Judging `summary` together with the attached evidence frames, what "
            "should happen to this candidate clip?"
        ),
        "criteria": {
            "accept": (
                "the evidence supports the description; the clip is meaningful and "
                "usable in an edit"
            ),
            "needs_evidence": (
                "possibly meaningful, but the frames or description are insufficient "
                "to be confident"
            ),
            "reject": (
                "not meaningful: dead time, accidental recording, or nothing happens"
            ),
            "flag_review": (
                "the evidence and description disagree, or the content needs a "
                "human decision"
            ),
        },
    },
    "is_highlight": {
        "type": "noul",
        "instructions": (
            "Could `summary` (with the frames) be a highlight worth keeping in a "
            "rough cut (action, emotion, key moment)?"
        ),
    },
    "self_contained": {
        "type": "noul",
        "instructions": (
            "Do the frames and `summary` describe a self-contained moment — the "
            "action starts and finishes inside the clip?"
        ),
    },
    "quality": {
        "type": "score",
        "instructions": "How good is this clip as edit material?",
        "criteria": [
            "poor: unusable (unreadable, shaky, empty)",
            "fair: usable but weak",
            "good: solid material",
            "excellent: standout material",
        ],
    },
}

# Code-side verdict tightening (uncalibrated probabilities; see module docstring).
HIGHLIGHT_ACCEPT = 0.7
COMPLETE_ACTION = 0.5
CONTRADICTED = 0.5


class VideoSegmentInput(BaseModel):
    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "What the candidate clip contains: activity, subjects, setting, duration, "
            "motion and audio notes. Only this text reaches the model, next to the frames."
        ),
    )
    frames: list[str] = Field(
        ..., min_length=1, max_length=4,
        description="1-4 base64 JPEG evidence frames spanning the candidate clip.",
    )
    duration_s: float | None = Field(
        default=None, gt=0,
        description="Clip length in seconds. Used only in code; not shown to the model.",
    )
    min_duration_s: float = Field(
        default=0.5, gt=0,
        description="Clips shorter than this cannot be accepted outright.",
    )

    @field_validator("summary")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary must describe the candidate clip")
        return v


class VideoSegmentOutput(BaseModel):
    verdict: str  # accept | needs_evidence | reject | flag_review
    is_highlight: float = Field(..., ge=0.0, le=1.0)
    self_contained: float = Field(..., ge=0.0, le=1.0)
    quality_level: int = Field(..., ge=0, le=3)
    quality_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class VideoSegmentTool(Tool):
    name = "decision_video_segment"
    description = (
        "Accept/reject a candidate video clip from evidence frames + summary (vision): "
        "accept / needs_evidence / reject / flag_review, highlight and quality estimates."
    )
    input_schema = VideoSegmentInput
    output_schema = VideoSegmentOutput

    async def run(
        self,
        bridge: DecisionBridge,
        summary: str,
        frames: list[str],
        duration_s: float | None = None,
        min_duration_s: float = 0.5,
    ) -> VideoSegmentOutput:
        bridge.require_vision(self.name)
        raw = bridge.predict_custom(
            summary, questions=_SEGMENT_QUESTIONS, state_key=_STATE_KEY, images=frames
        )
        require_dict(raw, self.name)
        verdict, confidence = choice_of(raw, "verdict", self.name)
        if verdict not in VERDICTS:
            raise ToolError(self.name, f"Unexpected verdict label {verdict!r}")
        highlight = yes_prob(raw, "is_highlight", self.name)
        self_contained = yes_prob(raw, "self_contained", self.name)
        level, text, _ = score_level(raw, "quality", self.name)

        # Contradiction: called a highlight but judged not meaningful, or vice versa.
        if (
            (highlight >= HIGHLIGHT_ACCEPT and verdict == "reject")
            or (highlight < 1 - HIGHLIGHT_ACCEPT and verdict == "accept")
        ):
            verdict = "flag_review"
        # A confident highlight that reads complete can be accepted even when the
        # model hedged; a too-short clip can never be accepted outright.
        elif (
            verdict == "needs_evidence"
            and highlight >= HIGHLIGHT_ACCEPT
            and self_contained >= COMPLETE_ACTION
            and (duration_s is None or duration_s >= min_duration_s)
        ):
            verdict = "accept"
        elif verdict == "accept" and duration_s is not None and duration_s < min_duration_s:
            verdict = "needs_evidence"

        return VideoSegmentOutput(
            verdict=verdict,
            is_highlight=highlight,
            self_contained=self_contained,
            quality_level=level,
            quality_text=text,
            confidence=confidence,
            details=raw,
        )
