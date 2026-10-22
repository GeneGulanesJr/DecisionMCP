"""Tool: scene-continuity assessment between adjacent analysis windows (AutoVid
hierarchical segmentation, phase 2).

Long videos are segmented into coarse fixed-duration windows (phase 1, done by
AutoVid); this tool decides, for each adjacent pair, whether they belong to the
same continuous scene — the merge/split/verify verdict that drives phase 4
(temporal merging).

What the model sees
-------------------

``context``: compact summaries of the left and right window (descriptions,
activities, environments, tags, timestamps) plus whether the scene-change
detector found a cut near the shared boundary (``near_cut``). When the engine
is vision-capable, the client also attaches the last frame of the left window
and the first frame of the right window — the strongest evidence for a cut.

Verdict semantics
-----------------

- ``same_scene``: one continuous scene — merge the windows.
- ``different_scene``: a scene change happens at the boundary — split.
- ``needs_verification``: evidence is thin or contradictory — AutoVid phase 3
  inspects frames around the boundary (and any nearby detector cut) before
  deciding.

This tool is deliberately *not* vision-gated: descriptions alone are often
enough, so text-only engines still serve it. Images are forwarded whenever the
engine can use them.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ..errors import ToolError
from ._helpers import choice_of, require_dict, score_level, yes_prob
from .base import Tool

_STATE_KEY = "context"

RELATIONS = ("same_scene", "different_scene", "needs_verification")

_CONTINUITY_QUESTIONS = {
    "relation": {
        "type": "choice",
        "instructions": (
            "`context` summarizes two adjacent windows of one video; when images "
            "are attached, the first shows the end of the left window and the "
            "second the start of the right window. Do the two windows belong to "
            "the same continuous scene?"
        ),
        "criteria": {
            "same_scene": (
                "one continuous scene: same setting and subjects, the activity "
                "continues across the boundary"
            ),
            "different_scene": (
                "different scenes: the setting, subjects or activity clearly "
                "change at the boundary"
            ),
            "needs_verification": (
                "the evidence is insufficient or contradictory; the boundary "
                "should be inspected more closely before deciding"
            ),
        },
    },
    "shared_subjects": {
        "type": "noul",
        "instructions": (
            "Do the two windows share the same main subjects and setting?"
        ),
    },
    "change_magnitude": {
        "type": "score",
        "instructions": "How much changes between the left and right window?",
        "criteria": [
            "nothing changes: same scene continues",
            "subtle: minor drift within one scene",
            "clear: a different shot or subject, possibly the same scene",
            "complete: entirely different scene",
        ],
    },
}


class VideoContinuityInput(BaseModel):
    context: str = Field(
        ...,
        min_length=1,
        description=(
            "JSON summary of both windows (descriptions, activities, environments, "
            "tags, timestamps, near_cut flag). Only this text reaches the model, "
            "next to any attached frames."
        ),
    )
    images: list[str] = Field(
        default_factory=list,
        max_length=2,
        description=(
            "Optional [end-of-left frame, start-of-right frame] as base64 JPEG; "
            "forwarded when the engine is vision-capable."
        ),
    )

    @field_validator("context")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("context must summarize both windows")
        return v


class VideoContinuityOutput(BaseModel):
    relation: str  # same_scene | different_scene | needs_verification
    shared_subjects: float = Field(..., ge=0.0, le=1.0)
    change_level: int = Field(..., ge=0, le=3)
    change_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class VideoContinuityTool(Tool):
    name = "decision_video_continuity"
    description = (
        "Judge two adjacent video windows: same_scene (merge) / different_scene "
        "(split) / needs_verification (inspect the boundary) — drives scene merging."
    )
    input_schema = VideoContinuityInput
    output_schema = VideoContinuityOutput

    async def run(
        self, bridge: DecisionBridge, context: str, images: list[str] | None = None
    ) -> VideoContinuityOutput:
        raw = bridge.predict_custom(
            context, questions=_CONTINUITY_QUESTIONS, state_key=_STATE_KEY,
            images=images or None,
        )
        require_dict(raw, self.name)
        relation, confidence = choice_of(raw, "relation", self.name)
        if relation not in RELATIONS:
            raise ToolError(self.name, f"Unexpected relation label {relation!r}")
        shared = yes_prob(raw, "shared_subjects", self.name)
        level, text, _ = score_level(raw, "change_magnitude", self.name)
        return VideoContinuityOutput(
            relation=relation,
            shared_subjects=shared,
            change_level=level,
            change_text=text,
            confidence=confidence,
            details=raw,
        )
