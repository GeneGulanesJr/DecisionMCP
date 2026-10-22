"""Tool: video scene-boundary verification (vision).

AutoVid's baseline scene detector (ffmpeg ``select=gt(scene,T)``) proposes
cut candidates; this tool looks at a few frames on each side of a candidate
and answers whether the cut is real, using the vision-capable d1-omni engine.

What the model sees
-------------------

``before`` / ``after`` frames are attached to the state as images (base64
JPEG); ``context`` is the only text, and interval/position metadata stays out
of it — position is asked as an ordinal ``score`` question instead of asking
the model to produce a number.

How the verdict is derived
--------------------------

- ``boundary``: the sides read as different scenes AND P(cut) is at least
  ``CUT_YES``.
- ``not_boundary``: the sides read as the same scene.
- ``shift``: everything else — a real scene difference the model cannot
  confirm as a clean cut (or simply hard to tell). ``position_hint`` says
  which way to move the cut when re-checking with a denser frame pair.

``confidence`` is the ``different_scene`` answer's confidence; callers should
treat low-confidence verdicts as "verify with denser frames" rather than as
fact. Requires an engine with ``supports_images`` — the tool refuses to guess
from text alone.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ..errors import ToolError
from ._helpers import choice_of, require_dict, score_level, yes_prob
from .base import Tool

_STATE_KEY = "context"

VERDICTS = ("boundary", "shift", "not_boundary")

_SIDE_DESCRIPTIONS = "at most three frames shortly before/after the candidate cut"

_BOUNDARY_QUESTIONS = {
    "different_scene": {
        "type": "choice",
        "instructions": (
            f"The attached frames show {_SIDE_DESCRIPTIONS}. Do the two sides "
            "show two clearly different scenes (place, subject, or camera setup)?"
        ),
        "criteria": {
            "same_scene": "one continuous scene: same place, subjects and lighting",
            "different_scene": (
                "two different scenes: the place, subject, shot size or lighting changes"
            ),
            "hard_to_tell": "the frames are too similar, blurred, or too few to judge",
        },
    },
    "is_cut": {
        "type": "noul",
        "instructions": (
            f"Do {_SIDE_DESCRIPTIONS} indicate a deliberate edit cut between them "
            "(an abrupt transition rather than continuous camera movement)?"
        ),
    },
    "cut_position": {
        "type": "score",
        "instructions": (
            "If the true cut is not exactly at the candidate, where is it relative "
            "to the candidate frame?"
        ),
        "criteria": [
            "clearly before the candidate frame",
            "essentially at the candidate frame",
            "clearly after the candidate frame",
        ],
    },
}

# Cut-off on P(cut) for calling a verified boundary (uncalibrated; see docstring).
CUT_YES = 0.5


class VideoBoundaryInput(BaseModel):
    context: str = Field(
        ...,
        min_length=1,
        description=(
            "Short text about the candidate (timestamp, detector score, source). "
            "Only this text reaches the model, next to the attached frames."
        ),
    )
    before: list[str] = Field(
        ..., min_length=1, max_length=3,
        description="Up to 3 base64 JPEG frames taken just before the candidate cut.",
    )
    after: list[str] = Field(
        ..., min_length=1, max_length=3,
        description="Up to 3 base64 JPEG frames taken just after the candidate cut.",
    )

    @field_validator("context")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("context must describe the candidate cut")
        return v


class VideoBoundaryOutput(BaseModel):
    verdict: str  # boundary | shift | not_boundary
    position_hint: str  # before | at | after
    cut_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class VideoBoundaryTool(Tool):
    name = "decision_video_boundary"
    description = (
        "Verify a candidate scene cut from frames before/after it (vision): boundary / "
        "shift / not_boundary, plus which way the true cut sits relative to the candidate."
    )
    input_schema = VideoBoundaryInput
    output_schema = VideoBoundaryOutput

    async def run(
        self, bridge: DecisionBridge, context: str, before: list[str], after: list[str]
    ) -> VideoBoundaryOutput:
        bridge.require_vision(self.name)
        raw = bridge.predict_custom(
            context,
            questions=_BOUNDARY_QUESTIONS,
            state_key=_STATE_KEY,
            images=[*before, *after],
        )
        require_dict(raw, self.name)
        scene, confidence = choice_of(raw, "different_scene", self.name)
        if scene not in ("same_scene", "different_scene", "hard_to_tell"):
            raise ToolError(self.name, f"Unexpected different_scene label {scene!r}")
        cut_p = yes_prob(raw, "is_cut", self.name)
        level, _, _ = score_level(raw, "cut_position", self.name)
        position_hint = ("before", "at", "after")[level]

        if scene == "different_scene" and cut_p >= CUT_YES:
            verdict = "boundary"
        elif scene == "same_scene" and cut_p < CUT_YES:
            verdict = "not_boundary"
        else:
            verdict = "shift"

        return VideoBoundaryOutput(
            verdict=verdict,
            position_hint=position_hint,
            cut_probability=cut_p,
            confidence=confidence,
            details=raw,
        )
