"""Tool: music-driven sequence review (AutoVid edit planning).

The music analysis (beats, sections, energy) is measurable — librosa does the
timing, never this tool. After AutoVid's deterministic planner drafts one
music section's cut plan (which clip fills which beat-aligned slot), this tool
reviews the draft with bounded actions:

- ``approve`` — the plan stands.
- ``replace_clip`` — one of the chosen clips doesn't fit the section's mood or
  repeats recently used footage; AutoVid swaps in the next-best candidate.
- ``slow_pace`` / ``fast_pace`` — the cutting density is wrong for the music
  (too frantic under a calm passage, or dragging under a driving one);
  AutoVid regenerates the section at double or half the beats per cut.

The model sees only the section summary (role, timings, beats-per-cut, chosen
clip ids and slot lengths). It never estimates beat timestamps itself.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ..errors import ToolError
from ._helpers import choice_of, require_dict
from .base import Tool

_STATE_KEY = "plan"

ACTIONS = ("approve", "replace_clip", "slow_pace", "fast_pace")

_SEQUENCE_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": (
            "Given the drafted music-section cut plan in `plan`, what should "
            "the editor change, if anything?"
        ),
        "criteria": {
            "approve": "the clip choices and pacing fit this section of the music",
            "replace_clip": (
                "the pacing works but one of the clips does not fit the music's "
                "mood here, or repeats footage used just before it"
            ),
            "slow_pace": (
                "the cuts are too frequent for the music in this section; fewer, "
                "longer shots would fit"
            ),
            "fast_pace": (
                "the cuts are too sparse for the music in this section; more "
                "frequent cuts would fit the rhythm"
            ),
        },
    },
    "concern": {
        "type": "choice",
        "instructions": "What is the main weakness of `plan`, if any?",
        "criteria": {
            "none": "no notable weakness",
            "pacing": "the cutting rhythm does not match the music",
            "repetition": "clips look or sound too similar to each other",
            "mood_fit": "the footage mood does not match this part of the music",
        },
    },
}


class MusicSequenceInput(BaseModel):
    plan: str = Field(
        ...,
        min_length=1,
        description=(
            "JSON summary of the drafted section plan: role (intro/build/climax/"
            "outro), timings, beats-per-cut, and the chosen clips with slot "
            "lengths. Only this reaches the model."
        ),
    )

    @field_validator("plan")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("plan must describe the drafted section")
        return v


class MusicSequenceOutput(BaseModel):
    action: str  # approve | replace_clip | slow_pace | fast_pace
    concern: str  # none | pacing | repetition | mood_fit
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class MusicSequenceTool(Tool):
    name = "decision_music_sequence"
    description = (
        "Review a drafted music-section cut plan (clip choices + pacing): approve / "
        "replace_clip / slow_pace / fast_pace, with the main concern named."
    )
    input_schema = MusicSequenceInput
    output_schema = MusicSequenceOutput

    async def run(self, bridge: DecisionBridge, plan: str) -> MusicSequenceOutput:
        raw = bridge.predict_custom(plan, questions=_SEQUENCE_QUESTIONS,
                                    state_key=_STATE_KEY)
        require_dict(raw, self.name)
        action, confidence = choice_of(raw, "action", self.name)
        if action not in ACTIONS:
            raise ToolError(self.name, f"Unexpected action label {action!r}")
        concern, _ = choice_of(raw, "concern", self.name)
        return MusicSequenceOutput(
            action=action,
            concern=concern,
            confidence=confidence,
            details=raw,
        )
