"""Tool: video region sampling policy (bounded analyzer actions).

AutoVid partitions each video into regions (between scene-change boundaries or
fixed windows) and, while analyzing, asks this tool what to do next per region.
The model sees only a compact stat summary the client composes; interval
bounds never reach it — the tool uses them in code to translate the model's
choice into a concrete, bounded action:

- ``sufficient``: keep the current sampling; no extra frames needed.
- ``densify``: halve the sampling interval (clamped to ``min_interval_s``);
  ``suggested_interval_s`` carries the new interval.
- ``inspect_more``: sample a few more frames around the ambiguous spots
  without changing the global interval.
- ``flag_review``: the region is suspicious enough that a human should look.

The thresholds are plain cut-offs on uncalibrated probabilities: treat
``action`` as a work-queue hint, and log it (AutoVid's decisions table) so the
contribution of decision-guided sampling stays measurable against the
fixed-interval baseline.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ._helpers import choice_of, require_dict, score_level, yes_prob
from .base import Tool

_STATE_KEY = "region"

ACTIONS = ("sufficient", "densify", "inspect_more", "flag_review")

_REGION_QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": (
            "Given the sampler statistics in `region`, what should the video "
            "analyzer do with this region next?"
        ),
        "criteria": {
            "sufficient": (
                "the existing frames describe the region well enough; no more sampling needed"
            ),
            "densify": (
                "something interesting is happening but the current frames are too sparse; "
                "sample this region more densely"
            ),
            "inspect_more": (
                "a specific spot is unclear and needs a few extra frames around it, "
                "without densifying the whole region"
            ),
            "flag_review": (
                "the statistics are confusing or contradictory; a human should look at "
                "this region"
            ),
        },
    },
    "has_content": {
        "type": "noul",
        "instructions": (
            "Does `region` look like it contains meaningful content (action, people, "
            "events, notable scene changes) rather than dead time?"
        ),
    },
    "ambiguity": {
        "type": "score",
        "instructions": "How ambiguous is what is happening in `region`?",
        "criteria": [
            "clearly understood",
            "mostly clear with minor gaps",
            "significantly unclear",
            "cannot tell what is happening",
        ],
    },
}

# How much densify tightens the interval, and the clamps applied in code.
DENSIFY_FACTOR = 0.5


class VideoRegionInput(BaseModel):
    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "Compact stat summary of the region the client composes, e.g. JSON with "
            "duration, scene-change count, mean/max motion, current sample interval, "
            "VLM uncertainty. Only this reaches the model."
        ),
    )
    current_interval_s: float = Field(
        ...,
        gt=0,
        description="Current grid sampling interval in seconds. Used only in code "
        "(densify target = half of this); not shown to the model.",
    )
    min_interval_s: float = Field(
        default=0.5, gt=0, description="Densify clamp floor, seconds."
    )
    max_interval_s: float = Field(
        default=30.0, gt=0, description="Upper clamp for suggested intervals, seconds."
    )

    @field_validator("summary")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary must describe the region; there is nothing to decide on")
        return v


class VideoRegionOutput(BaseModel):
    action: str  # sufficient | densify | inspect_more | flag_review
    suggested_interval_s: float | None = Field(default=None, gt=0)
    has_content: float = Field(..., ge=0.0, le=1.0)
    ambiguity_level: int = Field(..., ge=0)
    ambiguity_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class VideoRegionTool(Tool):
    name = "decision_video_region"
    description = (
        "Decide the next bounded sampling action for one video region from its stats "
        "summary: sufficient / densify (halve the interval) / inspect_more / flag_review."
    )
    input_schema = VideoRegionInput
    output_schema = VideoRegionOutput

    async def run(
        self,
        bridge: DecisionBridge,
        summary: str,
        current_interval_s: float,
        min_interval_s: float = 0.5,
        max_interval_s: float = 30.0,
    ) -> VideoRegionOutput:
        raw = bridge.predict_custom(
            summary, questions=_REGION_QUESTIONS, state_key=_STATE_KEY
        )
        require_dict(raw, self.name)
        action, confidence = choice_of(raw, "action", self.name)
        if action not in ACTIONS:
            raise ValueError(f"Unexpected action label {action!r}")
        has_content = yes_prob(raw, "has_content", self.name)
        level, text, _ = score_level(raw, "ambiguity", self.name)

        suggested: float | None = None
        if action == "densify":
            suggested = max(
                min(current_interval_s * DENSIFY_FACTOR, max_interval_s), min_interval_s
            )

        return VideoRegionOutput(
            action=action,
            suggested_interval_s=suggested,
            has_content=has_content,
            ambiguity_level=level,
            ambiguity_text=text,
            confidence=confidence,
            details=raw,
        )
