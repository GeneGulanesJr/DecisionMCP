"""Tool: music section mood/energy classification by listening (AutoVid
music-driven editing).

AutoVid measures the *timing* side of music with librosa (beats, energy
envelope, sections) and asks this tool for the part that needs ears: what the
section actually sounds like. The section's audio (16 kHz mono PCM16, up to
30 s) is base64-encoded in the request; the audio-capable engine (d1-omni)
hears it and answers typed questions. Timing questions are out of scope here —
librosa owns those.

Requires an engine with ``supports_audio``; refuses to guess mood from
energy numbers alone.
"""
from __future__ import annotations

import base64

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..bridge import DecisionBridge
from ..errors import ToolError
from ._helpers import choice_of, require_dict, score_level, yes_prob
from .base import Tool

_STATE_KEY = "context"

MOODS = ("calm", "neutral", "energetic", "driving")

SAMPLE_RATE = 16000
MAX_SECONDS = 30  # the checkpoint cuts at 30 s; decode no more

_MOOD_QUESTIONS = {
    "mood": {
        "type": "choice",
        "instructions": (
            "Listen to the audio clip described in `context`. What is its "
            "overall musical character?"
        ),
        "criteria": {
            "calm": "quiet, sparse, ambient or slow; little rhythmic drive",
            "neutral": "steady background music without a strong character",
            "energetic": "clear rhythm and movement; would suit active visuals",
            "driving": "intense, loud, propulsive; demands fast cutting",
        },
    },
    "energy": {
        "type": "score",
        "instructions": "How much energy does the audio have?",
        "criteria": [
            "very low: near silence or ambient wash",
            "low: gentle and restrained",
            "high: busy and assertive",
            "very high: loud and intense",
        ],
    },
    "builds": {
        "type": "noul",
        "instructions": (
            "Does the audio build in intensity toward the end (a rise or "
            "drop about to happen)?"
        ),
    },
}


class MusicMoodInput(BaseModel):
    context: str = Field(
        ...,
        min_length=1,
        description=(
            "Short text about the section (timings, measured energy). Only this "
            "text reaches the model, next to the audio."
        ),
    )
    audio_pcm16_b64: str = Field(
        ...,
        min_length=64,
        description=(
            "The section's audio as base64 of 16 kHz mono int16 PCM, up to 30 s."
        ),
    )

    @field_validator("context")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("context must describe the section")
        return v


class MusicMoodOutput(BaseModel):
    mood: str  # calm | neutral | energetic | driving
    energy_level: int = Field(..., ge=0, le=3)
    builds: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class MusicMoodTool(Tool):
    name = "decision_music_mood"
    description = (
        "Classify a music section by listening (audio input): calm / neutral / "
        "energetic / driving, an energy level, and whether it builds toward the end."
    )
    input_schema = MusicMoodInput
    output_schema = MusicMoodOutput

    async def run(self, bridge: DecisionBridge, context: str, audio_pcm16_b64: str) -> MusicMoodOutput:
        bridge.require_audio(self.name)
        raw_pcm = base64.b64decode(audio_pcm16_b64, validate=True)
        samples = np.frombuffer(raw_pcm, dtype=np.int16)[: MAX_SECONDS * SAMPLE_RATE]
        if samples.size == 0:
            raise ToolError(self.name, "audio decoded to zero samples")
        raw = bridge.predict_custom(
            context, questions=_MOOD_QUESTIONS, state_key=_STATE_KEY, audio=samples
        )
        require_dict(raw, self.name)
        mood, confidence = choice_of(raw, "mood", self.name)
        if mood not in MOODS:
            raise ToolError(self.name, f"Unexpected mood label {mood!r}")
        level, _, _ = score_level(raw, "energy", self.name)
        builds = yes_prob(raw, "builds", self.name)
        return MusicMoodOutput(
            mood=mood,
            energy_level=level,
            builds=builds,
            confidence=confidence,
            details=raw,
        )
