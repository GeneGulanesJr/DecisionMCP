"""Tests for the music-editing tools (sequence review + mood by listening)."""
from __future__ import annotations

import base64
import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from decision_mcp.bridge import DecisionBridge
from decision_mcp.errors import BridgeError
from decision_mcp.tools import TOOLS, MusicMoodTool, MusicSequenceTool
from tests.conftest import FakeEngine


def _answers(**entries: dict) -> dict:
    return {"answers": entries, "usage": {"input_tokens": 10, "output_tokens": 0}}


def _choice(label: str, conf: float) -> dict:
    return {"type": "choice", "choice": label, "confidence": conf,
            "probabilities": {"opt0": conf, "opt1": 1 - conf},
            "answer_confidence": conf}


def _score(level: float, conf: float, legend: dict) -> dict:
    probs = {str(i): 1.0 / len(legend) for i in legend}
    return {"type": "score", "score": level, "confidence": conf,
            "probabilities": probs, "legend": legend, "answer_confidence": conf}


def _noul(p: float) -> dict:
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p),
            "answer_confidence": max(p, 1 - p)}


def _audio_bridge(capture: list) -> MagicMock:
    """Audio-capable MagicMock bridge: records predict_custom calls, then
    returns whatever return_value the test configured."""
    bridge = MagicMock()
    bridge.audio_capable = True
    bridge.engine.name = "d1-omni-tf"

    mock = bridge.predict_custom

    def record(state, questions, state_key="text", images=None, audio=None):
        capture.append({"state": state, "questions": questions, "audio": audio})
        return mock.return_value

    mock.side_effect = record
    return bridge


def _pcm16_b64(seconds: float = 1.0, sr: int = 16000, hz: float = 440) -> str:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    samples = (np.sin(2 * math.pi * hz * t) * 12000).astype(np.int16)
    return base64.b64encode(samples.tobytes()).decode()


def test_music_tools_registered() -> None:
    names = {t.name for t in TOOLS}
    assert {"decision_music_sequence", "decision_music_mood"} <= names


# ---------------------------------------------------------------------------
# decision_music_sequence
# ---------------------------------------------------------------------------


async def test_sequence_review_translates() -> None:
    bridge = MagicMock()
    bridge.predict_custom.return_value = _answers(
        action=_choice("fast_pace", 0.8),
        concern=_choice("pacing", 0.7),
    )
    out = await MusicSequenceTool().run(bridge, plan='{"role": "intro"}')
    assert out.action == "fast_pace"
    assert out.concern == "pacing"
    assert out.confidence == pytest.approx(0.8)


async def test_sequence_review_rejects_blank_plan() -> None:
    with pytest.raises(ValidationError):
        MusicSequenceTool().input_schema(plan="  ")


# ---------------------------------------------------------------------------
# decision_music_mood
# ---------------------------------------------------------------------------


async def test_mood_decodes_pcm_and_answers() -> None:
    captured: list = []
    bridge = _audio_bridge(captured)
    bridge.predict_custom.return_value = _answers(
        mood=_choice("driving", 0.85),
        energy=_score(3.0, 0.9, {"0": "very low", "1": "low", "2": "high", "3": "very high"}),
        builds=_noul(0.7),
    )
    out = await MusicMoodTool().run(
        bridge, context='{"role": "climax"}', audio_pcm16_b64=_pcm16_b64(0.5)
    )
    assert out.mood == "driving"
    assert out.energy_level == 3
    assert out.builds == pytest.approx(0.7)
    # audio decoded to int16 numpy samples for the engine
    arr = captured[0]["audio"]
    assert isinstance(arr, np.ndarray) and arr.dtype == np.int16
    assert len(arr) == 8000


async def test_mood_requires_audio_engine() -> None:
    bridge = DecisionBridge(engine=FakeEngine(), preload=False)
    with pytest.raises(BridgeError, match="supports_audio"):
        await MusicMoodTool().run(bridge, context="a section",
                                  audio_pcm16_b64=_pcm16_b64(0.5))


async def test_mood_rejects_blank_context() -> None:
    with pytest.raises(ValidationError):
        MusicMoodTool().input_schema(context=" ", audio_pcm16_b64=_pcm16_b64(0.5))
