"""Tests for the AutoVid video decision tools (region / boundary / segment)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from decision_mcp.bridge import DecisionBridge
from decision_mcp.errors import BridgeError
from decision_mcp.tools import (
    TOOLS,
    VideoBoundaryTool,
    VideoRegionTool,
    VideoSegmentTool,
)
from decision_mcp.tools.video_boundary import CUT_YES
from decision_mcp.tools.video_segment import HIGHLIGHT_ACCEPT
from tests.conftest import FakeEngine


# ---------------------------------------------------------------------------
# Helpers: canned engine-shaped results
# ---------------------------------------------------------------------------


def _answers(**entries: dict) -> dict:
    return {"answers": entries, "usage": {"input_tokens": 10, "output_tokens": 0}}


def _noul(p: float) -> dict:
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p),
            "answer_confidence": max(p, 1 - p)}


def _choice(label: str, conf: float, probs: dict) -> dict:
    return {"type": "choice", "choice": label, "confidence": conf,
            "probabilities": probs, "answer_confidence": conf}


def _score(level: float, conf: float, legend: dict) -> dict:
    probs = {str(i): 1.0 / len(legend) for i in legend}
    return {"type": "score", "score": level, "confidence": conf,
            "probabilities": probs, "legend": legend, "answer_confidence": conf}


def _vision_bridge(state_capture: list) -> MagicMock:
    """A MagicMock bridge that records predict_custom state and passes vision.

    Records the call, then returns whatever ``predict_custom.return_value``
    the test configured (evaluated at call time).
    """
    bridge = MagicMock()
    bridge.vision_capable = True
    bridge.engine.name = "d1-omni-tf"

    mock = bridge.predict_custom

    def record(state, questions, state_key="text", images=None):
        state_capture.append({"state": state, "questions": questions, "images": images})
        return mock.return_value

    mock.side_effect = record
    return bridge


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_video_tools_registered() -> None:
    names = {t.name for t in TOOLS}
    assert {"decision_video_region", "decision_video_boundary", "decision_video_segment"} <= names


# ---------------------------------------------------------------------------
# decision_video_region
# ---------------------------------------------------------------------------


async def test_region_sufficient_has_no_interval_change() -> None:
    bridge = MagicMock()
    bridge.predict_custom.return_value = _answers(
        action=_choice("sufficient", 0.8, {"sufficient": 0.8}),
        has_content=_noul(0.2),
        ambiguity=_score(0.0, 0.9, {"0": "clear", "1": "minor", "2": "unclear", "3": "lost"}),
    )
    out = await VideoRegionTool().run(bridge, summary="{}", current_interval_s=2.0)
    assert out.action == "sufficient"
    assert out.suggested_interval_s is None
    assert out.ambiguity_level == 0
    assert out.confidence == pytest.approx(0.8)


async def test_region_densify_halves_interval_within_clamps() -> None:
    bridge = MagicMock()
    bridge.predict_custom.return_value = _answers(
        action=_choice("densify", 0.9, {"densify": 0.9}),
        has_content=_noul(0.9),
        ambiguity=_score(1.0, 0.9, {"0": "a", "1": "b", "2": "c", "3": "d"}),
    )
    tool = VideoRegionTool()
    out = await tool.run(bridge, summary="{}", current_interval_s=4.0)
    assert out.action == "densify"
    assert out.suggested_interval_s == pytest.approx(2.0)

    # densify clamps at min_interval_s
    out = await tool.run(bridge, summary="{}", current_interval_s=0.6, min_interval_s=0.5)
    assert out.suggested_interval_s == pytest.approx(0.5)


async def test_region_inspect_more_and_flag_review_pass_through() -> None:
    bridge = MagicMock()
    bridge.predict_custom.return_value = _answers(
        action=_choice("inspect_more", 0.7, {"inspect_more": 0.7}),
        has_content=_noul(0.5),
        ambiguity=_score(2.0, 0.8, {"0": "a", "1": "b", "2": "c", "3": "d"}),
    )
    out = await VideoRegionTool().run(bridge, summary="{}", current_interval_s=2.0)
    assert out.action == "inspect_more" and out.suggested_interval_s is None

    bridge.predict_custom.return_value = _answers(
        action=_choice("flag_review", 0.6, {"flag_review": 0.6}),
        has_content=_noul(0.9),
        ambiguity=_score(3.0, 0.8, {"0": "a", "1": "b", "2": "c", "3": "d"}),
    )
    out = await VideoRegionTool().run(bridge, summary="{}", current_interval_s=2.0)
    assert out.action == "flag_review"


async def test_region_rejects_blank_summary() -> None:
    with pytest.raises(ValidationError):
        VideoRegionInput_check()


def VideoRegionInput_check() -> None:
    VideoRegionTool().input_schema(summary="   ", current_interval_s=2.0)


# ---------------------------------------------------------------------------
# decision_video_boundary
# ---------------------------------------------------------------------------


async def test_boundary_requires_a_vision_engine() -> None:
    bridge = DecisionBridge(engine=FakeEngine(), preload=False)
    with pytest.raises(BridgeError, match="supports_images"):
        await VideoBoundaryTool().run(
            bridge, context="t=3.0", before=["<b64>"], after=["<b64>"]
        )


async def test_boundary_confirmed_when_scene_differs_and_cut_is_confident() -> None:
    captured: list = []
    bridge = _vision_bridge(captured)
    bridge.predict_custom.return_value = _answers(
        different_scene=_choice("different_scene", 0.9,
                                {"different_scene": 0.9, "same_scene": 0.05, "hard_to_tell": 0.05}),
        is_cut=_noul(0.9),
        cut_position=_score(1.0, 0.8, {"0": "before", "1": "at", "2": "after"}),
    )
    out = await VideoBoundaryTool().run(bridge, context="t=3.0", before=["<a>"], after=["<b>"])
    assert out.verdict == "boundary"
    assert out.position_hint == "at"
    assert out.cut_probability == pytest.approx(0.9)
    # frames travel with the call; the bridge composes the state dict itself
    assert captured[0]["images"] == ["<a>", "<b>"]
    assert captured[0]["state"] == "t=3.0"


async def test_boundary_not_boundary_when_same_scene() -> None:
    bridge = _vision_bridge([])
    bridge.predict_custom.return_value = _answers(
        different_scene=_choice("same_scene", 0.85,
                                {"different_scene": 0.05, "same_scene": 0.85, "hard_to_tell": 0.1}),
        is_cut=_noul(1 - CUT_YES - 0.01),
        cut_position=_score(1.0, 0.8, {"0": "before", "1": "at", "2": "after"}),
    )
    out = await VideoBoundaryTool().run(bridge, context="t=3.0", before=["<a>"], after=["<b>"])
    assert out.verdict == "not_boundary"


async def test_boundary_shift_when_unsure() -> None:
    bridge = _vision_bridge([])
    bridge.predict_custom.return_value = _answers(
        different_scene=_choice("different_scene", 0.5,
                                {"different_scene": 0.5, "same_scene": 0.25, "hard_to_tell": 0.25}),
        is_cut=_noul(0.3),
        cut_position=_score(0.0, 0.8, {"0": "before", "1": "at", "2": "after"}),
    )
    out = await VideoBoundaryTool().run(bridge, context="t=3.0", before=["<a>"], after=["<b>"])
    assert out.verdict == "shift"
    assert out.position_hint == "before"


async def test_boundary_rejects_empty_sides() -> None:
    with pytest.raises(ValidationError):
        VideoBoundaryTool().input_schema(context="t=3.0", before=[], after=["<b>"])


# ---------------------------------------------------------------------------
# decision_video_segment
# ---------------------------------------------------------------------------


async def test_segment_requires_a_vision_engine() -> None:
    bridge = DecisionBridge(engine=FakeEngine(), preload=False)
    with pytest.raises(BridgeError, match="supports_images"):
        await VideoSegmentTool().run(bridge, summary="a clip", frames=["<a>"])


async def test_segment_accepts_clean_highlight() -> None:
    bridge = _vision_bridge([])
    bridge.predict_custom.return_value = _answers(
        verdict=_choice("accept", 0.9, {"accept": 0.9}),
        is_highlight=_noul(0.95),
        self_contained=_noul(0.9),
        quality=_score(2.0, 0.9, {"0": "poor", "1": "fair", "2": "good", "3": "excellent"}),
    )
    out = await VideoSegmentTool().run(bridge, summary="goal scored", frames=["<a>"],
                                       duration_s=6.0)
    assert out.verdict == "accept"
    assert out.quality_level == 2
    assert out.is_highlight == pytest.approx(0.95)


async def test_segment_promotes_confident_complete_highlight() -> None:
    bridge = _vision_bridge([])
    bridge.predict_custom.return_value = _answers(
        verdict=_choice("needs_evidence", 0.6, {"needs_evidence": 0.6}),
        is_highlight=_noul(HIGHLIGHT_ACCEPT + 0.05),
        self_contained=_noul(0.9),
        quality=_score(2.0, 0.9, {"0": "poor", "1": "fair", "2": "good", "3": "excellent"}),
    )
    out = await VideoSegmentTool().run(bridge, summary="a dance", frames=["<a>"], duration_s=4.0)
    assert out.verdict == "accept"


async def test_segment_short_highlight_needs_evidence() -> None:
    bridge = _vision_bridge([])
    bridge.predict_custom.return_value = _answers(
        verdict=_choice("accept", 0.8, {"accept": 0.8}),
        is_highlight=_noul(0.9),
        self_contained=_noul(0.9),
        quality=_score(3.0, 0.9, {"0": "poor", "1": "fair", "2": "good", "3": "excellent"}),
    )
    out = await VideoSegmentTool().run(bridge, summary="flash", frames=["<a>"],
                                       duration_s=0.2, min_duration_s=0.5)
    assert out.verdict == "needs_evidence"


async def test_segment_contradiction_flags_review() -> None:
    bridge = _vision_bridge([])
    bridge.predict_custom.return_value = _answers(
        verdict=_choice("reject", 0.7, {"reject": 0.7}),
        is_highlight=_noul(0.98),
        self_contained=_noul(0.2),
        quality=_score(0.0, 0.9, {"0": "poor", "1": "fair", "2": "good", "3": "excellent"}),
    )
    out = await VideoSegmentTool().run(bridge, summary="odd", frames=["<a>"])
    assert out.verdict == "flag_review"
