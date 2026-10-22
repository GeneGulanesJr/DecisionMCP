"""End-to-end checks against the real d1-omni-600M engine (opt-in).

Proves the vision plumbing end to end — base64 frames in, PIL-decoded, vision
tower, answer orientation — using UNAMBIGUOUS synthetic frames (solid colors),
so a 600M model can pass them for reasons of capability rather than taste:

    DECISIONMCP_INTEGRATION=1 pytest tests/test_integration_d1_omni_tf.py

Needs the ``d1-omni-tf`` extra (torch CPU is fine), the checkpoint in
``./models`` (or the HF cache), and about 30 s to preload. Set
``DECISIONMCP_ENGINE=d1-omni-tf`` or rely on the fixture building the engine
directly.
"""
from __future__ import annotations

import base64
import io
import os

import pytest

from decision_mcp.tools import TOOLS

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DECISIONMCP_INTEGRATION") != "1",
        reason="set DECISIONMCP_INTEGRATION=1 to run against the real models",
    ),
]


def _jpeg_b64(rgb: tuple[int, int, int], size: int = 224) -> str:
    """A solid-color JPEG as base64 — maximally discriminable visual content."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=rgb).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


RED = _jpeg_b64((220, 30, 30))
BLUE = _jpeg_b64((30, 30, 220))


@pytest.fixture(scope="module")
def bridge():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from decision_mcp.bridge import DecisionBridge
    from decision_mcp.engines.d1_omni_tf import D1OmniTFEngine

    return DecisionBridge(engine=D1OmniTFEngine(), preload=True)


def _tool(name: str):
    return next(t for t in TOOLS if t.name == name)


# ---------------------------------------------------------------------------
# engine sanity: text + the image contract itself
# ---------------------------------------------------------------------------


def test_text_decision_directional(bridge) -> None:
    qs = {"has_dog": {"type": "noul", "instructions": "Does `context` mention a dog?"}}
    yes = bridge.engine.predict({"context": "a brown dog sleeps on the couch"}, qs)
    no = bridge.engine.predict({"context": "two cats sit by the window"}, qs)
    assert yes["answers"]["has_dog"]["noul"] > no["answers"]["has_dog"]["noul"]
    assert yes["answers"]["has_dog"]["noul"] > 0.5


def test_engine_reports_its_model_id(bridge) -> None:
    out = bridge.engine.predict(
        {"context": "x"},
        {"warm": {"type": "noul", "instructions": "Is `context` nonempty?"}},
    )
    assert "d1-omni-600M" in out.get("model", "")


# ---------------------------------------------------------------------------
# vision: boundary + segment tools over known frames
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_boundary_distinguishes_color_change(bridge) -> None:
    tool = _tool("decision_video_boundary")
    diff = await tool.run(bridge, context="candidate cut between two frames",
                          before=[RED], after=[BLUE])
    same = await tool.run(bridge, context="candidate cut between two frames",
                          before=[RED], after=[RED])
    # Directional contract: image-request noul answers are UNCALIBRATED (the
    # checkpoint skips temperature scaling for media), so verdicts hedge toward
    # "shift"/"needs_verification" instead of committing. What must hold is
    # discrimination: the changed pair never reads as a continuous scene, the
    # identical pair never confirms a cut, and the cut probability separates.
    assert diff.verdict != "not_boundary"
    assert same.verdict != "boundary"
    assert diff.cut_probability > same.cut_probability


@pytest.mark.asyncio
async def test_segment_separates_content_from_dead_time(bridge) -> None:
    tool = _tool("decision_video_segment")
    colorful = await tool.run(
        bridge,
        summary="A clip with strongly colored, changing visuals.",
        frames=[RED, BLUE, RED],
        duration_s=6.0,
    )
    dead = await tool.run(
        bridge,
        summary="A static, unchanging solid screen with no events.",
        frames=[RED, RED],
        duration_s=6.0,
    )
    # the colorful clip should rank as at least as acceptable as dead time
    assert colorful.verdict in ("accept", "needs_evidence", "flag_review")
    assert dead.verdict != "accept" or colorful.verdict == "accept"
    assert colorful.is_highlight >= dead.is_highlight


@pytest.mark.asyncio
async def test_continuity_text_only(bridge) -> None:
    tool = _tool("decision_video_continuity")
    same = await tool.run(bridge, context=(
        '{"left_window": {"description": "a red field, no movement"},'
        ' "right_window": {"description": "the same red field continues"},'
        ' "detector_cut_near_boundary": false}'
    ))
    diff = await tool.run(bridge, context=(
        '{"left_window": {"description": "a red field, no movement"},'
        ' "right_window": {"description": "a blue sky with clouds"},'
        ' "detector_cut_near_boundary": true}'
    ))
    # Directional: the continuous pair never reads as a scene change and the
    # changed pair never reads as a clean merge.
    assert same.relation != "different_scene"
    assert diff.relation != "same_scene"
