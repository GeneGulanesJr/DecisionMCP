"""Tests for error handling: bridge errors, tool parsing errors, server errors."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import FakeEngine
from fixtures import choice, noul, result

from decision_mcp.bridge import DecisionBridge
from decision_mcp.errors import (
    BridgeError,
    DecisionMCPError,
    ModelLoadError,
    ToolError,
    UnknownPresetError,
)
from decision_mcp.tools.email import EmailTool
from decision_mcp.tools.guard import GuardTool
from decision_mcp.tools.moderate import ModerateTool
from decision_mcp.tools.route import RouteTool
from decision_mcp.tools.triage import TriageTool


# ===========================================================================
# Bridge-level error tests
# ===========================================================================


def test_unknown_preset_raises_typed_error() -> None:
    """UnknownPresetError is a subclass of BridgeError and DecisionMCPError."""
    bridge = DecisionBridge(engine=FakeEngine(), preload=False)
    with pytest.raises(UnknownPresetError) as exc_info:
        bridge.predict("hello", preset="bogus")
    # UnknownPresetError should be catchable as BridgeError too
    assert isinstance(exc_info.value, BridgeError)
    assert isinstance(exc_info.value, DecisionMCPError)


def test_predict_wraps_upstream_runtime_error() -> None:
    """Any exception from the engine is wrapped as BridgeError."""
    bridge = DecisionBridge(
        engine=FakeEngine(), preload=False
    )
    bridge.engine.predict = MagicMock(side_effect=RuntimeError("GPU out of memory"))
    with pytest.raises(BridgeError) as exc_info:
        bridge.predict("hello", preset="guard")
    # Original exception is chained
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "GPU out of memory" in str(exc_info.value.__cause__)


def test_predict_custom_wraps_upstream_error() -> None:
    bridge = DecisionBridge(engine=FakeEngine(), preload=False)
    bridge.engine.predict = MagicMock(side_effect=ValueError("bad input"))
    with pytest.raises(BridgeError) as exc_info:
        bridge.predict_custom("hello", questions={})
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_bridge_error_message_never_contains_input() -> None:
    """Log context, not data: the message has the input length, not the input."""
    bridge = DecisionBridge(engine=FakeEngine(), preload=False)
    bridge.engine.predict = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(BridgeError) as exc_info:
        bridge.predict("super secret prompt", preset="guard")
    assert "super secret prompt" not in str(exc_info.value)
    assert "state_len=19" in str(exc_info.value)


def test_model_load_failure_wrapped() -> None:
    """An engine that fails to preload surfaces as ModelLoadError."""
    engine = FakeEngine(fail_preload=OSError("no such checkpoint"))
    with pytest.raises(ModelLoadError) as exc_info:
        DecisionBridge(engine=engine, preload=True)
    assert isinstance(exc_info.value.__cause__, OSError)


# ===========================================================================
# Tool-level error tests
# ===========================================================================


@pytest.mark.asyncio
async def test_guard_raises_on_non_dict_raw() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = "not a dict"
    with pytest.raises(ToolError, match="Expected dict"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_guard_raises_on_empty_dict() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = {}
    with pytest.raises(ToolError, match="empty result"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_guard_raises_on_missing_answers() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = {"model": "x"}
    with pytest.raises(ToolError, match="answers"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_guard_raises_on_wrong_answer_type() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(jailbreak=choice("yes"), prompt_injection=noul(0.1))
    with pytest.raises(ToolError, match="'noul' answer at 'jailbreak'"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_guard_raises_on_missing_value() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(
        jailbreak={"type": "noul", "answer_confidence": 0.9}, prompt_injection=noul(0.1)
    )
    with pytest.raises(ToolError, match="'noul'"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_guard_raises_on_missing_confidence() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(
        jailbreak={"type": "noul", "noul": 0.9}, prompt_injection=noul(0.1)
    )
    with pytest.raises(ToolError, match="answer_confidence"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_guard_raises_on_non_numeric_confidence() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(
        jailbreak={"type": "noul", "noul": 0.9, "answer_confidence": "not-a-number"},
        prompt_injection=noul(0.1),
    )
    with pytest.raises(ToolError, match="not numeric"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_guard_propagates_bridge_error() -> None:
    """BridgeError from upstream propagates through the tool unchanged."""
    bridge = MagicMock()
    bridge.predict.side_effect = BridgeError("upstream engine failed")
    with pytest.raises(BridgeError, match="upstream engine failed"):
        await GuardTool().run(bridge, prompt="test")


@pytest.mark.asyncio
async def test_triage_raises_on_missing_dimension() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(
        intent=choice("refund"),
        # is_urgent missing!
        refund_requested=noul(0.1),
        churn_risk=noul(0.1),
    )
    with pytest.raises(ToolError, match="frustration"):
        await TriageTool().run(bridge, text="...")


@pytest.mark.asyncio
async def test_moderate_raises_on_missing_dimension() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(
        toxic=noul(0.9),
        harassment=noul(0.1),
        # threat missing!
    )
    with pytest.raises(ToolError, match="threat"):
        await ModerateTool().run(bridge, text="...")


@pytest.mark.asyncio
async def test_email_raises_on_non_dict_at_key() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(
        category=choice("billing"),
        urgency="not-a-dict",  # malformed!
    )
    with pytest.raises(ToolError, match="urgency"):
        await EmailTool().run(bridge, body="...")


@pytest.mark.asyncio
async def test_route_raises_on_missing_probabilities() -> None:
    bridge = MagicMock()
    bridge.predict.return_value = result(
        difficulty={
            "type": "score",
            "score": 2.0,
            "legend": {"0": "a", "1": "b", "2": "c", "3": "d"},
            "answer_confidence": 0.5,
        }
    )
    with pytest.raises(ToolError, match="probabilities"):
        await RouteTool().run(bridge, prompt="...")


@pytest.mark.asyncio
async def test_route_raises_on_bad_preset() -> None:
    """Bridge rejects unknown presets; tool propagates the error."""
    bridge = MagicMock()
    bridge.predict.side_effect = UnknownPresetError("nope")
    with pytest.raises(UnknownPresetError):
        await RouteTool().run(bridge, prompt="...")


# ===========================================================================
# Exception hierarchy sanity check
# ===========================================================================


def test_all_custom_errors_inherit_from_decisionmcperror() -> None:
    """Catch DecisionMCPError to handle any project-specific failure."""
    for cls in [BridgeError, ModelLoadError, UnknownPresetError, ToolError]:
        assert issubclass(cls, DecisionMCPError), (
            f"{cls.__name__} must inherit from DecisionMCPError"
        )


def test_tool_error_carries_tool_name() -> None:
    err = ToolError("my_tool", "something went wrong")
    assert err.tool_name == "my_tool"
    assert "my_tool" in str(err)
    assert "something went wrong" in str(err)


def test_tool_error_chains_cause() -> None:
    original = ValueError("original problem")
    err = ToolError("my_tool", "wrapped", cause=original)
    assert err.cause is original
    assert err.__cause__ is original
    assert "original problem" in str(err)
