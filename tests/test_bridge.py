"""Tests for DecisionBridge (the engine-agnostic bridge layer)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from conftest import FakeEngine
from fixtures import noul, result

from decision_mcp.bridge import DecisionBridge
from decision_mcp.errors import UnknownPresetError
from decision_mcp.usage import UsageStore, current_tool


def _bridge(usage: UsageStore | None = None, predict=None) -> DecisionBridge:
    engine = FakeEngine()
    engine.predict = predict if predict is not None else MagicMock(return_value=result(a=noul(0.9)))
    return DecisionBridge(engine=engine, preload=False, usage=usage)


def test_presets_are_registered() -> None:
    expected = {"guard", "route", "moderate", "triage", "email"}
    assert set(_bridge().presets) == expected


def test_every_preset_has_a_state_key() -> None:
    bridge = _bridge()
    assert {s.state_key for s in bridge.presets.values()} == {
        "prompt", "request", "post", "message", "body",
    }


def test_bridge_uses_the_injected_engine() -> None:
    """Any DecisionEngine implementation plugs in — the bridge is not Laya-bound."""
    engine = FakeEngine()
    bridge = DecisionBridge(engine=engine, preload=False)
    assert bridge.engine is engine


def test_preload_is_forwarded_to_the_engine() -> None:
    engine = FakeEngine()
    DecisionBridge(engine=engine, preload=True)
    assert engine.preloaded is True


def test_default_engine_is_laya() -> None:
    pytest.importorskip("laya")
    bridge = DecisionBridge(preload=False)
    assert bridge.engine.name == "laya"


def test_predict_rejects_unknown_preset() -> None:
    bridge = _bridge()
    with pytest.raises(UnknownPresetError, match="Unknown preset"):
        bridge.predict("hello", preset="bogus")


def test_bridge_exposes_engine() -> None:
    """Bridge should expose an engine instance after construction."""
    assert _bridge().engine is not None


@pytest.mark.parametrize(
    "preset,key",
    [("guard", "prompt"), ("route", "request"), ("moderate", "post"),
     ("triage", "message"), ("email", "body")],
)
def test_predict_wraps_text_in_the_presets_state_key(preset: str, key: str) -> None:
    """Preset instructions name their input (e.g. `prompt`), so the text goes in that field."""
    bridge = _bridge()
    bridge.predict("hello", preset=preset)
    state, questions = bridge.engine.predict.call_args.args
    assert state == {key: "hello"}
    assert questions == bridge.presets[preset].questions


def test_predict_custom_wraps_text_in_given_state_key() -> None:
    bridge = _bridge()
    questions = {"q": {"type": "noul", "instructions": "Is `diff` big?"}}
    out = bridge.predict_custom("hello", questions, state_key="diff")
    bridge.engine.predict.assert_called_once_with({"diff": "hello"}, questions)
    assert out["answers"]["a"]["noul"] == 0.9


# ===========================================================================
# Usage logging
# ===========================================================================


def test_predict_records_usage(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.db")
    bridge = _bridge(store)
    token = current_tool.set("decision_guard")
    try:
        bridge.predict("hello", preset="guard")
    finally:
        current_tool.reset(token)
    stats = store.stats()
    assert stats["calls"] == 1
    assert stats["by_tool"] == {"decision_guard": 1}
    assert stats["input_tokens"] == 12


def test_failed_predict_is_recorded_as_error(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.db")
    bridge = _bridge(store, predict=MagicMock(side_effect=RuntimeError("boom")))
    with pytest.raises(Exception):
        bridge.predict("hello", preset="guard")
    assert store.stats()["errors"] == 1


def test_usage_failure_never_breaks_prediction() -> None:
    store = MagicMock()
    store.record.side_effect = RuntimeError("disk full")
    bridge = _bridge(store)
    out = bridge.predict("hello", preset="guard")
    assert out["answers"]["a"]["noul"] == 0.9
    store.record.assert_called_once()


def test_engine_version_is_recorded_with_each_call(tmp_path) -> None:
    """Per-row provenance: the engine's version travels with every usage row."""
    store = UsageStore(tmp_path / "usage.db")
    bridge = _bridge(store)
    bridge.predict("hello", preset="guard")
    assert store.stats()["by_engine_version"] == {"0.0.0": 1}
