"""Tests for LayaBridge (the upstream Laya wrapper)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from laya_fixtures import noul, result

from laya_mcp.bridge import LayaBridge
from laya_mcp.errors import UnknownPresetError
from laya_mcp.usage import UsageStore, current_tool


def _bridge(usage: UsageStore | None = None, predict=None) -> LayaBridge:
    bridge = LayaBridge(preload=False, usage=usage)
    bridge.router = MagicMock(predict=predict or MagicMock(return_value=result(a=noul(0.9))))
    return bridge


def test_presets_are_registered() -> None:
    expected = {"guard", "route", "moderate", "triage", "email"}
    assert set(LayaBridge.PRESETS) == expected


def test_every_preset_has_a_state_key() -> None:
    assert set(LayaBridge.PRESET_STATE_KEYS) == set(LayaBridge.PRESETS)


def test_predict_rejects_unknown_preset() -> None:
    bridge = LayaBridge(preload=False)
    with pytest.raises(UnknownPresetError, match="Unknown preset"):
        bridge.predict("hello", preset="bogus")


def test_bridge_exposes_router() -> None:
    """Bridge should expose a Router instance after construction."""
    bridge = LayaBridge(preload=False)
    assert bridge.router is not None


@pytest.mark.parametrize(
    "preset,key",
    [("guard", "prompt"), ("route", "request"), ("moderate", "post"),
     ("triage", "message"), ("email", "body")],
)
def test_predict_wraps_text_in_the_presets_state_key(preset: str, key: str) -> None:
    """Preset instructions name their input (e.g. `prompt`), so the text goes in that field."""
    bridge = _bridge()
    bridge.predict("hello", preset=preset)
    state, questions = bridge.router.predict.call_args.args
    assert state == {key: "hello"}
    assert questions == LayaBridge.PRESETS[preset]()


def test_predict_custom_wraps_text_in_given_state_key() -> None:
    bridge = _bridge()
    questions = {"q": {"type": "noul", "instructions": "Is `diff` big?"}}
    out = bridge.predict_custom("hello", questions, state_key="diff")
    bridge.router.predict.assert_called_once_with({"diff": "hello"}, questions)
    assert out["answers"]["a"]["noul"] == 0.9


# ===========================================================================
# Usage logging
# ===========================================================================


def test_predict_records_usage(tmp_path) -> None:
    store = UsageStore(tmp_path / "usage.db")
    bridge = _bridge(store)
    token = current_tool.set("laya_guard")
    try:
        bridge.predict("hello", preset="guard")
    finally:
        current_tool.reset(token)
    stats = store.stats()
    assert stats["calls"] == 1
    assert stats["by_tool"] == {"laya_guard": 1}
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
