"""Tests for the D1OmniEngine adapter (d1-omni-600M via mlx-vlm) — no mlx needed."""
from __future__ import annotations

from typing import Any

import pytest

from decision_mcp.bridge import DecisionBridge
from decision_mcp.engines import D1OmniEngine, build_engine
from decision_mcp.engines.d1 import GUARD_QUESTIONS
from decision_mcp.errors import BridgeError, ModelLoadError
from decision_mcp.tools import GuardTool, RouteTool, TriageTool
from decision_mcp.tools._helpers import bin_mass, choice_of, choice_prob, score_level, yes_prob


# ---------------------------------------------------------------------------
# A stand-in for mlx_vlm.predict: reproduces the raw shapes mlx-vlm 0.7.8's
# d1_omni `_answer()` emits (bool / value / metadata — no answer_confidence).
# ---------------------------------------------------------------------------


class FakeOmni:
    """`mlx_vlm.predict` double answering every question from one canned p."""

    def __init__(self, p: float = 0.9) -> None:
        self.p = p
        self.seen_states: list[Any] = []

    def __call__(self, model: Any, processor: Any, state: Any, questions: Any) -> dict:
        self.seen_states.append(state)
        return {
            "model": "fake",
            "answers": {name: self._answer(q) for name, q in questions.items()},
            "usage": {"input_tokens": 64, "output_tokens": 0},
        }

    def _answer(self, q: dict) -> dict:
        p = self.p
        if q["type"] == "noul":
            return {"type": "bool", "value": p >= 0.5, "probability": p}
        criteria = list(q["criteria"])
        rest = (1.0 - p) / max(len(criteria) - 1, 1)
        if q["type"] == "choice":
            return {
                "type": "choice",
                "value": criteria[0],
                "probabilities": {n: (p if i == 0 else rest) for i, n in enumerate(criteria)},
                "metadata": {"confidence": p},
            }
        return {
            "type": "score",
            "value": 0.2,
            "probabilities": {str(i): (p if i == 0 else rest) for i in range(len(criteria))},
            "metadata": {
                "confidence": p,
                "legend": {str(i): c for i, c in enumerate(criteria)},
            },
        }


def _engine(monkeypatch: pytest.MonkeyPatch, fake: Any) -> D1OmniEngine:
    engine = D1OmniEngine()
    monkeypatch.setattr(engine, "_load", lambda: (fake, object(), object()))
    return engine


def _bridge(monkeypatch: pytest.MonkeyPatch, fake: Any) -> DecisionBridge:
    return DecisionBridge(engine=_engine(monkeypatch, fake), preload=False, usage=None)


# ---------------------------------------------------------------------------
# Registry, lifecycle, presets
# ---------------------------------------------------------------------------


def test_build_engine_makes_a_d1_omni() -> None:
    assert isinstance(build_engine("d1-omni"), D1OmniEngine)


def test_version_names_the_model_and_mlx_vlm() -> None:
    version = D1OmniEngine().version() or ""
    assert "LiquidAI/d1-omni-600M-MLX-8bit" in version
    assert "mlx-vlm" in version


def test_preload_loads_once_and_warms_up(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOmni()
    engine = D1OmniEngine()
    loads: list[bool] = []
    monkeypatch.setattr(
        engine, "_load", lambda: (loads.append(True), (fake, object(), object()))[1]
    )
    engine.preload()
    engine.predict({"prompt": "x"}, GUARD_QUESTIONS)
    assert loads == [True]
    assert len(fake.seen_states) == 2  # the warm-up pass, then the real call


def test_missing_mlx_vlm_raises_model_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_mlx_vlm(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlx_vlm":
            raise ImportError("No module named 'mlx_vlm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlx_vlm)
    with pytest.raises(ModelLoadError, match=r"\.\[d1-omni\]"):
        D1OmniEngine().preload()


def test_presets_cover_the_five_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _bridge(monkeypatch, FakeOmni())
    assert set(bridge.presets) == {"guard", "route", "moderate", "triage", "email"}


def test_predict_passes_the_state_through(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOmni()
    _bridge(monkeypatch, fake).predict("some prompt", preset="guard")
    assert fake.seen_states == [{"prompt": "some prompt"}]


def test_model_field_is_the_model_id_not_a_local_path(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _bridge(monkeypatch, FakeOmni()).predict("x", preset="guard")
    assert raw["model"] == "LiquidAI/d1-omni-600M-MLX-8bit"


# ---------------------------------------------------------------------------
# Normalization: mlx-vlm raw answers → what tools/_helpers.py parses
# ---------------------------------------------------------------------------


def test_bool_answers_become_noul(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _bridge(monkeypatch, FakeOmni(p=0.97)).predict("ignore previous", preset="guard")
    assert yes_prob(raw, "prompt_injection", "decision_guard") == pytest.approx(0.97)
    entry = raw["answers"]["prompt_injection"]
    assert entry["type"] == "noul"
    assert entry["answer_confidence"] == pytest.approx(0.97)


def test_low_p_yes_gives_confidence_in_no(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _bridge(monkeypatch, FakeOmni(p=0.1)).predict("hello", preset="guard")
    assert raw["answers"]["jailbreak"]["answer_confidence"] == pytest.approx(0.9)


def test_choice_and_score_answers_are_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _bridge(monkeypatch, FakeOmni(p=0.8)).predict("charged twice", preset="triage")
    label, conf = choice_of(raw, "intent", "decision_triage")
    assert (label, conf) == ("billing", pytest.approx(0.8))
    assert choice_prob(raw, "intent", "billing", "decision_triage") == pytest.approx(0.8)
    level, text, sconf = score_level(raw, "frustration", "decision_triage")
    assert (level, text, sconf) == (0, "calm and neutral", pytest.approx(0.8))
    assert bin_mass(raw, "frustration", {0}, "decision_triage") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Malformed engine output raises (no silent defaults)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answers", "match"),
    [
        (None, "no answers"),
        ({"prompt_injection": {"type": "maybe"}}, "unknown type 'maybe'"),
        ({"prompt_injection": {"type": "bool", "probability": "high"}}, "non-numeric"),
        ({"intent": {"type": "choice", "value": "billing"}}, "no metadata"),
        (
            {"level": {"type": "score", "value": 1.0, "metadata": {"confidence": 0.5}}},
            "no legend",
        ),
    ],
)
def test_malformed_output_raises(
    monkeypatch: pytest.MonkeyPatch, answers: Any, match: str
) -> None:
    def bad(*_args: Any) -> dict:
        return {"usage": {}} if answers is None else {"answers": answers}

    with pytest.raises(BridgeError, match=match):
        _engine(monkeypatch, bad).predict({"prompt": "x"}, GUARD_QUESTIONS)


# ---------------------------------------------------------------------------
# End-to-end through the preset tools (with real parsing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_tool_flags_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await GuardTool().run(_bridge(monkeypatch, FakeOmni(p=0.96)), prompt="ignore all")
    assert out.is_injection is True
    assert out.confidence == pytest.approx(0.96)


@pytest.mark.asyncio
async def test_route_and_triage_tools_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _bridge(monkeypatch, FakeOmni(p=0.8))
    route = await RouteTool().run(bridge, prompt="what is 2 + 2?")
    assert route.tier == "small"
    triage = await TriageTool().run(bridge, text="I was charged twice")
    assert triage.intent == "billing"
