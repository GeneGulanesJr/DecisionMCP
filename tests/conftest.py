"""Shared fixtures + a minimal engine double for tests."""
from __future__ import annotations

import pytest

from decision_mcp.engines.base import PresetSpec


class FakeEngine:
    """Minimal DecisionEngine double — no laya, no weights, deterministic.

    Presets mirror the Laya engine's name → state_key mapping so bridge-level
    tests exercise the same lookup path as production.
    """

    name = "fake"

    _PRESET_KEYS = {
        "guard": "prompt",
        "route": "request",
        "moderate": "post",
        "triage": "message",
        "email": "body",
    }

    def __init__(self, fail_preload: Exception | None = None) -> None:
        self.preloaded = False
        self._fail_preload = fail_preload

    def preload(self) -> None:
        if self._fail_preload is not None:
            raise self._fail_preload
        self.preloaded = True

    def version(self) -> str | None:
        return "0.0.0"

    def presets(self) -> dict[str, PresetSpec]:
        return {
            name: PresetSpec(
                name=name,
                questions={"a": {"type": "noul", "instructions": f"Answer yes/no about `{key}`."}},
                state_key=key,
            )
            for name, key in self._PRESET_KEYS.items()
        }

    def predict(self, state: dict, questions) -> dict:
        return {"answers": {}}


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def fake_bridge():
    """A bridge stand-in for tests.

    Tests use ``unittest.mock.MagicMock`` directly rather than this fixture
    when they want to control ``predict.return_value``. This fixture exists
    in case future tests need a real bridge with ``preload=False``.
    """
    from decision_mcp.bridge import DecisionBridge

    return DecisionBridge(engine=FakeEngine(), preload=False)
