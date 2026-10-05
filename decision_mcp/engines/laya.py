"""The default engine: the Laya decision library (``laya.Router``).

Laya picks the right checkpoint (English / multilingual / typed-decisions)
per call based on script detection. ``laya`` is imported lazily, inside the
methods that need it, so ``import decision_mcp`` — and the whole mocked test
suite — works without the (heavy) library installed.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import Any

from .base import PresetSpec


class LayaEngine:
    """Adapter implementing :class:`decision_mcp.engines.base.DecisionEngine`."""

    name = "laya"

    def __init__(self) -> None:
        from laya import Router  # lazy: keeps the heavy import out of tests/tools

        self._router = Router(preload=False)

    def preload(self) -> None:
        self._router.preload()

    def version(self) -> str | None:
        try:
            return _metadata_version("laya")
        except PackageNotFoundError:
            return None

    def presets(self) -> dict[str, PresetSpec]:
        from laya.presets import (
            email_questions,
            guard_questions,
            moderation_questions,
            router_questions,
            triage_questions,
        )

        return {
            "guard": PresetSpec(name="guard", questions=guard_questions(), state_key="prompt"),
            "route": PresetSpec(name="route", questions=router_questions(), state_key="request"),
            "moderate": PresetSpec(
                name="moderate", questions=moderation_questions(), state_key="post"
            ),
            "triage": PresetSpec(
                name="triage", questions=triage_questions(), state_key="message"
            ),
            "email": PresetSpec(name="email", questions=email_questions(), state_key="body"),
        }

    def predict(self, state: dict[str, Any], questions: Any) -> Any:
        return self._router.predict(state, questions)
