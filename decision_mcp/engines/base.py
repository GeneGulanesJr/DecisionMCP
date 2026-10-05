"""Engine contract: the seam between DecisionMCP and a decision model.

Any object satisfying :class:`DecisionEngine` can serve as the backend of
:class:`decision_mcp.bridge.DecisionBridge`. Engines own model weights and
answer typed question schemas; they know nothing about MCP, tools, or usage
logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PresetSpec:
    """A named question schema the engine can answer.

    Attributes:
        name: Registry key used by preset tools (e.g. ``"guard"``).
        questions: Question dict (``{name: {"type", "instructions", ...}}``),
            already evaluated.
        state_key: Field name the question instructions use to refer to the
            input text (e.g. ``"prompt"``); the bridge passes
            ``{state_key: text}`` to :meth:`DecisionEngine.predict`.
    """

    name: str
    questions: Any
    state_key: str


@runtime_checkable
class DecisionEngine(Protocol):
    """Minimal contract for a pluggable decision engine.

    Implementations must be cheap to construct (no weights loaded) and load
    them in :meth:`preload` instead — the bridge only calls ``preload`` when
    actually serving. ``predict`` returns the normalized decision shape::

        {"answers": {q: {"type": "choice"|"noul"|"score", ...}},
         "usage": {...}, "routing": {...}}

    which is what :mod:`decision_mcp.tools._helpers` parses.
    """

    name: str

    def preload(self) -> None:
        """Load model weights. Called once when the bridge is built serving."""
        ...

    def version(self) -> str | None:
        """Engine package version, recorded per usage row for provenance."""
        ...

    def presets(self) -> dict[str, PresetSpec]:
        """Named presets this engine offers (may be empty)."""
        ...

    def predict(self, state: dict[str, Any], questions: Any) -> Any:
        """Answer ``questions`` about ``state`` (see class docstring for shape)."""
        ...
