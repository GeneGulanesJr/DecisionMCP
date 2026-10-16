"""Pluggable decision engines. See :class:`base.DecisionEngine` for the contract."""
from __future__ import annotations

from ..errors import BridgeError
from .base import DecisionEngine, PresetSpec
from .d1 import D1Engine
from .d1_omni import D1OmniEngine
from .laya import LayaEngine

# Registry backing the DECISIONMCP_ENGINE setting. Constructing an engine is
# cheap — heavy imports and weights live in each engine's preload path.
ENGINES: dict[str, type] = {
    LayaEngine.name: LayaEngine,
    D1Engine.name: D1Engine,
    D1OmniEngine.name: D1OmniEngine,
}


def build_engine(name: str) -> DecisionEngine:
    """Build the engine registered under ``name`` (weights load in ``preload()``).

    Raises:
        BridgeError: if ``name`` is not in :data:`ENGINES`.
    """
    try:
        return ENGINES[name]()
    except KeyError:
        raise BridgeError(
            f"Unknown engine {name!r}. Choose from {sorted(ENGINES)} (DECISIONMCP_ENGINE)."
        ) from None


__all__ = [
    "DecisionEngine",
    "D1Engine",
    "D1OmniEngine",
    "ENGINES",
    "LayaEngine",
    "PresetSpec",
    "build_engine",
]
