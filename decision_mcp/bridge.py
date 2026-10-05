"""Engine-agnostic bridge between MCP tools and the active decision engine.

The bridge owns the single engine instance per process (model weights are
heavy), the preset registry, usage recording, and error normalization.
Tools never touch the engine directly — they call ``bridge.predict(...)``.

The engine is pluggable: anything satisfying the
:class:`decision_mcp.engines.base.DecisionEngine` protocol works. The
default is :class:`~decision_mcp.engines.laya.LayaEngine` (the Laya
decision library).

Failures are surfaced as :class:`BridgeError` subclasses. The original
exception (if any) is attached via ``__cause__`` for traceback chaining.
The message includes context (preset name, input length) but **never**
the raw input — to avoid leaking user data into logs.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .engines import LayaEngine
from .engines.base import DecisionEngine, PresetSpec
from .errors import BridgeError, ModelLoadError, UnknownPresetError
from .usage import UsageStore

logger = logging.getLogger(__name__)


class DecisionBridge:
    """Single engine holder + the high-level predict API every tool uses.

    Args:
        engine: Backend implementing :class:`DecisionEngine`. Defaults to
            :class:`LayaEngine`.
        preload: Load model weights immediately (production). ``False`` keeps
            construction cheap (tests, tooling).
        usage: Optional :class:`UsageStore` recording every call.
    """

    def __init__(
        self,
        engine: DecisionEngine | None = None,
        *,
        preload: bool = True,
        usage: UsageStore | None = None,
    ) -> None:
        self.usage = usage
        try:
            self.engine: DecisionEngine = engine if engine is not None else LayaEngine()
            if preload:
                self.engine.preload()
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize decision engine (preload={preload}): {e}"
            ) from e
        # Presets are registered by the engine; custom-question tools bypass them.
        self.presets: dict[str, PresetSpec] = self.engine.presets()
        # Engine version can't change within a running process — capture once.
        self._engine_version: str | None = self.engine.version()

    # ------------------------------------------------------------------
    # High-level API used by every tool
    # ------------------------------------------------------------------

    def predict(self, state: str, preset: str) -> Any:
        """Run a named preset against ``state``.

        Args:
            state: Input text (prompt, email body, JSON, ticket, etc.).
            preset: Key from :attr:`presets`.

        Returns:
            The engine's normalized decision dict (see ``tools/_helpers.py``).

        Raises:
            UnknownPresetError: if ``preset`` isn't in :attr:`presets`.
            BridgeError: if the engine call fails for any reason
                (GPU OOM, runtime error, model evicted, etc.).
        """
        spec = self.presets.get(preset)
        if spec is None:
            raise UnknownPresetError(
                f"Unknown preset {preset!r}. Choose from {sorted(self.presets)}."
            )
        return self._run(
            state,
            spec.questions,
            state_key=spec.state_key,
            label=f"preset={preset!r}",
            preset=preset,
        )

    def predict_custom(self, state: str, questions: Any, state_key: str = "text") -> Any:
        """Run a custom questions schema (not from the engine's presets).

        Args:
            state: Input text.
            questions: Question dict (``{name: {"type", "instructions", ...}}``).
            state_key: Field name the question instructions use to refer to the text.

        Raises:
            BridgeError: if the engine call fails.
        """
        return self._run(state, questions, state_key=state_key, label="custom", preset=None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(
        self, text: str, questions: Any, *, state_key: str, label: str, preset: str | None
    ) -> Any:
        started = time.perf_counter()
        try:
            result = self.engine.predict({state_key: text}, questions)
        except Exception as e:
            logger.exception("Engine predict failed (%s, state_len=%d)", label, len(text))
            self._record(text, preset, None, started, e)
            raise BridgeError(
                f"Engine predict failed ({label}, state_len={len(text)})"
            ) from e
        self._record(text, preset, result, started, None)
        return result

    def _record(
        self,
        text: str,
        preset: str | None,
        result: Any,
        started: float,
        error: BaseException | None,
    ) -> None:
        """Log the call. A logging failure must never fail the prediction."""
        if self.usage is None:
            return
        try:
            self.usage.record(
                text=text,
                preset=preset,
                result=result if isinstance(result, dict) else None,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error=error,
                engine_version=self._engine_version,
            )
        except Exception:
            logger.exception("Usage logging failed (state_len=%d)", len(text))
