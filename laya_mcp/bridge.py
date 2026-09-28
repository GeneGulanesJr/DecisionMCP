"""Wrapper around the upstream Laya library.

Holds a single ``Router`` instance so model weights are loaded exactly once
per process. Every tool in :mod:`laya_mcp.tools` calls ``bridge.predict(...)``.

Failures are surfaced as :class:`BridgeError` subclasses. The original
exception (if any) is attached via ``__cause__`` for traceback chaining.
The message includes context (preset name, input length) but **never**
the raw input — to avoid leaking user data into logs.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from laya import Router
from laya.presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)

from .errors import BridgeError, ModelLoadError, UnknownPresetError
from .usage import UsageStore

logger = logging.getLogger(__name__)


class LayaBridge:
    """Thin in-process wrapper around Laya.

    The :class:`laya.Router` picks the right checkpoint (English /
    multilingual / typed-decisions) per call based on script detection.
    Tools don't need to know about that — they just call ``predict``.
    """

    PRESETS: dict[str, Callable] = {
        "guard": guard_questions,
        "route": router_questions,
        "moderate": moderation_questions,
        "triage": triage_questions,
        "email": email_questions,
    }

    # Each preset's instructions refer to its input by name (e.g. "`prompt`"), so the
    # text must be passed as ``{key: text}``. Mirrors ``laya.cli.PRESET_STATE_KEYS``.
    PRESET_STATE_KEYS: dict[str, str] = {
        "guard": "prompt",
        "route": "request",
        "moderate": "post",
        "triage": "message",
        "email": "body",
    }

    def __init__(self, preload: bool = True, usage: UsageStore | None = None) -> None:
        self.usage = usage
        try:
            self.router = Router(preload=preload)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize Laya Router (preload={preload}): {e}"
            ) from e

    # ------------------------------------------------------------------
    # High-level API used by every tool
    # ------------------------------------------------------------------

    def predict(self, state: str, preset: str) -> Any:
        """Run a named preset against ``state``.

        Args:
            state: Input text (prompt, email body, JSON, ticket, etc.).
            preset: Key from :attr:`PRESETS`.

        Returns:
            Whatever Laya's ``Router.predict()`` returns (see ``tools/_helpers.py``).

        Raises:
            UnknownPresetError: if ``preset`` isn't in :attr:`PRESETS`.
            BridgeError: if the upstream Laya call fails for any reason
                (GPU OOM, runtime error, model evicted, etc.).
        """
        if preset not in self.PRESETS:
            raise UnknownPresetError(
                f"Unknown preset {preset!r}. Choose from {sorted(self.PRESETS)}."
            )
        return self._run(
            state,
            self.PRESETS[preset](),
            state_key=self.PRESET_STATE_KEYS[preset],
            label=f"preset={preset!r}",
            preset=preset,
        )

    def predict_custom(self, state: str, questions: Any, state_key: str = "text") -> Any:
        """Run a custom questions schema (not from upstream presets).

        Args:
            state: Input text.
            questions: Laya question dict (``{name: {"type", "instructions", ...}}``).
            state_key: Field name the question instructions use to refer to the text.

        Raises:
            BridgeError: if the upstream Laya call fails.
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
            result = self.router.predict({state_key: text}, questions)
        except Exception as e:
            logger.exception("Laya predict failed (%s, state_len=%d)", label, len(text))
            self._record(text, preset, None, started, e)
            raise BridgeError(
                f"Laya predict failed ({label}, state_len={len(text)})"
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
            )
        except Exception:
            logger.exception("Usage logging failed (state_len=%d)", len(text))
