"""Engine: LiquidAI ``d1-omni-600M`` on Apple Silicon through MLX (``mlx-vlm``).

https://huggingface.co/LiquidAI/d1-omni-600M-MLX-8bit — an 8-bit MLX export of
d1-omni-600M (0.73 GB; vision tower and decision scorer kept in float16). Like
d1-3B it answers named ``noul`` / ``choice`` / ``score`` questions over a state
in one forward pass with zero generated tokens, so it reuses the d1 engine's
preset question schemas unchanged.

The heavy import (``mlx_vlm``, which pulls ``mlx``) happens inside
:meth:`D1OmniEngine._load`, so ``import decision_mcp`` — and the whole mocked
test suite — works without it installed. Install with the ``d1-omni`` extra
(Apple Silicon only)::

    pip install -e ".[d1-omni]"

mlx-vlm answers in its own shape (``bool`` / ``value`` / ``metadata``);
:func:`_normalize` rewrites it into the shape the tool parsers in
:mod:`decision_mcp.tools._helpers` read.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import Any

from ..errors import BridgeError, ModelLoadError
from .base import PresetSpec
from .d1 import _PRESETS  # same Decision Index question schemas as d1-3B

DEFAULT_MODEL_ID = "LiquidAI/d1-omni-600M-MLX-8bit"

# One tiny question run in preload(): the first MLX forward pass compiles
# kernels (~3 s on an M4), which would otherwise land on the first tool call.
_WARMUP_QUESTIONS: dict[str, Any] = {
    "warm": {"type": "noul", "instructions": "Is `text` a greeting?"},
}


# ---------------------------------------------------------------------------
# Answer normalization: mlx-vlm d1-omni shape → the shape tools/_helpers.py parses.
# ---------------------------------------------------------------------------


def _as_probability(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeError(f"d1-omni returned a non-numeric probability for {context}: {value!r}")
    return float(value)


def _normalize(raw: Any) -> dict:
    """Validate mlx-vlm's d1-omni result and rewrite each answer in place.

    mlx-vlm emits::

        bool:   {"type": "bool", "value": <bool>, "probability": <P(yes)>}
        choice: {"type": "choice", "value": <label>, "probabilities": {...},
                 "metadata": {"confidence": c}}
        score:  {"type": "score", "value": <expected level>, "probabilities": {...},
                 "metadata": {"confidence": c, "legend": {...}}}

    and the tools read ``noul`` / ``choice`` / ``score`` value keys, a top-level
    ``legend`` and ``answer_confidence`` on every answer. The original keys are
    kept alongside.

    Raises:
        BridgeError: if the result is not that shape.
    """
    if not isinstance(raw, dict):
        raise BridgeError(f"d1-omni returned {type(raw).__name__}, expected a dict")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise BridgeError(f"d1-omni returned no answers (got keys: {sorted(raw)})")
    for name, entry in answers.items():
        if not isinstance(entry, dict):
            raise BridgeError(f"d1-omni answer {name!r} is {type(entry).__name__}, expected a dict")
        kind = entry.get("type")
        if kind in ("bool", "noul"):
            p = _as_probability(entry.get("probability"), f"noul answer {name!r}")
            conf = max(p, 1.0 - p)
            entry.update(type="noul", noul=p, confidence=conf, answer_confidence=conf)
            continue
        if kind not in ("choice", "score"):
            raise BridgeError(
                f"d1-omni answer {name!r} has unknown type {kind!r} (expected bool/choice/score)"
            )
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            raise BridgeError(f"d1-omni {kind} answer {name!r} has no metadata: {entry!r}")
        conf = _as_probability(metadata.get("confidence"), f"{kind} answer {name!r}")
        if kind == "choice":
            entry.update(choice=entry.get("value"))
        else:
            if not isinstance(metadata.get("legend"), dict):
                raise BridgeError(f"d1-omni score answer {name!r} has no legend: {entry!r}")
            score = _as_probability(entry.get("value"), f"score answer {name!r}")
            entry.update(score=score, legend=metadata["legend"])
        entry.update(confidence=conf, answer_confidence=conf)
    return raw


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class D1OmniEngine:
    """Adapter implementing :class:`decision_mcp.engines.base.DecisionEngine`.

    Cheap to construct; weights load on the first :meth:`preload` or
    :meth:`predict` call.
    """

    name = "d1-omni"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self._model_id = model_id
        self._loaded: tuple[Any, Any, Any] | None = None  # (predict fn, model, processor)

    # -- lifecycle ----------------------------------------------------------

    def preload(self) -> None:
        predict, model, processor = self._ensure_loaded()
        _normalize(predict(model, processor, {"text": "hello"}, _WARMUP_QUESTIONS))

    def version(self) -> str | None:
        try:
            mlx_vlm_version = _metadata_version("mlx-vlm")
        except PackageNotFoundError:
            mlx_vlm_version = "unknown"
        return f"{self._model_id} (mlx-vlm {mlx_vlm_version})"

    def presets(self) -> dict[str, PresetSpec]:
        return dict(_PRESETS)

    def _load(self) -> tuple[Any, Any, Any]:
        """Download/load the checkpoint. The heavy import lives here on purpose."""
        try:
            from mlx_vlm import load, predict
        except ImportError as e:
            raise ModelLoadError(
                f"The d1-omni engine needs mlx-vlm>=0.7.8 on Apple Silicon ({e}); "
                "install it with: pip install -e '.[d1-omni]'"
            ) from e
        try:
            model, processor = load(self._model_id)
        except Exception as e:
            raise ModelLoadError(f"Failed to load d1-omni model {self._model_id!r}: {e}") from e
        return predict, model, processor

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    # -- inference ----------------------------------------------------------

    def predict(self, state: dict[str, Any], questions: Any) -> dict:
        """Answer ``questions`` about ``state`` in one forward pass.

        ``state`` is ``{state_key: text}``; d1-omni serializes a dict state as
        JSON and the question instructions refer to it by field name.
        """
        predict, model, processor = self._ensure_loaded()
        raw = _normalize(predict(model, processor, state, questions))
        raw["model"] = self._model_id  # mlx-vlm reports the local snapshot path
        return raw
