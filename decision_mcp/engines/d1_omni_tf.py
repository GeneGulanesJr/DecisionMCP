"""Engine: LiquidAI ``d1-omni-600M`` via transformers (CUDA/CPU, e.g. Windows).

The transformers twin of :mod:`.d1_omni` (whose MLX export only runs on Apple
Silicon). Like d1-3B it answers named ``noul`` / ``choice`` / ``score``
questions over a state in one forward pass with zero generated tokens, so it
reuses the d1 engine's preset question schemas and normalizer unchanged.

Unlike the text-only d1-3B, d1-omni is an omni (vision+audio) model: its
checkpoint's ``system_one`` also accepts an ``images`` list next to the state
(see the ``FakeD1Model`` double in the test suite). This engine therefore
declares ``supports_images = True`` and forwards ``state["images"]`` — a list
of base64-encoded JPEG frames that vision tools attach — into that kwarg.

The heavy imports (``torch``, ``transformers``) happen inside
:meth:`D1OmniTFEngine._load_model`, so ``import decision_mcp`` — and the whole
mocked test suite — works without them installed. Install with the new extra::

    pip install -e ".[d1-omni-tf]"

Note: the exact media contract of the remote-code checkpoint should be probed
once after installing torch (``python -m decision_mcp.probe_d1omni``); until
then treat image support as best-effort on this engine.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from typing import Any

import numpy as np

from ..errors import ModelLoadError
from .base import PresetSpec
from .d1 import _PRESETS, _normalize

DEFAULT_MODEL_ID = "LiquidAI/d1-omni-600M"


class D1OmniTFEngine:
    """Adapter implementing :class:`decision_mcp.engines.base.DecisionEngine`.

    Cheap to construct; weights load on the first :meth:`preload` or
    :meth:`predict` call.
    """

    name = "d1-omni-tf"
    supports_images = True
    supports_audio = True

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> None:
        self._model_id = model_id
        self._device = device
        self._model: Any | None = None

    # -- lifecycle ----------------------------------------------------------

    def preload(self) -> None:
        self._ensure_model()

    def version(self) -> str | None:
        try:
            transformers_version = _metadata_version("transformers")
        except PackageNotFoundError:
            transformers_version = "unknown"
        return f"{self._model_id} (transformers {transformers_version})"

    def presets(self) -> dict[str, PresetSpec]:
        return dict(_PRESETS)

    def _load_model(self) -> Any:
        """Download/load the checkpoint. Heavy imports live here on purpose."""
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            raise ModelLoadError(
                f"The d1-omni-tf engine needs the transformers stack ({e}); "
                "install it with: pip install -e '.[d1-omni-tf]'"
            ) from e
        device = self._device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        try:
            # The checkpoint ships its own modeling code, hence trust_remote_code.
            model = AutoModel.from_pretrained(
                self._model_id, trust_remote_code=True, dtype=dtype
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load d1-omni model {self._model_id!r}: {e}"
            ) from e
        return model.to(device).eval()

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    # -- inference ----------------------------------------------------------

    def predict(self, state: dict[str, Any], questions: Any) -> dict:
        """Answer ``questions`` about ``state`` (plus images/audio) in one pass.

        ``state`` is ``{state_key: text}`` plus optional media keys the bridge
        inserted for vision/audio-capable engines: ``"images"`` (base64 JPEG
        frames) and ``"audio"`` (mono 16 kHz numpy array, int16 or float).
        The checkpoint JSON-serializes the state as text, so media keys are
        removed here — media travel exclusively through the ``images`` /
        ``audio`` kwargs (decoded: PIL images; the checkpoint cuts audio at
        30 s).
        """
        model = self._ensure_model()

        state = dict(state)
        raw_b64 = state.pop("images", None)
        audio = state.pop("audio", None)

        if raw_b64:
            import base64
            import io

            from PIL import Image

            images = []
            for b64 in raw_b64:
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                images.append(img)
        else:
            images = None

        if audio is not None:
            audio = np.asarray(audio)
            if audio.dtype not in (np.int16, np.float32, np.float64):
                raise ModelLoadError(
                    "audio must be int16 or float samples (16 kHz mono), got "
                    f"{audio.dtype}"
                )

        raw = model.system_one(state, questions, images=images, audio=audio)
        raw["model"] = self._model_id
        return _normalize(raw, questions)
