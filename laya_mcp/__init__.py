"""LayaMCP — HTTP MCP server wrapping the Laya decision engine."""
from __future__ import annotations

import os
from pathlib import Path

# Keep model weights inside the project (./models) instead of ~/.cache/huggingface.
# Must run before huggingface_hub is imported (it reads this at import time), so it
# lives here rather than in config.py. An explicit HF_HUB_CACHE / HF_HOME still wins.
if "HF_HUB_CACHE" not in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HUB_CACHE"] = str(Path(__file__).resolve().parent.parent / "models")

__version__ = "0.1.0"
