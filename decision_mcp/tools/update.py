"""Tool: check for (and optionally apply) decision-engine updates.

Currently the only registered engine is Laya, so this tool upgrades the
``laya`` package and its model weights: ``check`` compares the installed
``laya`` package with PyPI, the locally cached model commit with the
Hugging Face Hub head, and looks for Laya models on the Hub that this
version doesn't know about.

``apply`` upgrades the package and re-downloads changed model weights. It runs
``pip install``, so it is disabled unless ``DECISIONMCP_ALLOW_UPDATES=true`` (the
HTTP server has no auth). Nothing takes effect in the running process; the
server must be restarted.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, constants as hf_constants, snapshot_download
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from ..bridge import DecisionBridge
from ..config import settings
from ..errors import ToolError
from .base import Tool

_PYPI_URL = "https://pypi.org/pypi/laya/json"
_HUB_AUTHOR = "convaiinnovations"


class UpdateInput(BaseModel):
    action: Literal["check", "apply"] = Field(
        "check",
        description=(
            "'check' reports what is out of date (read-only). 'apply' upgrades laya and "
            "re-downloads changed models; needs DECISIONMCP_ALLOW_UPDATES=true and a restart."
        ),
    )


class RepoStatus(BaseModel):
    repo: str
    models: list[str] = Field(..., description="Router checkpoint names served from this repo.")
    cached_commit: str | None = Field(..., description="Commit in the local cache (None = not downloaded).")
    remote_commit: str
    update_available: bool


class UpdateOutput(BaseModel):
    laya_installed: str
    laya_latest: str
    laya_update_available: bool
    model_repos: list[RepoStatus]
    new_models: list[str] = Field(
        default_factory=list,
        description="Laya models on the Hub that this laya version doesn't load. Report-only.",
    )
    applied: list[str] = Field(default_factory=list)
    restart_required: bool = False


def _known_models() -> tuple[dict[str, list[str]], set[str], set[str]]:
    """``({repo: [checkpoint names]}, all known repos, known subfolders)`` from installed laya."""
    from laya.router import DEFAULT_MODELS, STANDALONE_MODELS

    by_repo: dict[str, list[str]] = {}
    subfolders: set[str] = set()
    for name, spec in DEFAULT_MODELS.items():
        repo, sub = spec if isinstance(spec, tuple) else (spec, None)
        by_repo.setdefault(repo, []).append(name)
        if sub:
            subfolders.add(sub)
    all_repos = set(by_repo) | set(STANDALONE_MODELS.values())
    return by_repo, all_repos, subfolders


def _cached_commit(repo: str) -> str | None:
    ref = Path(hf_constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def _latest_laya(tool: str) -> str:
    try:
        with urllib.request.urlopen(_PYPI_URL, timeout=10) as resp:  # noqa: S310 (fixed https URL)
            version = json.load(resp)["info"]["version"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as e:
        raise ToolError(tool, "Could not read the latest laya version from PyPI", cause=e) from e
    if not isinstance(version, str):
        raise ToolError(tool, f"Unexpected PyPI version value: {version!r}")
    return version


def _newer(latest: str, installed: str, tool: str) -> bool:
    try:
        return Version(latest) > Version(installed)
    except InvalidVersion as e:
        raise ToolError(tool, f"Unparseable version ({installed!r} vs {latest!r})", cause=e) from e


def _check(tool: str) -> UpdateOutput:
    installed = metadata.version("laya")
    latest = _latest_laya(tool)
    by_repo, all_repos, known_subfolders = _known_models()

    api = HfApi()
    repos: list[RepoStatus] = []
    new_models: list[str] = []
    try:
        for repo, names in by_repo.items():
            remote = api.model_info(repo).sha
            if not remote:
                raise ToolError(tool, f"Hub returned no commit for {repo}")
            cached = _cached_commit(repo)
            repos.append(
                RepoStatus(
                    repo=repo,
                    models=names,
                    cached_commit=cached,
                    remote_commit=remote,
                    update_available=cached != remote,
                )
            )
            # A new model can be a new subfolder of a repo we already load...
            for path in api.list_repo_files(repo):
                head, _, tail = path.partition("/")
                if tail == "rl_agent_config.json" and head not in known_subfolders:
                    new_models.append(f"{repo}:{head}")
        # ...or a whole new Laya repo from the same publisher.
        for m in api.list_models(author=_HUB_AUTHOR, search="laya"):
            if m.id not in all_repos:
                new_models.append(m.id)
    except ToolError:
        raise
    except Exception as e:  # network / Hub API errors
        raise ToolError(tool, "Could not query the Hugging Face Hub", cause=e) from e

    return UpdateOutput(
        laya_installed=installed,
        laya_latest=latest,
        laya_update_available=_newer(latest, installed, tool),
        model_repos=repos,
        new_models=sorted(set(new_models)),
    )


def _install_laya(tool: str) -> None:
    """Upgrade the ``laya`` package in the running interpreter's environment."""
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable, "--upgrade", "laya"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "laya"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
    if proc.returncode != 0:
        raise ToolError(tool, f"Installing laya failed: {proc.stderr.strip()[-500:]}")


def _apply(tool: str) -> UpdateOutput:
    if not settings.allow_updates:
        raise ToolError(
            tool,
            "Updates are disabled. Set DECISIONMCP_ALLOW_UPDATES=true and restart to allow 'apply'.",
        )
    status = _check(tool)
    applied: list[str] = []
    if status.laya_update_available:
        _install_laya(tool)
        applied.append(f"laya {status.laya_installed} -> {status.laya_latest}")
    for repo in status.model_repos:
        if repo.update_available:
            try:
                snapshot_download(repo_id=repo.repo)
            except Exception as e:
                raise ToolError(tool, f"Downloading {repo.repo} failed", cause=e) from e
            applied.append(f"downloaded {repo.repo}@{repo.remote_commit[:8]}")
    status.applied = applied
    status.restart_required = bool(applied)
    return status


class UpdateTool(Tool):
    name = "decision_update"
    description = (
        "Check whether the decision engine (Laya library) or its models are out of date and "
        "whether new Laya models exist on the Hugging Face Hub. action='apply' upgrades "
        "(opt-in, needs restart)."
    )
    input_schema = UpdateInput
    output_schema = UpdateOutput

    async def run(self, bridge: DecisionBridge, action: str = "check") -> UpdateOutput:
        # Network + subprocess work: keep it off the event loop.
        fn = _apply if action == "apply" else _check
        return await asyncio.to_thread(fn, self.name)
