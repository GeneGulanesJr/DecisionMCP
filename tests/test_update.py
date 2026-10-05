"""Tests for the decision_update tool. Network, pip and downloads are all mocked."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from decision_mcp.errors import ToolError
from decision_mcp.tools import update
from decision_mcp.tools.update import UpdateTool

REPO = "convaiinnovations/laya"
_KNOWN_REPOS = {"convaiinnovations/laya", "convaiinnovations/laya-multilingual", "convaiinnovations/laya-typed-decisions"}


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Fake PyPI + Hub + engine model registry. Defaults: everything up to date."""
    state = SimpleNamespace(
        installed="0.3.21",
        latest="0.3.21",
        remote="a" * 40,
        cached="a" * 40,
        files=[
            "model.safetensors",
            "rl_agent_config.json",
            "multilingual/rl_agent_config.json",
            "typed-decisions/rl_agent_config.json",
        ],
        hub_models=[REPO, "convaiinnovations/laya-multilingual", "convaiinnovations/laya-typed-decisions"],
    )
    monkeypatch.setattr(update.metadata, "version", lambda name: state.installed)
    monkeypatch.setattr(update, "_latest_laya", lambda tool: state.latest)
    monkeypatch.setattr(update, "_cached_commit", lambda repo: state.cached)
    # _known_models lazily imports the installed laya; fake its registry view.
    monkeypatch.setattr(
        update,
        "_known_models",
        lambda: (
            {"convaiinnovations/laya": ["english", "multilingual", "typed-decisions"]},
            set(_KNOWN_REPOS),
            {"multilingual", "typed-decisions"},
        ),
    )
    api = MagicMock()
    api.model_info.side_effect = lambda repo: SimpleNamespace(sha=state.remote)
    api.list_repo_files.side_effect = lambda repo: state.files
    api.list_models.side_effect = lambda **kw: [SimpleNamespace(id=i) for i in state.hub_models]
    monkeypatch.setattr(update, "HfApi", lambda: api)
    state.api = api
    return state


@pytest.mark.asyncio
async def test_check_reports_up_to_date(hub) -> None:
    out = await UpdateTool().run(None, action="check")
    assert out.laya_update_available is False
    assert out.new_models == []
    assert [r.update_available for r in out.model_repos] == [False]
    assert out.model_repos[0].models == ["english", "multilingual", "typed-decisions"]
    assert out.applied == [] and out.restart_required is False


@pytest.mark.asyncio
async def test_check_detects_newer_laya_and_model_commit(hub) -> None:
    hub.latest = "0.4.0"
    hub.remote = "b" * 40
    out = await UpdateTool().run(None, action="check")
    assert out.laya_update_available is True
    assert out.model_repos[0].update_available is True
    assert out.model_repos[0].remote_commit == "b" * 40


@pytest.mark.asyncio
async def test_check_flags_missing_local_download(hub) -> None:
    hub.cached = None
    out = await UpdateTool().run(None, action="check")
    assert out.model_repos[0].cached_commit is None
    assert out.model_repos[0].update_available is True


@pytest.mark.asyncio
async def test_check_does_not_flag_older_pypi_as_update(hub) -> None:
    hub.installed, hub.latest = "0.3.21", "0.3.5"
    assert (await UpdateTool().run(None, action="check")).laya_update_available is False


@pytest.mark.asyncio
async def test_check_finds_new_subfolder_and_new_repo(hub) -> None:
    hub.files += ["reasoning/rl_agent_config.json", "reasoning/encoder/config.json"]
    hub.hub_models += ["convaiinnovations/laya-code"]
    out = await UpdateTool().run(None, action="check")
    assert sorted(out.new_models) == sorted([f"{REPO}:reasoning", "convaiinnovations/laya-code"])


@pytest.mark.asyncio
async def test_check_raises_when_hub_unreachable(hub) -> None:
    hub.api.model_info.side_effect = OSError("network down")
    with pytest.raises(ToolError, match="Hugging Face Hub"):
        await UpdateTool().run(None, action="check")


def test_latest_laya_raises_when_pypi_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(update.urllib.request, "urlopen", boom)
    with pytest.raises(ToolError, match="PyPI"):
        update._latest_laya("decision_update")


@pytest.mark.asyncio
async def test_apply_refused_unless_enabled(hub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update.settings, "allow_updates", False)
    install = MagicMock()
    monkeypatch.setattr(update, "_install_laya", install)
    with pytest.raises(ToolError, match="DECISIONMCP_ALLOW_UPDATES"):
        await UpdateTool().run(None, action="apply")
    install.assert_not_called()


@pytest.mark.asyncio
async def test_apply_upgrades_and_downloads(hub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update.settings, "allow_updates", True)
    hub.latest, hub.remote = "0.4.0", "b" * 40
    install, download = MagicMock(), MagicMock()
    monkeypatch.setattr(update, "_install_laya", install)
    monkeypatch.setattr(update, "snapshot_download", download)
    out = await UpdateTool().run(None, action="apply")
    install.assert_called_once()
    download.assert_called_once_with(repo_id=REPO)
    assert out.applied == ["laya 0.3.21 -> 0.4.0", f"downloaded {REPO}@{'b' * 8}"]
    assert out.restart_required is True


@pytest.mark.asyncio
async def test_apply_when_up_to_date_does_nothing(hub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update.settings, "allow_updates", True)
    install, download = MagicMock(), MagicMock()
    monkeypatch.setattr(update, "_install_laya", install)
    monkeypatch.setattr(update, "snapshot_download", download)
    out = await UpdateTool().run(None, action="apply")
    install.assert_not_called()
    download.assert_not_called()
    assert out.applied == [] and out.restart_required is False


def test_known_models_from_installed_laya() -> None:
    """The real registry view (gated: needs the laya extra installed)."""
    pytest.importorskip("laya")
    by_repo, all_repos, subfolders = update._known_models()
    assert by_repo["convaiinnovations/laya"] == ["english", "multilingual", "typed-decisions"]
    assert "convaiinnovations/laya-multilingual" in all_repos
    assert "multilingual" in subfolders


def test_install_laya_reports_pip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = SimpleNamespace(returncode=1, stderr="no matching distribution")
    monkeypatch.setattr(update.subprocess, "run", lambda *a, **k: proc)
    with pytest.raises(ToolError, match="no matching distribution"):
        update._install_laya("decision_update")


def test_install_laya_only_ever_installs_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    update._install_laya("decision_update")
    assert seen["cmd"][-2:] == ["--upgrade", "laya"]
