"""Tests for the usage/session log and the decision_usage tool."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fixtures import choice, result

from decision_mcp.errors import ToolError
from decision_mcp.tools.usage_report import UsageTool
from decision_mcp.usage import UsageStore, current_session, current_tool


def _record(store: UsageStore, text: str = "hello", tool: str = "decision_guard", session: str = "s1"):
    t, s = current_tool.set(tool), current_session.set(session)
    try:
        return store.record(
            text=text,
            preset="guard",
            result=result(topic=choice("coding")),
            elapsed_ms=12.345,
            engine_version="1.2.3",
        )
    finally:
        current_tool.reset(t)
        current_session.reset(s)


def test_text_not_stored_by_default(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db")
    _record(store, text="my private prompt")
    out = tmp_path / "out.jsonl"
    assert store.export_jsonl(out) == 1
    row = json.loads(out.read_text())
    assert row["input"] is None
    assert row["answers"]["topic"]["choice"] == "coding"
    assert row["tool"] == "decision_guard"
    assert row["session_id"] == "s1"
    assert row["model"] == "english"
    assert "my private prompt" not in out.read_text()


def test_export_jsonl_labeled_only(tmp_path) -> None:
    """labeled_only drops rows without stored text (training-ready pairs only)."""
    store = UsageStore(tmp_path / "u.db", store_text=True)
    _record(store, text="train on me", session="s-keep")
    # decision_secret_risk text is never stored, so its row has no label input.
    _record(store, text="secret", tool="decision_secret_risk", session="s-drop")
    out = tmp_path / "labeled.jsonl"
    assert store.export_jsonl(out, labeled_only=True) == 1
    row = json.loads(out.read_text())
    assert row["input"] == "train on me"
    assert row["session_id"] == "s-keep"
    # plain export still returns both rows
    assert store.export_jsonl(tmp_path / "all.jsonl") == 2


def test_engine_version_recorded(tmp_path) -> None:
    """Every row carries the engine version that produced the labels."""
    store = UsageStore(tmp_path / "u.db")
    _record(store)
    out = tmp_path / "v.jsonl"
    store.export_jsonl(out)
    row = json.loads(out.read_text())
    assert row["engine_version"] == "1.2.3"
    assert store.stats()["by_engine_version"] == {"1.2.3": 1}


def test_migration_adds_engine_version_to_existing_db(tmp_path) -> None:
    """A pre-provenance DB is migrated in place; old rows stay readable."""
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts REAL NOT NULL,
            session_id TEXT,
            tool TEXT,
            preset TEXT,
            model TEXT,
            status TEXT NOT NULL,
            error TEXT,
            elapsed_ms REAL,
            input_chars INTEGER NOT NULL,
            input_tokens INTEGER,
            answers TEXT,
            routing TEXT,
            input_text TEXT
        );
        INSERT INTO calls(run_id, ts, status, input_chars)
        VALUES ('legacy', 0.0, 'ok', 3);
        """
    )
    conn.commit()
    conn.close()

    store = UsageStore(db)
    _record(store, session="s-new")
    out = tmp_path / "m.jsonl"
    store.export_jsonl(out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2
    legacy = next(r for r in rows if r["run_id"] == "legacy")
    assert legacy["engine_version"] is None  # old row: provenance unknown
    fresh = next(r for r in rows if r["session_id"] == "s-new")
    assert fresh["engine_version"] == "1.2.3"  # new row: version captured


def test_migration_renames_legacy_laya_version_column(tmp_path) -> None:
    """A LayaMCP-era DB with `laya_version` keeps its data under the new name."""
    import sqlite3

    db = tmp_path / "laya-era.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts REAL NOT NULL,
            session_id TEXT,
            tool TEXT,
            preset TEXT,
            model TEXT,
            status TEXT NOT NULL,
            error TEXT,
            elapsed_ms REAL,
            input_chars INTEGER NOT NULL,
            input_tokens INTEGER,
            answers TEXT,
            routing TEXT,
            input_text TEXT,
            laya_version TEXT
        );
        INSERT INTO calls(run_id, ts, status, input_chars, laya_version)
        VALUES ('legacy-laya', 0.0, 'ok', 3, '0.3.21');
        """
    )
    conn.commit()
    conn.close()

    store = UsageStore(db)
    out = tmp_path / "l.jsonl"
    store.export_jsonl(out)
    row = json.loads(out.read_text())
    assert row["engine_version"] == "0.3.21"  # renamed, data preserved
    cols = {r["name"] for r in store._db.execute("PRAGMA table_info(calls)")}
    assert "engine_version" in cols and "laya_version" not in cols


def test_text_stored_when_enabled(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db", store_text=True)
    _record(store, text="my prompt")
    out = tmp_path / "out.jsonl"
    store.export_jsonl(out)
    assert json.loads(out.read_text())["input"] == "my prompt"
    assert store.stats()["rows_with_text"] == 1


def test_secret_risk_text_is_never_stored(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db", store_text=True)
    _record(store, text="AKIAxxxxxxxx", tool="decision_secret_risk")
    out = tmp_path / "out.jsonl"
    store.export_jsonl(out)
    assert json.loads(out.read_text())["input"] is None
    assert "AKIA" not in out.read_text()


def test_export_skips_errors_and_respects_since(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db")
    _record(store)
    store.record(text="x", preset="guard", result=None, elapsed_ms=1, error=RuntimeError("boom"), engine_version=None)
    out = tmp_path / "out.jsonl"
    assert store.export_jsonl(out) == 1  # error row excluded
    assert store.export_jsonl(out, since=9_999_999_999) == 0


def test_sessions_are_tracked(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db")
    store.open_session("s1", "test-client/1.0")
    _record(store, session="s1")
    _record(store, session="s1")
    _record(store, session="s2")
    store.close_session("s1")
    stats = store.stats()
    assert stats["calls"] == 3
    assert stats["sessions_with_calls"] == 2
    assert stats["sessions_total"] == 1  # only s1 was explicitly opened
    row = store._db.execute("SELECT * FROM sessions WHERE session_id='s1'").fetchone()
    assert row["client"] == "test-client/1.0"
    assert row["ended_at"] is not None


def test_store_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "sub" / "u.db"
    first = UsageStore(path)
    _record(first)
    first.close()
    assert UsageStore(path).stats()["calls"] == 1


@pytest.mark.asyncio
async def test_usage_tool_stats(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db")
    _record(store)
    bridge = MagicMock(usage=store)
    out = await UsageTool().run(bridge, action="stats")
    assert out.stats["calls"] == 1
    assert out.export_path is None


@pytest.mark.asyncio
async def test_usage_tool_export_writes_under_data_dir(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db")
    _record(store)
    bridge = MagicMock(usage=store)
    out = await UsageTool().run(bridge, action="export")
    assert out.exported_rows == 1
    assert out.export_path.startswith(str(tmp_path / "exports"))
    assert len(open(out.export_path).read().splitlines()) == 1


@pytest.mark.asyncio
async def test_usage_tool_export_labeled_only(tmp_path) -> None:
    store = UsageStore(tmp_path / "u.db", store_text=True)
    _record(store, text="pair me")
    bridge = MagicMock(usage=store)
    out = await UsageTool().run(bridge, action="export", labeled_only=True)
    assert out.exported_rows == 1
    row = json.loads(open(out.export_path).read())
    assert row["input"] == "pair me"


@pytest.mark.asyncio
async def test_usage_tool_errors_when_disabled() -> None:
    with pytest.raises(ToolError, match="disabled"):
        await UsageTool().run(MagicMock(usage=None), action="stats")
