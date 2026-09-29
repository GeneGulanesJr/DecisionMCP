"""Tests for the usage-corpus backup script."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backup_usage import backup, main  # noqa: E402


def _seed(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE calls (id INTEGER PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO calls(note) VALUES ('hello')")
    conn.commit()
    conn.close()


def test_backup_snapshots_contents(tmp_path) -> None:
    db = tmp_path / "usage.db"
    _seed(db)
    dest = backup(db, tmp_path / "backups")
    assert dest.exists()
    conn = sqlite3.connect(dest)
    assert conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 1
    conn.close()


def test_backup_retention_prunes_oldest(tmp_path) -> None:
    db = tmp_path / "usage.db"
    _seed(db)
    out = tmp_path / "backups"
    keep = 2
    dests = [backup(db, out, keep=keep) for _ in range(4)]
    remaining = sorted(out.glob("usage-*.db"))
    # Exactly `keep` newest snapshots survive (names may be recycled after
    # pruning, so assert on recency by mtime, not on path history).
    assert len(remaining) == keep
    newest = max(remaining, key=lambda p: p.stat().st_mtime_ns)
    assert newest == dests[-1]
    oldest_kept = min(remaining, key=lambda p: p.stat().st_mtime_ns)
    assert oldest_kept.stat().st_mtime_ns <= newest.stat().st_mtime_ns


def test_main_missing_db_fails(tmp_path, capsys) -> None:
    rc = main(["--db", str(tmp_path / "nope.db"), "--out", str(tmp_path / "b")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
