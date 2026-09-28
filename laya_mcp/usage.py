"""Session + usage log (SQLite) for collecting training data.

Laya ships hooks (``run_id``, token usage, timing per call) but persists
nothing, so LayaMCP records every prediction itself. One row per call in
``calls``; one row per MCP connection in ``sessions``.

Privacy: input text is stored only when ``store_text`` is on, and never for
tools in :data:`NEVER_STORE_TEXT` (their inputs are secrets by definition).
Without text, rows still carry the model's answers, routing and latency.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Set by the server around each request so the bridge can attribute a
# prediction to the MCP tool and connection that caused it.
current_tool: ContextVar[str | None] = ContextVar("layamcp_current_tool", default=None)
current_session: ContextVar[str | None] = ContextVar("layamcp_current_session", default=None)

NEVER_STORE_TEXT = frozenset({"laya_secret_risk"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at   REAL,
    client     TEXT
);
CREATE TABLE IF NOT EXISTS calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    ts           REAL NOT NULL,
    session_id   TEXT,
    tool         TEXT,
    preset       TEXT,
    model        TEXT,
    status       TEXT NOT NULL,
    error        TEXT,
    elapsed_ms   REAL,
    input_chars  INTEGER NOT NULL,
    input_tokens INTEGER,
    answers      TEXT,
    routing      TEXT,
    input_text   TEXT
);
CREATE INDEX IF NOT EXISTS calls_ts ON calls(ts);
CREATE INDEX IF NOT EXISTS calls_tool ON calls(tool);
CREATE INDEX IF NOT EXISTS calls_session ON calls(session_id);
"""


class UsageStore:
    """Thread-safe SQLite store. Cheap enough to call inline per prediction."""

    def __init__(self, path: Path, *, store_text: bool = False) -> None:
        self.path = Path(path)
        self.store_text = store_text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def open_session(self, session_id: str, client: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO sessions(session_id, started_at, client) VALUES (?, ?, ?)",
                (session_id, time.time(), client),
            )
            self._db.commit()

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            self._db.commit()

    def record(
        self,
        *,
        text: str,
        preset: str | None,
        result: dict | None,
        elapsed_ms: float,
        error: BaseException | None = None,
    ) -> str:
        """Log one prediction. Returns the generated ``run_id``.

        ``tool`` and ``session_id`` come from the request context vars.
        """
        tool = current_tool.get()
        run_id = uuid.uuid4().hex
        result = result or {}
        usage = result.get("usage") or {}
        routing = result.get("routing") or {}
        keep_text = self.store_text and tool not in NEVER_STORE_TEXT
        with self._lock:
            self._db.execute(
                "INSERT INTO calls(run_id, ts, session_id, tool, preset, model, status, error,"
                " elapsed_ms, input_chars, input_tokens, answers, routing, input_text)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    time.time(),
                    current_session.get(),
                    tool,
                    preset,
                    routing.get("model"),
                    "error" if error else "ok",
                    type(error).__name__ if error else None,
                    round(elapsed_ms, 2),
                    len(text),
                    usage.get("input_tokens"),
                    json.dumps(result.get("answers")) if result.get("answers") else None,
                    json.dumps(routing) if routing else None,
                    text if keep_text else None,
                ),
            )
            self._db.commit()
        return run_id

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def stats(self, since: float | None = None) -> dict[str, Any]:
        where, args = ("WHERE ts >= ?", (since,)) if since else ("", ())
        with self._lock:
            total = self._db.execute(
                f"SELECT COUNT(*) n, SUM(status='error') errs, AVG(elapsed_ms) avg_ms,"
                f" SUM(input_tokens) toks, COUNT(DISTINCT session_id) sess,"
                f" SUM(input_text IS NOT NULL) with_text FROM calls {where}",
                args,
            ).fetchone()
            by_tool = self._db.execute(
                f"SELECT COALESCE(tool, preset, '?') t, COUNT(*) n FROM calls {where}"
                f" GROUP BY t ORDER BY n DESC",
                args,
            ).fetchall()
            by_model = self._db.execute(
                f"SELECT COALESCE(model, '?') m, COUNT(*) n FROM calls {where} GROUP BY m", args
            ).fetchall()
            sessions = self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {
            "calls": total["n"],
            "errors": total["errs"] or 0,
            "avg_latency_ms": round(total["avg_ms"], 2) if total["avg_ms"] is not None else None,
            "input_tokens": total["toks"] or 0,
            "sessions_with_calls": total["sess"],
            "sessions_total": sessions,
            "rows_with_text": total["with_text"] or 0,
            "by_tool": {r["t"]: r["n"] for r in by_tool},
            "by_model": {r["m"]: r["n"] for r in by_model},
            "store_text": self.store_text,
            "db_path": str(self.path),
        }

    def export_jsonl(self, out_path: Path, since: float | None = None) -> int:
        """Write successful calls as JSONL and return the row count.

        Each line: ``{run_id, ts, session_id, tool, preset, model, input, answers}``.
        ``input`` is ``null`` unless text storage was on when the call was made.
        """
        where, args = ("AND ts >= ?", (since,)) if since else ("", ())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM calls WHERE status = 'ok' {where} ORDER BY id", args
            ).fetchall()
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {
                            "run_id": r["run_id"],
                            "ts": r["ts"],
                            "session_id": r["session_id"],
                            "tool": r["tool"],
                            "preset": r["preset"],
                            "model": r["model"],
                            "input": r["input_text"],
                            "answers": json.loads(r["answers"]) if r["answers"] else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._db.close()
