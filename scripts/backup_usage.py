"""Nightly backup of the usage corpus.

Uses SQLite's online backup API (WAL-safe, no locks held on the live DB),
writes a self-contained snapshot to the backup dir, prunes old snapshots
beyond ``--keep``. Run manually or via the launchd agent
(``com.gulaneskorp.layamcp-backup``).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = Path("data/usage.db")
DEFAULT_OUT = Path("data/backups")
DEFAULT_KEEP = 14


def backup(db_path: Path, out_dir: Path, *, keep: int = DEFAULT_KEEP) -> Path:
    """Snapshot ``db_path`` into ``out_dir``; return the snapshot path.

    Raises ``sqlite3.Error`` on failure — caller decides how to surface it.
    Retention: keeps the ``keep`` newest ``usage-*.db`` snapshots.
    """
    if not db_path.exists():
        print(f"no usage DB at {db_path}; nothing to back up")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = out_dir / f"usage-{stamp}.db"
    n = 1
    while dest.exists():  # same-second reruns must not overwrite snapshots
        n += 1
        dest = out_dir / f"usage-{stamp}-{n}.db"

    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(dest)
        try:
            with dst:
                src.backup(dst)  # online: safe while the server is writing
        finally:
            dst.close()
    finally:
        src.close()

    if keep > 0:
        snapshots = sorted(
            out_dir.glob("usage-*.db"), key=lambda p: p.stat().st_mtime_ns
        )  # ns mtime: strictly ordered even for rapid successive backups
        for old in snapshots[:-keep]:
            old.unlink(missing_ok=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="usage DB path")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="backup dir")
    p.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="snapshots to keep (0 = all)")
    args = p.parse_args(argv)
    try:
        dest = backup(args.db, args.out, keep=args.keep)
    except sqlite3.Error as e:
        print(f"backup failed: {e}", file=sys.stderr)
        return 1
    if dest is None:
        return 0
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
