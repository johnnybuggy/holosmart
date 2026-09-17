"""SQLite database access: schema bootstrap, per-thread connections, vector helpers.

The schema lives in ``schema.sql`` next to this module (single source of truth)
and is applied **idempotently** on every :meth:`Database.connect` call via
:meth:`sqlite3.Connection.executescript` (all statements use ``IF NOT EXISTS``).

Thread-safety: connections are short-lived and never shared across threads —
each thread (UI, scanner worker, analysis workers) calls :meth:`Database.connect`
(or enters :meth:`Database.transaction`) to get its own connection.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Database", "vec_to_blob", "blob_to_vec", "SCHEMA_PATH", "SCHEMA_SQL"]

#: Path of the SQL schema shipped next to this module.
SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"

#: Schema SQL, read once at import (the file is part of the package).
SCHEMA_SQL: str = SCHEMA_PATH.read_text(encoding="utf-8")

#: Columns added after the initial schema: ``(table, column, ddl)``.  Fresh
#: databases get them from ``SCHEMA_SQL``; existing ones via ``ALTER TABLE``
#: (idempotent — the PRAGMA check skips columns that already exist).
_MIGRATION_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("tracks", "source_mtime", "REAL"),
    ("tracks", "source_size", "INTEGER"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns shipped after a database was first created."""
    for table, column, ddl in _MIGRATION_COLUMNS:
        existing = {row["name"] for row in conn.execute(
            f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            log.info("Migrated %s: added column %s", table, column)


class Database:
    """Factory for short-lived SQLite connections to one database file."""

    def __init__(self, db_path: Path | str) -> None:
        """Remember the target database file; parent dirs are created on connect."""
        self.db_path: Path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        """Open a **fresh** connection and return it fully initialised.

        The connection uses ``sqlite3.Row`` rows, ``PRAGMA foreign_keys=ON``
        (required for the schema's ON DELETE CASCADE clean-ups), WAL journal
        mode and ``synchronous=NORMAL``.  Missing parent directories are
        created and the schema is applied idempotently, so the first connect
        on a fresh file yields a ready-to-use database.  The caller owns the
        connection and must close it (or write via :meth:`transaction`).
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - e.g. exotic in-memory paths
            log.debug("Could not ensure parent directory of %s", self.db_path, exc_info=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA_SQL)
        _migrate(conn)
        conn.commit()
        log.debug("Opened SQLite connection to %s", self.db_path)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a fresh connection wrapped in one transaction.

        Commits when the block exits normally, rolls back and re-raises on any
        exception.  The connection is closed when the block ends, so it must
        not be used outside the ``with`` body.
        """
        conn = self.connect()
        try:
            yield conn
        except BaseException:
            conn.rollback()
            log.debug("Transaction on %s rolled back", self.db_path, exc_info=True)
            raise
        else:
            conn.commit()
        finally:
            conn.close()


def vec_to_blob(vec: np.ndarray) -> bytes:
    """Encode a vector as little-endian float32 bytes for the BLOB columns."""
    return np.asarray(vec, dtype="<f4").reshape(-1).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    """Decode float32 BLOB bytes back into a writable 1-D numpy vector."""
    return np.frombuffer(blob, dtype="<f4").copy()
