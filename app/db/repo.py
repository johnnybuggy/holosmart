"""Data-access functions for the HoloSmart library database.

Every function takes an open ``sqlite3.Connection`` as its first argument
(obtained from :class:`app.db.database.Database`, which enables foreign keys
and WAL) and returns ``sqlite3.Row`` objects.  The embedding getter
:func:`get_chunk_embeddings` additionally provides a decoded ``'vec'`` key
(a :class:`numpy.ndarray`) on each row.

The functions only execute SQL on the given connection; transaction
boundaries are the caller's responsibility (see
:meth:`app.db.database.Database.transaction`).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

import numpy as np

from .database import blob_to_vec, vec_to_blob

log = logging.getLogger(__name__)

__all__ = [
    "add_folder", "remove_folder", "list_folders", "get_folder_by_path",
    "upsert_track", "list_tracks", "get_track", "get_track_by_path",
    "set_track_status", "set_track_description", "delete_tracks_missing",
    "clear_track_analysis",
    "replace_chunks", "get_chunks", "get_chunks_for_tracks",
    "add_chunk_embedding", "get_chunk_embeddings", "set_track_embedding",
    "get_chunk_embedding_rows",
    "get_track_embedding", "get_track_embeddings", "tracks_with_embeddings",
    "get_track_chunk_models", "get_track_embedding_counts",
    "get_all_track_embedding_counts",
    "add_chunk_tags", "get_chunk_tags", "get_track_tags",
    "create_playlist", "add_playlist_items", "list_playlists", "get_playlist",
    "get_playlist_items", "delete_playlist",
    "set_setting", "get_setting", "save_ollama_models", "get_ollama_models",
    "create_reduction", "set_reduction_result", "list_reductions",
    "get_reduction", "delete_reduction", "replace_reduced_embeddings",
    "get_reduced_chunk_embeddings", "tracks_with_reduced_chunks",
    "get_chunk_vector_models",
    "chunk_vector_counts",
    "set_noise_filter_result",
    "update_noise_filter_result",
    "noise_track_signatures",
    "record_noise_run_tracks",
    "pending_noise_tracks",
    "get_noise_filter",
    "list_noise_filters",
    "get_noise_chunk_ids",
    "delete_noise_filter",
]

#: Track columns that may be supplied through the ``meta`` dict of
#: :func:`upsert_track` (all optional).
_TRACK_META_COLUMNS: tuple[str, ...] = (
    "filename", "extension", "size_bytes", "mtime", "container", "codec",
    "sample_rate", "channels", "bit_depth", "bitrate_kbps", "duration_sec",
    "title", "artist", "album", "genre", "year", "track_no",
    "source_mtime", "source_size",
)

#: Settings key under which :func:`save_ollama_models` stores its JSON list.
_OLLAMA_MODELS_KEY = "ollama_embedding_models"


def _norm_path(path: str) -> str:
    """Canonicalise a path the same way everywhere it is stored or looked up."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


class _EmbeddingRow:
    """Row-like view of an ``embeddings`` row plus a decoded ``'vec'`` key.

    :class:`sqlite3.Row` instances are immutable, so the decoded numpy vector
    cannot be attached to one directly.  This wrapper mirrors the underlying
    row's behaviour — lookup by column name or position, ``keys()``,
    ``len()``, iteration over values, ``dict(row)`` — and adds ``row["vec"]``.
    """

    __slots__ = ("_row", "_vec")

    def __init__(self, row: sqlite3.Row, vec: np.ndarray) -> None:
        self._row = row
        self._vec = vec

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str) and key == "vec":
            return self._vec
        return self._row[key]

    def keys(self) -> list[str]:
        keys = list(self._row.keys())
        if "vec" not in keys:
            keys.append("vec")
        return keys

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (IndexError, KeyError):
            return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key == "vec" or key in self._row.keys())

    def __len__(self) -> int:
        return len(self._row) + 1

    def __iter__(self):  # sqlite3.Row iterates over values; mirror that
        for name in self.keys():
            yield self[name]

    def __repr__(self) -> str:
        return f"<_EmbeddingRow model={self._row['model']!r} dim={self._row['dim']} vec={self._vec[:4]!r}...>"


# ---------------------------------------------------------------- folders ----
def add_folder(conn: sqlite3.Connection, path: str) -> int:
    """Insert *path* into ``folders`` unless already present; return its id."""
    norm = _norm_path(path)
    conn.execute("INSERT OR IGNORE INTO folders(path) VALUES (?)", (norm,))
    row = conn.execute("SELECT id FROM folders WHERE path = ?", (norm,)).fetchone()
    if row is None:  # pragma: no cover - INSERT OR IGNORE cannot fail here
        raise RuntimeError(f"add_folder: could not persist folder {norm!r}")
    return int(row["id"])


def remove_folder(conn: sqlite3.Connection, folder_id: int) -> None:
    """Delete a folder; its tracks (and via cascade their chunks, embeddings
    and tags) are removed as well."""
    # Delete tracks explicitly first so their cascades fire even on a
    # connection created without ``PRAGMA foreign_keys = ON``.
    conn.execute("DELETE FROM tracks WHERE folder_id = ?", (folder_id,))
    conn.execute("DELETE FROM folders WHERE id = ?", (folder_id,))


def list_folders(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all tracked folders ordered by path."""
    return conn.execute("SELECT * FROM folders ORDER BY path").fetchall()


def get_folder_by_path(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    """Return the folder row for *path* or ``None`` when not tracked."""
    return conn.execute(
        "SELECT * FROM folders WHERE path = ?", (_norm_path(path),)
    ).fetchone()


# ----------------------------------------------------------------- tracks ----
def upsert_track(conn: sqlite3.Connection, folder_id: int, path: str,
                 meta: dict) -> int:
    """Insert or update a track identified by its unique *path*; return its id.

    ``meta`` may contain any subset of the optional track columns
    (filename, extension, size_bytes, mtime, container, codec, sample_rate,
    channels, bit_depth, bitrate_kbps, duration_sec, title, artist, album,
    genre, year, track_no).  Unknown keys are ignored with a debug log;
    keys that are absent (or ``None``) leave previously stored values
    untouched on update.  ``filename`` falls back to the path's basename.
    ``status``/``status_message``/``description`` are never touched here.
    """
    norm = _norm_path(path)
    meta = meta or {}
    unknown = sorted(set(meta) - set(_TRACK_META_COLUMNS))
    if unknown:
        log.debug("upsert_track(%s): ignoring unknown meta keys %s", norm, unknown)

    entries: list[tuple[str, Any]] = [
        ("folder_id", int(folder_id)),
        ("path", norm),
        ("filename", str(meta["filename"]) if meta.get("filename")
         else os.path.basename(norm)),
    ]
    for col in _TRACK_META_COLUMNS:
        if col == "filename" or col not in meta or meta[col] is None:
            continue
        entries.append((col, meta[col]))

    names = ", ".join(name for name, _ in entries)
    placeholders = ", ".join("?" for _ in entries)
    updates = ", ".join(f"{name} = excluded.{name}" for name, _ in entries
                        if name != "path")
    conn.execute(
        f"INSERT INTO tracks ({names}) VALUES ({placeholders}) "
        f"ON CONFLICT(path) DO UPDATE SET {updates}",
        tuple(value for _, value in entries),
    )
    row = conn.execute("SELECT id FROM tracks WHERE path = ?", (norm,)).fetchone()
    if row is None:  # pragma: no cover - defensive
        raise RuntimeError(f"upsert_track: track {norm!r} was not persisted")
    return int(row["id"])


def list_tracks(conn: sqlite3.Connection,
                folder_id: int | None = None) -> list[sqlite3.Row]:
    """Return tracks (optionally only those of one folder) ordered by path."""
    if folder_id is None:
        return conn.execute("SELECT * FROM tracks ORDER BY path").fetchall()
    return conn.execute(
        "SELECT * FROM tracks WHERE folder_id = ? ORDER BY path", (folder_id,)
    ).fetchall()


def get_track(conn: sqlite3.Connection, track_id: int) -> sqlite3.Row | None:
    """Return one track row by id, or ``None``."""
    return conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()


def get_track_by_path(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    """Return one track row by path, or ``None``."""
    return conn.execute(
        "SELECT * FROM tracks WHERE path = ?", (_norm_path(path),)
    ).fetchone()


def set_track_status(conn: sqlite3.Connection, track_id: int, status: str,
                     message: str | None = None) -> None:
    """Store the analysis *status* (new|analyzing|analyzed|error) and message.

    ``message=None`` clears any previous message.  Reaching ``analyzed``
    also stamps ``last_analyzed_at``.
    """
    if status == "analyzed":
        conn.execute(
            "UPDATE tracks SET status = ?, status_message = ?, "
            "last_analyzed_at = datetime('now') WHERE id = ?",
            (status, message, track_id),
        )
    else:
        conn.execute(
            "UPDATE tracks SET status = ?, status_message = ? WHERE id = ?",
            (status, message, track_id),
        )


def set_track_description(conn: sqlite3.Connection, track_id: int,
                          description: str) -> None:
    """Store the aggregated human-readable description of a track."""
    conn.execute("UPDATE tracks SET description = ? WHERE id = ?",
                 (description, track_id))


def delete_tracks_missing(conn: sqlite3.Connection, folder_id: int,
                          existing_paths: set[str]) -> int:
    """Delete the folder's tracks whose paths are not in *existing_paths*.

    Returns the number of tracks removed; their chunks/embeddings/tags go
    with them through the schema cascades.  An empty *existing_paths* removes
    every track of the folder.
    """
    keep = {_norm_path(p) for p in existing_paths or set()}
    rows = conn.execute(
        "SELECT id, path FROM tracks WHERE folder_id = ?", (folder_id,)
    ).fetchall()
    doomed = [int(r["id"]) for r in rows if r["path"] not in keep]
    if not doomed:
        return 0
    placeholders = ", ".join("?" for _ in doomed)
    cur = conn.execute(
        f"DELETE FROM tracks WHERE id IN ({placeholders})", tuple(doomed)
    )
    return int(cur.rowcount) if cur.rowcount >= 0 else len(doomed)


# ----------------------------------------------------------------- chunks ----
def replace_chunks(conn: sqlite3.Connection, track_id: int,
                   items: list[tuple[int, float, float]]) -> list[int]:
    """Replace all chunks of a track with *items* of ``(idx, start_sec, end_sec)``.

    Returns the new chunk ids in *items* order.  Previously stored chunks of
    the track (and their embeddings and tags, via cascade) are removed first.
    """
    conn.execute("DELETE FROM chunks WHERE track_id = ?", (track_id,))
    ids: list[int] = []
    for idx, start_sec, end_sec in items:
        cur = conn.execute(
            "INSERT INTO chunks(track_id, idx, start_sec, end_sec) VALUES (?, ?, ?, ?)",
            (track_id, int(idx), float(start_sec), float(end_sec)),
        )
        ids.append(int(cur.lastrowid))
    return ids


def track_model_coverage(conn: sqlite3.Connection, track_id: int,
                         ) -> tuple[int, dict[str, int]]:
    """``(n_chunks, {model: distinct chunks carrying that model's vectors})``.

    The completeness check for the analysis skip policy: a track is only
    "already analyzed" when EVERY enabled model has a vector on EVERY
    chunk — a track analyzed with FFT only is *not* complete once MERT-330M
    is enabled and must be visited again (incrementally).
    """
    n_chunks = int(conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE track_id = ?",
        (int(track_id),)).fetchone()[0])
    coverage = {str(r["model"]): int(r["n"]) for r in conn.execute(
        "SELECT e.model AS model, COUNT(DISTINCT e.chunk_id) AS n "
        "FROM embeddings e JOIN chunks c ON c.id = e.chunk_id "
        "WHERE c.track_id = ? GROUP BY e.model", (int(track_id),))}
    return n_chunks, coverage


def get_chunks(conn: sqlite3.Connection, track_id: int) -> list[sqlite3.Row]:
    """Return the chunks of one track ordered by ``idx``."""
    return conn.execute(
        "SELECT * FROM chunks WHERE track_id = ? ORDER BY idx", (track_id,)
    ).fetchall()


def get_chunks_for_tracks(conn: sqlite3.Connection,
                          track_ids: list[int]) -> list[sqlite3.Row]:
    """Return the chunks of several tracks ordered by track, then ``idx``."""
    if not track_ids:
        return []
    placeholders = ", ".join("?" for _ in track_ids)
    return conn.execute(
        f"SELECT * FROM chunks WHERE track_id IN ({placeholders}) "
        "ORDER BY track_id, idx",
        tuple(track_ids),
    ).fetchall()


# ------------------------------------------------------------- embeddings ----
def add_chunk_embedding(conn: sqlite3.Connection, chunk_id: int, model: str,
                        vec: np.ndarray) -> None:
    """Store (or replace) the *model* embedding vector of one chunk.

    The vector is stored as little-endian float32 bytes together with its
    dimension and L2 norm.
    """
    v = np.asarray(vec, dtype="<f4").reshape(-1)
    norm = float(np.linalg.norm(v)) if v.size else 0.0
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector, norm) "
        "VALUES (?, ?, ?, ?, ?)",
        (chunk_id, model, int(v.size), vec_to_blob(v), norm),
    )


def get_chunk_embeddings(conn: sqlite3.Connection, chunk_id: int,
                         model: str | None = None) -> list[sqlite3.Row]:
    """Return embedding rows of one chunk (optionally only *model*'s).

    Each row behaves like a ``sqlite3.Row`` of the ``embeddings`` table and
    additionally carries ``row["vec"]`` — the decoded float32 numpy vector.
    """
    sql = "SELECT * FROM embeddings WHERE chunk_id = ?"
    args: tuple = (chunk_id,)
    if model is not None:
        sql += " AND model = ?"
        args = (chunk_id, model)
    rows = conn.execute(sql + " ORDER BY model, id", args).fetchall()
    return [_EmbeddingRow(row, blob_to_vec(row["vector"])) for row in rows]


def set_track_embedding(conn: sqlite3.Connection, track_id: int, model: str,
                        vec: np.ndarray) -> None:
    """Upsert a per-track aggregate vector (centroid or Ollama text embedding)."""
    v = np.asarray(vec, dtype="<f4").reshape(-1)
    norm = float(np.linalg.norm(v)) if v.size else 0.0
    conn.execute(
        "INSERT INTO track_embeddings(track_id, model, dim, vector, norm) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(track_id, model) DO UPDATE SET dim = excluded.dim, "
        "vector = excluded.vector, norm = excluded.norm, "
        "created_at = datetime('now')",
        (track_id, model, int(v.size), vec_to_blob(v), norm),
    )


def get_track_embedding(conn: sqlite3.Connection, track_id: int,
                        model: str) -> np.ndarray | None:
    """Return the stored vector for *(track, model)* or ``None``."""
    row = conn.execute(
        "SELECT vector FROM track_embeddings WHERE track_id = ? AND model = ?",
        (track_id, model),
    ).fetchone()
    return blob_to_vec(row["vector"]) if row else None


def get_chunk_embedding_rows(conn: sqlite3.Connection,
                             models: list[str] | None = None
                             ) -> list[sqlite3.Row]:
    """All chunk embeddings joined with chunk/track info, for visualisation.

    One row per ``(chunk, model)`` pair, ordered by track, chunk index and
    model.  Each row behaves like the ``embeddings`` row plus the decoded
    ``row["vec"]`` vector, ``row["chunk_idx"]``, ``row["start_sec"]``,
    ``row["track_filename"]`` and ``row["track_path"]`` — everything the
    Visualisation dialog needs to place and identify a point.  Pass
    *models* to restrict to those plugin names (no filter = all models).
    """
    sql = (
        "SELECT e.*, c.track_id AS track_id, c.idx AS chunk_idx, "
        "c.start_sec AS start_sec, "
        "t.filename AS track_filename, t.path AS track_path "
        "FROM embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "JOIN tracks t ON t.id = c.track_id"
    )
    args: tuple = ()
    if models:
        placeholders = ", ".join("?" for _ in models)
        sql += f" WHERE e.model IN ({placeholders})"
        args = tuple(models)
    rows = conn.execute(sql + " ORDER BY t.id, c.idx, e.model", args).fetchall()
    return [_EmbeddingRow(row, blob_to_vec(row["vector"])) for row in rows]


def get_track_embeddings(conn: sqlite3.Connection, model: str,
                         track_ids: list[int] | None = None) -> dict[int, np.ndarray]:
    """Return ``{track_id: vector}`` for *model*, optionally restricted to
    *track_ids* (an empty list yields an empty dict)."""
    if track_ids is not None and not track_ids:
        return {}
    sql = "SELECT track_id, vector FROM track_embeddings WHERE model = ?"
    args: tuple = (model,)
    if track_ids is not None:
        placeholders = ", ".join("?" for _ in track_ids)
        sql += f" AND track_id IN ({placeholders})"
        args = (model, *[int(t) for t in track_ids])
    return {int(r["track_id"]): blob_to_vec(r["vector"])
            for r in conn.execute(sql, args).fetchall()}


def tracks_with_embeddings(conn: sqlite3.Connection, model: str) -> list[int]:
    """Return the ids of tracks that have a *model* embedding, ascending."""
    rows = conn.execute(
        "SELECT DISTINCT track_id FROM track_embeddings WHERE model = ? "
        "ORDER BY track_id",
        (model,),
    ).fetchall()
    return [int(r["track_id"]) for r in rows]


# ------------------------------------------------------------------- tags ----
def add_chunk_tags(conn: sqlite3.Connection, chunk_id: int, model: str,
                   tags: list[tuple[str, float]]) -> None:
    """Store ``(text, score)`` tags of one model for one chunk (idempotent
    per ``(chunk_id, model, text)`` — re-adding updates the score)."""
    conn.executemany(
        "INSERT OR REPLACE INTO chunk_tags(chunk_id, model, text, score) "
        "VALUES (?, ?, ?, ?)",
        [(chunk_id, model, str(text), None if score is None else float(score))
         for text, score in tags],
    )


def get_chunk_tags(conn: sqlite3.Connection, chunk_id: int,
                   model: str | None = None) -> list[sqlite3.Row]:
    """Return a chunk's tags, best score first (optionally only *model*'s)."""
    sql = "SELECT * FROM chunk_tags WHERE chunk_id = ?"
    args: tuple = (chunk_id,)
    if model is not None:
        sql += " AND model = ?"
        args = (chunk_id, model)
    return conn.execute(sql + " ORDER BY score DESC, text ASC", args).fetchall()


def get_track_tags(conn: sqlite3.Connection, track_id: int,
                   model: str | None = None) -> list[sqlite3.Row]:
    """Return all tags of a track's chunks joined with ``chunks.idx`` as
    ``chunk_idx``, best score first (optionally only *model*'s)."""
    sql = (
        "SELECT ct.*, c.idx AS chunk_idx FROM chunk_tags ct "
        "JOIN chunks c ON c.id = ct.chunk_id "
        "WHERE c.track_id = ?"
    )
    args: tuple = (track_id,)
    if model is not None:
        sql += " AND ct.model = ?"
        args = (track_id, model)
    return conn.execute(sql + " ORDER BY ct.score DESC, ct.text ASC", args).fetchall()


# -------------------------------------------------------------- playlists ----
def create_playlist(conn: sqlite3.Connection, name: str,
                    seed_track_id: int | None = None,
                    method: str | None = None) -> int:
    """Create a playlist and return its id."""
    cur = conn.execute(
        "INSERT INTO playlists(name, seed_track_id, method) VALUES (?, ?, ?)",
        (name, seed_track_id, method),
    )
    return int(cur.lastrowid)


def add_playlist_items(conn: sqlite3.Connection, playlist_id: int,
                       items: list[tuple[int, float]]) -> None:
    """Set the playlist's items to *items* of ``(track_id, similarity)``.

    Items get positions 1..n in the given order; previously stored items of
    the playlist are replaced, so calling twice never leaves stale rows.
    """
    conn.execute("DELETE FROM playlist_items WHERE playlist_id = ?", (playlist_id,))
    conn.executemany(
        "INSERT INTO playlist_items(playlist_id, track_id, position, similarity) "
        "VALUES (?, ?, ?, ?)",
        [(playlist_id, int(track_id), position, None if similarity is None
          else float(similarity))
         for position, (track_id, similarity) in enumerate(items, start=1)],
    )


def list_playlists(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all playlists, newest first."""
    return conn.execute(
        "SELECT * FROM playlists ORDER BY created_at DESC, id DESC"
    ).fetchall()


def get_playlist(conn: sqlite3.Connection, playlist_id: int) -> sqlite3.Row | None:
    """Return one playlist row by id, or ``None``."""
    return conn.execute(
        "SELECT * FROM playlists WHERE id = ?", (playlist_id,)
    ).fetchone()


def get_playlist_items(conn: sqlite3.Connection,
                       playlist_id: int) -> list[sqlite3.Row]:
    """Return a playlist's items joined with their track columns, in order."""
    return conn.execute(
        "SELECT pi.id, pi.playlist_id, pi.position, pi.similarity, pi.track_id, "
        "t.path, t.filename, t.title, t.artist, t.album, t.duration_sec, t.status "
        "FROM playlist_items pi JOIN tracks t ON t.id = pi.track_id "
        "WHERE pi.playlist_id = ? ORDER BY pi.position",
        (playlist_id,),
    ).fetchall()


def delete_playlist(conn: sqlite3.Connection, playlist_id: int) -> None:
    """Delete a playlist and (via cascade) its items."""
    conn.execute("DELETE FROM playlist_items WHERE playlist_id = ?", (playlist_id,))
    conn.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))


# --------------------------------------------------------------- settings ----
def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update one settings key (TEXT values; JSON-encode lists)."""
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, None if value is None else str(value)),
    )


def get_setting(conn: sqlite3.Connection, key: str,
                default: str | None = None) -> str | None:
    """Return the stored value for *key*, or *default* when unset/NULL."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    value = row["value"] if row else None
    return default if value is None else value


def save_ollama_models(conn: sqlite3.Connection, models: list[str]) -> None:
    """Persist the detected Ollama embedding-model names as a JSON list."""
    set_setting(conn, _OLLAMA_MODELS_KEY, json.dumps([str(m) for m in models]))


def get_ollama_models(conn: sqlite3.Connection) -> list[str]:
    """Return the stored Ollama embedding-model names (``[]`` when unset/broken)."""
    raw = get_setting(conn, _OLLAMA_MODELS_KEY)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        log.warning("Corrupted %s setting: %r", _OLLAMA_MODELS_KEY, raw)
        return []
    if not isinstance(data, list):
        return []
    return [str(m) for m in data]


# ------------------------------------------------- chunk-model discovery ----
# Appended for the Pareto chunk-level similarity search
# (app.similarity.pareto): it needs to know, without touching the
# per-track ``track_embeddings`` aggregates, which embedding models exist on
# each track's chunks.
def get_track_chunk_models(conn: sqlite3.Connection) -> dict[int, list[str]]:
    """Map each track id to the sorted embedding models found on its chunks.

    Joins ``chunks`` with ``embeddings``, so tracks without any chunk
    embedding (or without chunks at all) are omitted.  Used by the Pareto
    search to find candidate tracks sharing at least one model with the
    seed's chunks.
    """
    rows = conn.execute(
        "SELECT DISTINCT c.track_id AS track_id, e.model AS model "
        "FROM chunks c JOIN embeddings e ON e.chunk_id = c.id "
        "ORDER BY c.track_id, e.model"
    ).fetchall()
    result: dict[int, list[str]] = {}
    for row in rows:
        result.setdefault(int(row["track_id"]), []).append(str(row["model"]))
    return result


def get_track_embedding_counts(conn: sqlite3.Connection,
                               track_id: int) -> dict[str, int]:
    """Map embedding model -> number of chunk embeddings of one track.

    An empty dict means the track has no chunk embeddings at all.  Used by
    the file tree's per-model status columns (``update_track_status``).
    """
    rows = conn.execute(
        "SELECT e.model AS model, COUNT(*) AS n "
        "FROM chunks c JOIN embeddings e ON e.chunk_id = c.id "
        "WHERE c.track_id = ? GROUP BY e.model",
        (int(track_id),),
    ).fetchall()
    return {str(row["model"]): int(row["n"]) for row in rows}


def get_all_track_embedding_counts(
    conn: sqlite3.Connection,
) -> dict[int, dict[str, int]]:
    """Map each track id to ``{model: chunk-embedding count}`` in one query.

    Tracks without chunk embeddings are omitted (their models are simply
    absent).  Used when the file tree is (re)built, so per-model status
    columns for a whole library cost a single grouped query instead of one
    query per track.
    """
    rows = conn.execute(
        "SELECT c.track_id AS track_id, e.model AS model, COUNT(*) AS n "
        "FROM chunks c JOIN embeddings e ON e.chunk_id = c.id "
        "GROUP BY c.track_id, e.model"
    ).fetchall()
    result: dict[int, dict[str, int]] = {}
    for row in rows:
        result.setdefault(int(row["track_id"]), {})[str(row["model"])] = int(row["n"])
    return result


# ------------------------------------------------ analysis result cleanup ----
# Appended for the "Clear Analysis" action (app.ui.main_window): removes
# every analysis artefact of one track so it becomes re-analyzable via the
# normal Analyze paths, without deleting the track itself.
def clear_track_analysis(conn: sqlite3.Connection, track_id: int) -> None:
    """Delete all analysis results of the track *track_id* and reset it.

    Removes the track's chunks together with their embeddings and tags, its
    ``track_embeddings`` aggregates (centroids **and** ``ollama:<model>``
    text vectors) and clears ``status``, ``status_message`` and
    ``description`` — the track itself is kept.

    The chunk-child rows are deleted through explicit sub-selects *before*
    the ``chunks`` rows, mirroring :func:`remove_folder`, so the cleanup also
    works on a connection created without ``PRAGMA foreign_keys = ON``.

    ``last_analyzed_at`` is intentionally kept (historical record).  The
    cleared track is re-analyzed by the normal Analyze paths: batch runs
    skip only tracks whose status is ``analyzed``, so it is picked up again.
    """
    conn.execute(
        "DELETE FROM embeddings WHERE chunk_id IN "
        "(SELECT id FROM chunks WHERE track_id = ?)", (track_id,))
    conn.execute(
        "DELETE FROM chunk_tags WHERE chunk_id IN "
        "(SELECT id FROM chunks WHERE track_id = ?)", (track_id,))
    conn.execute("DELETE FROM chunks WHERE track_id = ?", (track_id,))
    conn.execute("DELETE FROM track_embeddings WHERE track_id = ?", (track_id,))
    conn.execute(
        "UPDATE tracks SET status = 'new', status_message = NULL, "
        "description = NULL WHERE id = ?", (track_id,))


# ------------------------------------------------------------- reductions ---

def create_reduction(conn: sqlite3.Connection, name: str, source_model: str,
                     method: str, params: dict | None,
                     n_components: int) -> int:
    """Register a dimensionality-reduction dataset; returns its id.

    *name* must be unique (it is the user-facing dataset label); *params*
    is stored as JSON so a reduction can be reproduced/documented.
    """
    cur = conn.execute(
        "INSERT INTO reductions(name, source_model, method, params, "
        "n_components) VALUES (?, ?, ?, ?, ?)",
        (str(name), str(source_model), str(method),
         json.dumps(params or {}, ensure_ascii=False), int(n_components)),
    )
    return int(cur.lastrowid)


def update_reduction(conn: sqlite3.Connection, reduction_id: int,
                     name: str, method: str, params: dict | None,
                     n_components: int) -> None:
    """Overwrite a reduction's metadata (a rerun/refresh of the dataset).

    Keeps the id (and therefore the ``red:<id>`` dataset references) while
    the fitted vectors are replaced by
    :func:`replace_reduced_embeddings`.
    """
    conn.execute(
        "UPDATE reductions SET name = ?, method = ?, params = ?, "
        "n_components = ? WHERE id = ?",
        (str(name), str(method),
         json.dumps(params or {}, ensure_ascii=False), int(n_components),
         int(reduction_id)),
    )


def reduction_coverage(conn: sqlite3.Connection, reduction_id: int,
                       source_model: str) -> tuple[int, int]:
    """How current a reduction is: ``(covered, current)`` chunk counts.

    *covered* counts the reduction's vectors that still reference a live
    chunk (re-analysis replaces chunk ids, so old rows can dangle);
    *current* is the source dataset's present chunk-vector count.  A
    reduction is outdated when ``covered < current`` — new chunks are
    missing from the projection.
    """
    covered = conn.execute(
        "SELECT COUNT(*) FROM reduced_embeddings r "
        "JOIN chunks c ON c.id = r.chunk_id WHERE r.reduction_id = ?",
        (int(reduction_id),)).fetchone()[0]
    total = conn.execute(
        "SELECT COUNT(*) FROM embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id WHERE e.model = ?",
        (str(source_model),)).fetchone()[0]
    return int(covered), int(total)


def set_reduction_result(conn: sqlite3.Connection, reduction_id: int,
                         n_vectors: int,
                         explained_variance: list[float] | None = None) -> None:
    """Record the outcome of a finished reduction run."""
    conn.execute(
        "UPDATE reductions SET n_vectors = ?, explained_variance = ? "
        "WHERE id = ?",
        (int(n_vectors),
         None if explained_variance is None
         else json.dumps([float(v) for v in explained_variance]),
         int(reduction_id)),
    )


def list_reductions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All stored reductions, newest first."""
    return conn.execute(
        "SELECT * FROM reductions ORDER BY id DESC").fetchall()


def get_reduction(conn: sqlite3.Connection, reduction_id: int) -> sqlite3.Row | None:
    """One reduction row (``red:<id>`` dataset metadata) or ``None``."""
    return conn.execute(
        "SELECT * FROM reductions WHERE id = ?", (int(reduction_id),),
    ).fetchone()


def delete_reduction(conn: sqlite3.Connection, reduction_id: int) -> None:
    """Drop a reduction, its stored vectors and any of its noise filters."""
    conn.execute(
        "DELETE FROM track_embeddings WHERE model = ?",
        (f"red:{int(reduction_id)}",),
    )
    conn.execute(
        "DELETE FROM noise_filters WHERE dataset = ?",
        (f"red:{int(reduction_id)}",),
    )
    conn.execute("DELETE FROM reductions WHERE id = ?", (int(reduction_id),))


def replace_reduced_embeddings(
    conn: sqlite3.Connection, reduction_id: int,
    items: list[tuple[int, np.ndarray]],
) -> None:
    """Replace the reduced vector set of one reduction with *items*.

    *items* are ``(chunk_id, vector)`` pairs; the previous rows are deleted
    first, so re-running a reduction never leaves stale vectors behind.
    """
    conn.execute("DELETE FROM reduced_embeddings WHERE reduction_id = ?",
                 (int(reduction_id),))
    conn.executemany(
        "INSERT INTO reduced_embeddings(reduction_id, chunk_id, dim, vector, "
        "norm) VALUES (?, ?, ?, ?, ?)",
        [(int(reduction_id), int(chunk_id), int(np.asarray(vec).size),
          vec_to_blob(np.asarray(vec)), float(np.linalg.norm(vec)))
         for chunk_id, vec in items],
    )


def get_reduced_chunk_embeddings(
    conn: sqlite3.Connection, reduction_id: int,
    track_id: int | None = None,
) -> list[sqlite3.Row]:
    """Reduced vectors of one reduction, optionally only one track's chunks.

    Rows carry the decoded ``row["vec"]`` plus ``chunk_id``; ordering
    follows track, then chunk index.
    """
    sql = (
        "SELECT r.chunk_id AS chunk_id, r.dim AS dim, r.vector AS vector, "
        "c.track_id AS track_id, c.idx AS chunk_idx "
        "FROM reduced_embeddings r JOIN chunks c ON c.id = r.chunk_id "
        "WHERE r.reduction_id = ?"
    )
    args: tuple = (int(reduction_id),)
    if track_id is not None:
        sql += " AND c.track_id = ?"
        args = (int(reduction_id), int(track_id))
    rows = conn.execute(sql + " ORDER BY c.track_id, c.idx", args).fetchall()
    return [_EmbeddingRow(row, blob_to_vec(row["vector"])) for row in rows]


def tracks_with_reduced_chunks(conn: sqlite3.Connection,
                               reduction_id: int) -> list[int]:
    """Ids of tracks holding at least one reduced vector, ascending."""
    rows = conn.execute(
        "SELECT DISTINCT c.track_id AS track_id "
        "FROM reduced_embeddings r JOIN chunks c ON c.id = r.chunk_id "
        "WHERE r.reduction_id = ? ORDER BY c.track_id",
        (int(reduction_id),),
    ).fetchall()
    return [int(r["track_id"]) for r in rows]


def get_chunk_vector_models(conn: sqlite3.Connection) -> list[str]:
    """Model names holding at least one chunk embedding, alphabetical."""
    rows = conn.execute(
        "SELECT DISTINCT model FROM embeddings ORDER BY model").fetchall()
    return [str(r["model"]) for r in rows]


def chunk_vector_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """``{model: number of stored chunk vectors}`` for every model."""
    rows = conn.execute(
        "SELECT model, COUNT(*) AS n FROM embeddings GROUP BY model"
    ).fetchall()
    return {str(r["model"]): int(r["n"]) for r in rows}


def set_noise_filter_result(conn: sqlite3.Connection, dataset: str,
                            method: str, params: str, n_vectors: int,
                            n_noise: int, noise_chunk_ids) -> int:
    """Store one noise-filter run, replacing any previous one.

    Returns the new filter id. ``noise_chunk_ids`` are the chunks the
    clustering labelled as noise (label -1).
    """
    conn.execute(
        "DELETE FROM noise_filters WHERE dataset = ? AND method = ?",
        (str(dataset), str(method)))
    cur = conn.execute(
        "INSERT INTO noise_filters(dataset, method, params, n_vectors, "
        "n_noise) VALUES (?, ?, ?, ?, ?)",
        (str(dataset), str(method), params, int(n_vectors), int(n_noise)))
    filter_id = int(cur.lastrowid)
    conn.executemany(
        "INSERT OR IGNORE INTO noise_chunks(filter_id, chunk_id) "
        "VALUES (?, ?)",
        [(filter_id, int(chunk_id)) for chunk_id in noise_chunk_ids])
    return filter_id


def noise_track_signatures(conn: sqlite3.Connection, dataset: str,
                           ) -> dict[int, tuple[int, int]]:
    """``{track_id: (n_chunks, sum(chunk_id))}`` for one vector dataset.

    The cheap per-song signature the noise bookkeeping compares against:
    re-analysis changes a track's chunk ids, which changes both numbers.
    """
    rows = conn.execute(
        "SELECT c.track_id AS track_id, COUNT(*) AS n, SUM(c.id) AS s "
        "FROM chunks c JOIN embeddings e ON e.chunk_id = c.id "
        "WHERE e.model = ? GROUP BY c.track_id", (str(dataset),)).fetchall()
    return {int(r["track_id"]): (int(r["n"]), int(r["s"])) for r in rows}


def record_noise_run_tracks(conn: sqlite3.Connection, dataset: str,
                            method: str, signatures: dict[int,
                                                          tuple[int, int]]
                            ) -> None:
    """Store which tracks a (dataset, method) run has clustered."""
    conn.executemany(
        "INSERT OR REPLACE INTO noise_run_tracks"
        "(dataset, method, track_id, n_chunks, chunk_sum) "
        "VALUES (?, ?, ?, ?, ?)",
        [(str(dataset), str(method), int(track_id), int(n), int(s))
         for track_id, (n, s) in signatures.items()])


def pending_noise_tracks(conn: sqlite3.Connection, dataset: str,
                         method: str, min_track_chunks: int) -> list[int]:
    """Track ids of *dataset* not yet clustered for *method*.

    Pending = no bookkeeping row, or the stored signature no longer
    matches the track's current chunks (re-analyzed / newly analyzed).
    Only songs with at least *min_track_chunks* chunk vectors are
    considered — fewer cannot be density-judged meaningfully.
    """
    current = noise_track_signatures(conn, dataset)
    stored = {int(r["track_id"]): (int(r["n_chunks"]), int(r["chunk_sum"]))
              for r in conn.execute(
                  "SELECT track_id, n_chunks, chunk_sum FROM "
                  "noise_run_tracks WHERE dataset = ? AND method = ?",
                  (str(dataset), str(method))).fetchall()}
    return sorted(track_id for track_id, sig in current.items()
                  if sig[0] >= min_track_chunks and stored.get(track_id)
                  != sig)


def get_noise_filter(conn: sqlite3.Connection, dataset: str,
                     method: str) -> sqlite3.Row | None:
    """The stored noise-filter run for *dataset* + *method*, if any."""
    row = conn.execute(
        "SELECT * FROM noise_filters WHERE dataset = ? AND method = ?",
        (str(dataset), str(method))).fetchone()
    return row


def update_noise_filter_result(conn: sqlite3.Connection, filter_id: int,
                               params: str, n_vectors: int,
                               noise_chunk_ids) -> None:
    """Replace the flag set of one stored filter row IN PLACE.

    The incremental post-analysis path uses this: only the given chunk ids
    change — other songs' flags in the same (dataset, method) filter stay
    untouched.  ``n_noise`` is recomputed from the new set.
    """
    conn.execute(
        "UPDATE noise_filters SET params = ?, n_vectors = ?, n_noise = ? "
        "WHERE id = ?",
        (str(params), int(n_vectors), len(set(noise_chunk_ids)),
         int(filter_id)))
    conn.execute("DELETE FROM noise_chunks WHERE filter_id = ?",
                 (int(filter_id),))
    conn.executemany(
        "INSERT OR IGNORE INTO noise_chunks(filter_id, chunk_id) "
        "VALUES (?, ?)",
        [(int(filter_id), int(chunk_id)) for chunk_id in noise_chunk_ids])


def list_noise_filters(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All stored noise-filter runs, newest first."""
    return conn.execute(
        "SELECT * FROM noise_filters ORDER BY created_at DESC, id DESC"
    ).fetchall()


def get_noise_chunk_ids(conn: sqlite3.Connection,
                        filter_id: int) -> set[int]:
    """Chunk ids the given filter run labelled as noise."""
    rows = conn.execute(
        "SELECT chunk_id FROM noise_chunks WHERE filter_id = ?",
        (int(filter_id),)).fetchall()
    return {int(r["chunk_id"]) for r in rows}


def delete_noise_filter(conn: sqlite3.Connection, filter_id: int) -> None:
    """Drop one noise-filter run and its chunk flags (FK cascade)."""
    conn.execute("DELETE FROM noise_filters WHERE id = ?", (int(filter_id),))
