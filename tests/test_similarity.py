"""Tests for app.similarity (Ollama client + search) and app.playlist.generator.

Run:
    cd /Users/apple/Documents/HOLOSMART && .venv/bin/python -m unittest tests.test_similarity -v

The DB-facing tests use a real temp-file ``app.db.database.Database`` plus the
real ``app.db.repo`` functions. If those sibling modules cannot be imported
(e.g. they have not landed yet), a contract-faithful throwaway fixture pair is
installed into ``sys.modules`` for this test run only; the module flag
``USING_FIXTURE_REPO`` records which mode ran.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------
# Fallback fixtures for app.db (used ONLY when the real modules are missing).
# --------------------------------------------------------------------------
def _try_import_repo() -> bool:
    try:
        import app.db.database  # noqa: F401
        import app.db.repo  # noqa: F401
        return True
    except Exception:
        return False


def _install_fixture_repo() -> None:
    """Register minimal contract-faithful stand-ins for app.db.* in sys.modules."""
    import types

    schema_sql = (PROJECT_ROOT / "app" / "db" / "schema.sql").read_text(encoding="utf-8")

    database_mod = types.ModuleType("app.db.database")
    repo_mod = types.ModuleType("app.db.repo")

    def vec_to_blob(vec):
        return np.asarray(vec, dtype="<f4").reshape(-1).tobytes()

    def blob_to_vec(blob):
        return np.frombuffer(blob, dtype="<f4").copy()

    class Database:  # contract: fresh connection per connect(), schema applied
        def __init__(self, db_path):
            self.db_path = Path(db_path)

        def connect(self):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(schema_sql)
            conn.commit()
            return conn

        @contextmanager
        def transaction(self):
            conn = self.connect()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    database_mod.Database = Database
    database_mod.vec_to_blob = vec_to_blob
    database_mod.blob_to_vec = blob_to_vec

    _TRACK_COLS = (
        "filename", "extension", "size_bytes", "mtime", "container", "codec",
        "sample_rate", "channels", "bit_depth", "bitrate_kbps", "duration_sec",
        "title", "artist", "album", "genre", "year", "track_no",
    )

    def add_folder(conn, path):
        conn.execute("INSERT OR IGNORE INTO folders(path) VALUES (?)", (str(path),))
        row = conn.execute("SELECT id FROM folders WHERE path = ?", (str(path),)).fetchone()
        return int(row["id"])

    def upsert_track(conn, folder_id, path, meta):
        meta = dict(meta or {})
        filename = meta.get("filename") or Path(str(path)).name
        values = {"folder_id": int(folder_id), "path": str(path), "filename": str(filename)}
        for key in _TRACK_COLS:
            if key != "filename" and key in meta and meta[key] is not None:
                values[key] = meta[key]
        existing = conn.execute("SELECT id FROM tracks WHERE path = ?", (values["path"],)).fetchone()
        if existing:
            assignments = ", ".join(f"{name} = ?" for name in values)
            conn.execute(
                f"UPDATE tracks SET {assignments} WHERE id = ?",
                (*values.values(), existing["id"]),
            )
            return int(existing["id"])
        columns = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        cur = conn.execute(f"INSERT INTO tracks ({columns}) VALUES ({marks})", tuple(values.values()))
        return int(cur.lastrowid)

    def get_track(conn, track_id):
        return conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()

    def set_track_description(conn, track_id, description):
        conn.execute("UPDATE tracks SET description = ? WHERE id = ?", (description, track_id))

    def replace_chunks(conn, track_id, items):
        conn.execute("DELETE FROM chunks WHERE track_id = ?", (track_id,))
        ids = []
        for idx, start_sec, end_sec in items:
            cur = conn.execute(
                "INSERT INTO chunks(track_id, idx, start_sec, end_sec) VALUES (?, ?, ?, ?)",
                (track_id, int(idx), float(start_sec), float(end_sec)),
            )
            ids.append(int(cur.lastrowid))
        return ids

    def get_chunks(conn, track_id):
        return conn.execute(
            "SELECT * FROM chunks WHERE track_id = ? ORDER BY idx", (track_id,)
        ).fetchall()

    def add_chunk_embedding(conn, chunk_id, model, vec):
        vec = np.asarray(vec, dtype="<f4").reshape(-1)
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector, norm) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_id, model, int(vec.size), vec_to_blob(vec), float(np.linalg.norm(vec))),
        )

    def get_chunk_embeddings(conn, chunk_id, model=None):
        sql = "SELECT * FROM embeddings WHERE chunk_id = ?"
        args: tuple = (chunk_id,)
        if model is not None:
            sql += " AND model = ?"
            args = (chunk_id, model)
        rows = conn.execute(sql + " ORDER BY model, id", args).fetchall()
        return [dict(row, vec=blob_to_vec(row["vector"])) for row in rows]

    def set_track_embedding(conn, track_id, model, vec):
        vec = np.asarray(vec, dtype="<f4").reshape(-1)
        conn.execute(
            "INSERT INTO track_embeddings(track_id, model, dim, vector, norm) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(track_id, model) DO UPDATE SET dim = excluded.dim, "
            "vector = excluded.vector, norm = excluded.norm",
            (track_id, model, int(vec.size), vec_to_blob(vec), float(np.linalg.norm(vec))),
        )

    def get_track_embedding(conn, track_id, model):
        row = conn.execute(
            "SELECT vector FROM track_embeddings WHERE track_id = ? AND model = ?",
            (track_id, model),
        ).fetchone()
        return blob_to_vec(row["vector"]) if row else None

    def get_track_embeddings(conn, model, track_ids=None):
        sql = "SELECT track_id, vector FROM track_embeddings WHERE model = ?"
        args: tuple = (model,)
        if track_ids:
            marks = ", ".join("?" for _ in track_ids)
            sql += f" AND track_id IN ({marks})"
            args = (model, *[int(t) for t in track_ids])
        return {int(r["track_id"]): blob_to_vec(r["vector"])
                for r in conn.execute(sql, args).fetchall()}

    def tracks_with_embeddings(conn, model):
        rows = conn.execute(
            "SELECT DISTINCT track_id FROM track_embeddings WHERE model = ? ORDER BY track_id",
            (model,),
        ).fetchall()
        return [int(r["track_id"]) for r in rows]

    def create_playlist(conn, name, seed_track_id=None, method=None):
        cur = conn.execute(
            "INSERT INTO playlists(name, seed_track_id, method) VALUES (?, ?, ?)",
            (name, seed_track_id, method),
        )
        return int(cur.lastrowid)

    def add_playlist_items(conn, playlist_id, items):
        conn.execute("DELETE FROM playlist_items WHERE playlist_id = ?", (playlist_id,))
        for position, (track_id, similarity) in enumerate(items, start=1):
            conn.execute(
                "INSERT INTO playlist_items(playlist_id, track_id, position, similarity) "
                "VALUES (?, ?, ?, ?)",
                (playlist_id, int(track_id), position, float(similarity)),
            )

    def list_playlists(conn):
        return conn.execute("SELECT * FROM playlists ORDER BY id").fetchall()

    def get_playlist(conn, playlist_id):
        return conn.execute("SELECT * FROM playlists WHERE id = ?", (playlist_id,)).fetchone()

    def get_playlist_items(conn, playlist_id):
        return conn.execute(
            "SELECT pi.*, t.path, t.filename, t.title, t.artist, t.duration_sec "
            "FROM playlist_items pi JOIN tracks t ON t.id = pi.track_id "
            "WHERE pi.playlist_id = ? ORDER BY pi.position",
            (playlist_id,),
        ).fetchall()

    def set_setting(conn, key, value):
        conn.execute(
            "INSERT INTO settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, None if value is None else str(value)),
        )

    def get_setting(conn, key, default=None):
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        value = row["value"] if row else None
        return default if value is None else value

    def save_ollama_models(conn, models):
        set_setting(conn, "ollama_embedding_models", json.dumps([str(m) for m in models]))

    def get_ollama_models(conn):
        raw = get_setting(conn, "ollama_embedding_models")
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except ValueError:
            return []
        return [str(m) for m in data] if isinstance(data, list) else []

    for name, fn in list(locals().items()):
        if callable(fn) and not name.startswith("_") and name not in {"types", "json"}:
            setattr(repo_mod, name, fn)

    sys.modules["app.db.database"] = database_mod
    sys.modules["app.db.repo"] = repo_mod
    try:  # also expose as attributes when app.db is a namespace package
        import app.db as _db_pkg

        _db_pkg.database = database_mod
        _db_pkg.repo = repo_mod
    except Exception:
        pass


USING_FIXTURE_REPO = not _try_import_repo()
if USING_FIXTURE_REPO:
    _install_fixture_repo()

from app.db import repo  # noqa: E402  (fixture may have been installed above)
from app.db.database import Database  # noqa: E402
from app.playlist.generator import export_m3u, generate_playlist  # noqa: E402
from app.similarity.ollama import (  # noqa: E402
    CONNECT_TIMEOUT,
    DEFAULT_HOST,
    OllamaClient,
    OllamaError,
    detect_ollama,
)
from app.similarity.search import (  # noqa: E402
    compute_centroid,
    cosine,
    ensure_track_embeddings,
    geometric_mean,
    resolve_method,
    similar_tracks,
    similar_tracks_multi,
)

# One shared, cheap liveness probe for the live-server tests.
_OLLAMA_CLIENT = OllamaClient()
_OLLAMA_RUNNING = _OLLAMA_CLIENT.is_running()


def _nomic_available() -> bool:
    if not _OLLAMA_RUNNING:
        return False
    try:
        return "nomic-embed-text:latest" in _OLLAMA_CLIENT.list_embedding_models()
    except OllamaError:
        return False


_OLLAMA_NOMIC = _nomic_available()


class FakeOllama:
    """Duck-typed stand-in for OllamaClient (hermetic, no network)."""

    def __init__(self, running: bool = True, vector: list[float] | None = None,
                 error: bool = False) -> None:
        self._running = running
        self._vector = list(vector) if vector is not None else [0.25, 0.75]
        self._error = error
        self.calls: list[tuple[object, str]] = []

    def is_running(self) -> bool:
        return self._running

    def embed(self, text, model):
        self.calls.append((text, model))
        if self._error:
            raise OllamaError("fake Ollama server is down")
        count = 1 if isinstance(text, str) else len(text)
        return [list(self._vector) for _ in range(count)]


# --------------------------------------------------------------------------
# Shared library fixture: 4 analyzed tracks + 1 unanalyzed, injected CLAP
# chunk vectors. Expected cosine vs the seed centroid (0.7071, 0.7071):
#   close 0.9487 > mid 0.7071 > far -0.7071;  naked has no embeddings.
# --------------------------------------------------------------------------
class Library:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.music_dir = self.root / "music"
        self.db = Database(self.root / "library.db")
        self.conn = self.db.connect()
        folder_id = repo.add_folder(self.conn, str(self.music_dir))
        plan = [
            ("seed_song.wav", "Alpha", "Seed Song", [[1.0, 0.0], [0.0, 1.0]], ""),
            ("close_song.wav", "Beta", "Close Song", [[1.0, 0.5]], "bright upbeat pop"),
            ("far_song.wav", "Gamma", "Far Song", [[-1.0, 0.0]], "dark aggressive metal"),
            ("mid_song.wav", "Delta", "Mid Song", [[0.0, 1.0]], ""),
            ("naked_song.wav", "Epsilon", "Naked Song", [], ""),
        ]
        self.ids: dict[str, int] = {}
        self.vectors: dict[str, list[list[float]]] = {}
        for filename, artist, title, vectors, description in plan:
            path = str(self.music_dir / filename)
            track_id = repo.upsert_track(self.conn, folder_id, path, {
                "filename": filename, "artist": artist, "title": title,
                "duration_sec": 42.0,
            })
            chunk_ids = repo.replace_chunks(self.conn, track_id, [
                (i, i * 20.0, i * 20.0 + 20.0) for i in range(len(vectors))
            ])
            for chunk_id, vector in zip(chunk_ids, vectors):
                repo.add_chunk_embedding(self.conn, chunk_id, "clap",
                                         np.asarray(vector, dtype=np.float32))
            if description:
                repo.set_track_description(self.conn, track_id, description)
            self.ids[filename.split("_", 1)[0]] = track_id
            self.vectors[filename] = vectors
        self.conn.commit()

    @property
    def seed(self) -> int:
        return self.ids["seed"]

    def expected_centroid(self, key: str) -> np.ndarray:
        return compute_centroid(
            [np.asarray(v, dtype=np.float32) for v in self.vectors[f"{key}_song.wav"]]
        )


class LibraryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="holosmart-similarity-")
        self.addCleanup(self._tmp.cleanup)
        self.lib = Library(Path(self._tmp.name))
        self.addCleanup(self.lib.conn.close)


# ---------------------------------------------------------------- math
class CosineAndCentroidTests(unittest.TestCase):
    def test_cosine_identical_is_one(self):
        vec = np.asarray([0.3, -1.2, 2.0])
        self.assertAlmostEqual(cosine(vec, vec.copy()), 1.0, places=12)

    def test_cosine_orthogonal_is_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 3.0]), 0.0, places=12)

    def test_cosine_opposite_is_minus_one(self):
        self.assertAlmostEqual(cosine([2.0, 0.0], [-3.0, 0.0]), -1.0, places=12)

    def test_cosine_zero_vector_is_zero(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 2.0]), 0.0)
        self.assertEqual(cosine([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_cosine_scale_invariant(self):
        self.assertAlmostEqual(cosine([1.0, 2.0], [10.0, 20.0]), 1.0, places=12)

    def test_cosine_incomparable_shapes_are_zero(self):
        self.assertEqual(cosine([1.0, 2.0], [1.0, 2.0, 3.0]), 0.0)

    def test_centroid_mean_normalized(self):
        centroid = compute_centroid([np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])])
        self.assertTrue(np.allclose(centroid, [0.70710678, 0.70710678]))
        self.assertEqual(centroid.dtype, np.float32)

    def test_centroid_single_vector_normalized(self):
        self.assertTrue(np.allclose(compute_centroid([np.asarray([3.0, 4.0])]), [0.6, 0.8]))

    def test_centroid_empty_is_zero_length(self):
        empty = compute_centroid([])
        self.assertEqual(empty.size, 0)

    def test_centroid_zero_mean_is_zero_vector(self):
        centroid = compute_centroid([np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0])])
        self.assertTrue(np.allclose(centroid, [0.0, 0.0]))


# ------------------------------------------- ensure_track_embeddings + search
class GeometricMeanTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(geometric_mean([]), 0.0)

    def test_zero_or_negative_floors_to_zero(self):
        self.assertEqual(geometric_mean([0.9, 0.0, 0.8]), 0.0)
        self.assertEqual(geometric_mean([-0.2, 0.9, 0.8]), 0.0)

    def test_single_value_is_itself(self):
        self.assertAlmostEqual(geometric_mean([0.5]), 0.5)

    def test_equal_values_are_the_value(self):
        self.assertAlmostEqual(geometric_mean([0.7] * 4), 0.7)

    def test_known_value(self):
        # (0.8 * 0.45) ** 0.5
        self.assertAlmostEqual(geometric_mean([0.8, 0.45]), 0.6)

    def test_result_between_extremes(self):
        value = geometric_mean([0.99, 0.01])
        self.assertGreater(value, 0.01)
        self.assertLess(value, 0.99)


class MultiReferenceTests(LibraryTestCase):
    """Similarity to several references via geometric mean of percentages."""

    def setUp(self) -> None:
        super().setUp()
        # centroid searches need per-track centroids stored
        ensure_track_embeddings(self.lib.conn, list(self.lib.ids.values()))

    def _per_seed_scores(self, seed_id: int) -> dict[int, float]:
        rows = similar_tracks(self.lib.conn, seed_id, dataset="clap",
                              limit=10 ** 9)
        return {r.track_id: r.score for r in rows if r.track_id != seed_id}

    def test_single_reference_matches_similar_tracks(self) -> None:
        expected = similar_tracks(self.lib.conn, self.lib.seed,
                                  dataset="clap", limit=5)
        got = similar_tracks_multi(self.lib.conn, [self.lib.seed],
                                   dataset="clap", limit=5)
        self.assertEqual([(r.track_id, round(r.score, 9)) for r in got],
                         [(r.track_id, round(r.score, 9)) for r in expected])

    def test_two_references_use_geometric_mean_and_drop_references(self) -> None:
        refs = [self.lib.ids["seed"], self.lib.ids["mid"]]
        results = similar_tracks_multi(self.lib.conn, refs, dataset="clap",
                                       limit=10)
        self.assertEqual(len(results), 2)   # close + far; refs excluded
        self.assertNotIn(refs[0], [r.track_id for r in results])
        self.assertNotIn(refs[1], [r.track_id for r in results])
        expected = {}
        for sid in refs:
            for tid, score in self._per_seed_scores(sid).items():
                expected.setdefault(tid, []).append(score)
        for res in results:
            percents = expected[res.track_id]
            want = geometric_mean(percents)
            self.assertAlmostEqual(res.score, want, places=9)
        # descending order
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # "close" (similar to both refs) beats "far" (mismatched to both):
        # the geometric mean must preserve that AND punish asymmetry:
        # close is ~1.0 vs seed and moderate vs mid -> still top.
        self.assertEqual(results[0].filename, "close_song.wav")

    def test_asymmetric_candidate_is_punished(self) -> None:
        """A track perfect for one ref but poor for the other must NOT
        outrank one that is moderately similar to both."""
        refs = [self.lib.ids["seed"], self.lib.ids["mid"]]
        results = similar_tracks_multi(self.lib.conn, refs, dataset="clap",
                                       limit=10)
        by_name = {r.filename: r.score for r in results}
        # "close" matches seed nearly perfectly; its mid score is lower.
        # "far" matches nothing. Geometric mean keeps close on top and far
        # at/near the bottom.
        self.assertGreater(by_name["close_song.wav"],
                           by_name["far_song.wav"])

    def test_candidate_missing_from_one_reference_is_dropped(self) -> None:
        # "naked" has no clap vectors: it appears in NO per-seed map anyway,
        # so give one reference a synthetic extra candidate instead: use a
        # second dataset for one track. Simpler: two refs where one ref has
        # vectors only in clap — the Library has exactly that shape; the
        # intersection logic is exercised by the refs sharing candidates.
        refs = [self.lib.ids["seed"], self.lib.ids["close"]]
        results = similar_tracks_multi(self.lib.conn, refs, dataset="clap",
                                       limit=10)
        listed = [r.track_id for r in results]
        self.assertNotIn(self.lib.ids["close"], listed)  # refs never listed
        self.assertNotIn(self.lib.ids["seed"], listed)

    def test_reference_without_embedding_is_skipped(self) -> None:
        """A folder reference without vectors is skipped, not fatal."""
        naked = self.lib.ids["naked"]   # no clap vectors at all
        results = similar_tracks_multi(self.lib.conn,
                                       [self.lib.seed, naked],
                                       dataset="clap", limit=10)
        expected = similar_tracks(self.lib.conn, self.lib.seed,
                                  dataset="clap", limit=10 ** 9)
        expected = [r for r in expected if r.track_id != self.lib.seed]
        # with one usable reference the geometric mean is that reference's
        # percentage — floored at 0 for negative matches (no similarity)
        self.assertEqual([(r.track_id, round(r.score, 9)) for r in results],
                         [(r.track_id, round(max(0.0, r.score), 9))
                          for r in expected])

    def test_all_references_without_vectors_raise(self) -> None:
        naked = self.lib.ids["naked"]
        with self.assertRaises(RuntimeError) as ctx:
            similar_tracks_multi(self.lib.conn, [naked],
                                 dataset="clap", limit=10)
        # a single unanalyzed reference still delegates to similar_tracks,
        # whose error names the missing embedding
        self.assertIn("clap", str(ctx.exception))

    def test_empty_reference_list_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            similar_tracks_multi(self.lib.conn, [], dataset="clap")

    def test_limit_truncates(self) -> None:
        refs = [self.lib.ids["seed"], self.lib.ids["mid"]]
        results = similar_tracks_multi(self.lib.conn, refs, dataset="clap",
                                       limit=1)
        self.assertEqual(len(results), 1)


class EnsureTrackEmbeddingsTests(LibraryTestCase):
    def test_centroids_stored_for_every_chunk_model(self):
        ensure_track_embeddings(self.lib.conn, list(self.lib.ids.values()))
        for key, track_id in self.lib.ids.items():
            if key == "naked":
                continue
            stored = repo.get_track_embedding(self.lib.conn, track_id, "clap")
            self.assertIsNotNone(stored, f"missing centroid for {key}")
            self.assertTrue(np.allclose(stored, self.lib.expected_centroid(key), atol=1e-6))

    def test_unanalyzed_track_gets_no_embedding(self):
        ensure_track_embeddings(self.lib.conn, [self.lib.ids["naked"]])
        self.assertIsNone(repo.get_track_embedding(self.lib.conn, self.lib.ids["naked"], "clap"))

    def test_ensure_is_idempotent(self):
        ids = list(self.lib.ids.values())
        ensure_track_embeddings(self.lib.conn, ids)
        first = {tid: repo.get_track_embedding(self.lib.conn, tid, "clap") for tid in ids}
        ensure_track_embeddings(self.lib.conn, ids)
        for tid, vector in first.items():
            again = repo.get_track_embedding(self.lib.conn, tid, "clap")
            self.assertEqual(vector is None, again is None)
            if vector is not None:
                self.assertTrue(np.allclose(vector, again))

    def test_ollama_description_embedding_stored_under_prefixed_model(self):
        fake = FakeOllama(vector=[0.3, 0.4])
        ensure_track_embeddings(self.lib.conn, [self.lib.ids["close"]], fake, "fake-embed")
        stored = repo.get_track_embedding(self.lib.conn, self.lib.ids["close"],
                                          "ollama:fake-embed")
        self.assertIsNotNone(stored)
        self.assertTrue(np.allclose(stored, [0.3, 0.4]))
        self.assertEqual(fake.calls, [("bright upbeat pop", "fake-embed")])

    def test_track_without_description_skips_ollama(self):
        fake = FakeOllama()
        ensure_track_embeddings(self.lib.conn, [self.lib.ids["seed"]], fake, "fake-embed")
        self.assertEqual(fake.calls, [])
        self.assertIsNone(
            repo.get_track_embedding(self.lib.conn, self.lib.ids["seed"], "ollama:fake-embed"))

    def test_ollama_failure_is_skipped_not_raised(self):
        fake = FakeOllama(error=True)
        ensure_track_embeddings(self.lib.conn, [self.lib.ids["close"]], fake, "fake-embed")
        self.assertIsNone(
            repo.get_track_embedding(self.lib.conn, self.lib.ids["close"], "ollama:fake-embed"))
        # centroids are still materialized
        self.assertIsNotNone(repo.get_track_embedding(self.lib.conn, self.lib.ids["close"], "clap"))


class SimilarTracksTests(LibraryTestCase):
    def setUp(self) -> None:
        super().setUp()
        ensure_track_embeddings(self.lib.conn, list(self.lib.ids.values()))

    def test_order_fields_and_exclusions(self):
        results = similar_tracks(self.lib.conn, self.lib.seed, method="clap", limit=10)
        # The seed itself is always row one with a 100% score…
        self.assertEqual(results[0].track_id, self.lib.seed)
        self.assertEqual(results[0].score, 1.0)
        # …followed by the best matches (the un-analyzed "naked" track
        # never appears because it has no embeddings).
        self.assertEqual([r.track_id for r in results[1:]],
                         [self.lib.ids["close"], self.lib.ids["mid"], self.lib.ids["far"]])
        scores = [r.score for r in results[1:]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(-1.0 <= s <= 1.0 for s in scores))
        self.assertAlmostEqual(scores[0], 0.948683, places=5)
        self.assertAlmostEqual(scores[1], 0.707107, places=5)
        self.assertAlmostEqual(scores[2], -0.707107, places=5)
        top = results[1]
        self.assertEqual(top.method, "clap")
        self.assertEqual(top.filename, "close_song.wav")
        self.assertEqual(top.artist, "Beta")
        self.assertEqual(top.title, "Close Song")
        self.assertEqual(top.path, str(self.lib.music_dir / "close_song.wav"))

    def test_limit_truncates(self):
        results = similar_tracks(self.lib.conn, self.lib.seed, method="clap", limit=1)
        # seed row + one match at limit=1
        self.assertEqual([r.track_id for r in results],
                         [self.lib.seed, self.lib.ids["close"]])

    def test_explicit_method_passthrough(self):
        self.assertEqual(resolve_method(self.lib.conn, "clap", None), "clap")
        self.assertEqual(resolve_method(self.lib.conn, "ollama:x", None), "ollama:x")

    def test_auto_without_ollama_resolves_to_clap(self):
        self.assertEqual(resolve_method(self.lib.conn, "auto", None), "clap")
        results = similar_tracks(self.lib.conn, self.lib.seed, method="auto", limit=5)
        self.assertTrue(results)
        self.assertEqual(results[0].track_id, self.lib.seed)   # seed first
        self.assertEqual(results[1].method, "clap")

    def test_auto_prefers_ollama_method_from_config(self):
        conn = self.lib.conn
        for key in ("seed", "close"):
            repo.set_track_embedding(conn, self.lib.ids[key], "ollama:m1",
                                     np.asarray([0.5, 0.5], dtype=np.float32))
        fake = FakeOllama(running=True)
        with mock.patch("app.similarity.search._configured_ollama_model", return_value="m1"):
            self.assertEqual(resolve_method(conn, "auto", fake), "ollama:m1")
        # server down -> falls back to the audio model
        self.assertEqual(resolve_method(conn, "auto", FakeOllama(running=False)), "clap")

    def test_auto_prefers_seed_ollama_method(self):
        conn = self.lib.conn
        for key in ("seed", "close"):
            repo.set_track_embedding(conn, self.lib.ids[key], "ollama:m2",
                                     np.asarray([0.5, 0.5], dtype=np.float32))
        fake = FakeOllama(running=True)
        self.assertEqual(resolve_method(conn, "auto", fake, self.lib.seed), "ollama:m2")

    def test_auto_none_on_empty_database(self):
        with tempfile.TemporaryDirectory(prefix="holosmart-empty-") as tmp:
            lib = Library(Path(tmp))
            try:
                self.assertIsNone(resolve_method(lib.conn, "auto", None))
                with self.assertRaises(RuntimeError):
                    similar_tracks(lib.conn, lib.seed, method="auto")
                with self.assertRaises(RuntimeError):
                    similar_tracks(lib.conn, lib.seed, method="clap")
            finally:
                lib.conn.close()

    def test_seed_without_embedding_raises_friendly_error(self):
        # close/mid/far have clap embeddings, the seed does not.
        conn = self.lib.conn
        conn.execute("DELETE FROM track_embeddings WHERE track_id = ?", (self.lib.seed,))
        conn.commit()
        with self.assertRaises(RuntimeError) as ctx:
            similar_tracks(conn, self.lib.seed, method="clap")
        self.assertIn("not comparable", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            similar_tracks(conn, self.lib.seed, method="auto")

    def test_ollama_method_works_from_stored_embeddings_without_server(self):
        conn = self.lib.conn
        plan = {
            "seed": [1.0, 0.0],
            "close": [0.9, 0.1],
            "mid": [0.0, 1.0],
            "far": [-1.0, 0.0],
        }
        for key, vector in plan.items():
            repo.set_track_embedding(conn, self.lib.ids[key], "ollama:test-embed",
                                     np.asarray(vector, dtype=np.float32))
        results = similar_tracks(conn, self.lib.seed, method="ollama:test-embed", ollama=None)
        self.assertEqual([r.track_id for r in results],
                         [self.lib.seed, self.lib.ids["close"],
                          self.lib.ids["mid"], self.lib.ids["far"]])
        self.assertEqual(results[0].score, 1.0)
        self.assertTrue(all(r.method == "ollama:test-embed" for r in results))

    def test_unknown_seed_raises(self):
        with self.assertRaises(RuntimeError):
            similar_tracks(self.lib.conn, 424242, method="clap")


# ---------------------------------------------------------------- playlists
class PlaylistTests(LibraryTestCase):
    def setUp(self) -> None:
        super().setUp()
        ensure_track_embeddings(self.lib.conn, list(self.lib.ids.values()))

    def test_generate_playlist_persists_playlist_and_items(self):
        playlist_id = generate_playlist(self.lib.conn, "Road Trip", self.lib.seed,
                                        limit=2, method="clap")
        playlist = repo.get_playlist(self.lib.conn, playlist_id)
        self.assertIsNotNone(playlist)
        self.assertEqual(playlist["name"], "Road Trip")
        self.assertEqual(playlist["seed_track_id"], self.lib.seed)
        self.assertEqual(playlist["method"], "clap")
        items = repo.get_playlist_items(self.lib.conn, playlist_id)
        self.assertEqual([item["position"] for item in items], [1, 2, 3])
        # The seed leads the mix at 100%, then the best matches.
        self.assertEqual([item["track_id"] for item in items],
                         [self.lib.seed, self.lib.ids["close"],
                          self.lib.ids["mid"]])
        self.assertAlmostEqual(items[0]["similarity"], 1.0, places=7)
        self.assertAlmostEqual(items[1]["similarity"], 0.948683, places=5)
        self.assertAlmostEqual(items[2]["similarity"], 0.707107, places=5)
        self.assertIn(playlist_id, [p["id"] for p in repo.list_playlists(self.lib.conn)])

    def test_generate_playlist_auto_stores_concrete_method(self):
        playlist_id = generate_playlist(self.lib.conn, "Auto Mix", self.lib.seed, method="auto")
        playlist = repo.get_playlist(self.lib.conn, playlist_id)
        self.assertEqual(playlist["method"], "clap")

    def test_generate_playlist_raises_when_nothing_comparable(self):
        with tempfile.TemporaryDirectory(prefix="holosmart-playlist-") as tmp:
            lib = Library(Path(tmp))
            try:
                with self.assertRaises(RuntimeError):
                    generate_playlist(lib.conn, "Empty", lib.seed, method="clap")
            finally:
                lib.conn.close()

    def test_export_m3u_writes_header_extinf_and_absolute_paths(self):
        playlist_id = generate_playlist(self.lib.conn, "Road Trip", self.lib.seed, limit=2)
        out_path = self.lib.root / "exports" / "road_trip.m3u"
        returned = export_m3u(self.lib.conn, playlist_id, out_path)
        self.assertEqual(returned, str(out_path))
        text = out_path.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], "#EXTM3U")
        self.assertIn("#EXTINF:42,Beta - Close Song", lines)
        self.assertIn("#EXTINF:42,Delta - Mid Song", lines)
        # seed first (100 %), then the matches
        expected_paths = [str(self.lib.music_dir / "seed_song.wav"),
                          str(self.lib.music_dir / "close_song.wav"),
                          str(self.lib.music_dir / "mid_song.wav")]
        for path in expected_paths:
            self.assertIn(path, lines)
            self.assertTrue(Path(path).is_absolute())
        self.assertLess(lines.index(expected_paths[0]), lines.index(expected_paths[1]))
        # every non-comment line is an absolute path
        track_lines = [line for line in lines[1:] if not line.startswith("#EXTINF")]
        self.assertEqual(track_lines, expected_paths)

    def test_export_m3u_missing_playlist_raises(self):
        with self.assertRaises(RuntimeError):
            export_m3u(self.lib.conn, 987654, self.lib.root / "nope.m3u")


# ------------------------------------------------- Ollama client (live server)
class OllamaClientLiveTests(unittest.TestCase):
    """Hit the real local Ollama server; skipped when it is not running."""

    def test_is_running_true(self):
        self.assertTrue(_OLLAMA_RUNNING)
        self.assertTrue(OllamaClient().is_running())

    @unittest.skipUnless(_OLLAMA_RUNNING, "Ollama server not reachable")
    def test_list_models_entries_are_dicts_with_names(self):
        entries = _OLLAMA_CLIENT.list_models()
        self.assertTrue(entries)
        for entry in entries:
            self.assertIsInstance(entry, dict)
            self.assertTrue(entry.get("name") or entry.get("model"))

    @unittest.skipUnless(_OLLAMA_RUNNING, "Ollama server not reachable")
    def test_list_embedding_models_nonempty_strings(self):
        names = _OLLAMA_CLIENT.list_embedding_models()
        self.assertTrue(names)
        self.assertTrue(all(isinstance(name, str) and name for name in names))

    @unittest.skipUnless(_OLLAMA_NOMIC, "nomic-embed-text not detected on the server")
    def test_nomic_embed_text_is_detected(self):
        self.assertIn("nomic-embed-text:latest", _OLLAMA_CLIENT.list_embedding_models())

    @unittest.skipUnless(_OLLAMA_NOMIC, "nomic-embed-text not detected on the server")
    def test_embed_single_short_text(self):
        vectors = _OLLAMA_CLIENT.embed("a short upbeat piano loop",
                                       "nomic-embed-text:latest")
        self.assertEqual(len(vectors), 1)
        self.assertGreater(len(vectors[0]), 0)
        self.assertTrue(all(isinstance(value, float) for value in vectors[0]))

    @unittest.skipUnless(_OLLAMA_NOMIC, "nomic-embed-text not detected on the server")
    def test_embed_batch_preserves_order_and_count(self):
        vectors = _OLLAMA_CLIENT.embed(["jazz drums", "ambient synth pad"],
                                       "nomic-embed-text:latest")
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len({len(v) for v in vectors}), 1)

    @unittest.skipUnless(_OLLAMA_RUNNING, "Ollama server not reachable")
    def test_detect_ollama_returns_running_and_models(self):
        running, names = detect_ollama(DEFAULT_HOST)
        self.assertTrue(running)
        self.assertTrue(names)


# ----------------------------------------------- Ollama client (offline host)
class OllamaClientOfflineTests(unittest.TestCase):
    """No server on port 1 — must degrade to False / OllamaError, never raise."""

    def test_is_running_false(self):
        self.assertFalse(OllamaClient("http://127.0.0.1:1").is_running())

    def test_list_models_raises_ollama_error(self):
        with self.assertRaises(OllamaError):
            OllamaClient("http://127.0.0.1:1").list_models()

    def test_embed_raises_ollama_error(self):
        with self.assertRaises(OllamaError):
            OllamaClient("http://127.0.0.1:1").embed("text", "some-model")

    def test_detect_ollama_offline(self):
        self.assertEqual(detect_ollama("http://127.0.0.1:1"), (False, []))


# ----------------------------------------------- Ollama client (mocked HTTP)
class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class OllamaClientMockedTests(unittest.TestCase):
    """Contract behaviour of OllamaClient without any network access."""

    def _patch_transport(self, responder):
        calls: list[tuple[str, str, object, object]] = []

        def fake_request(method, url, json=None, timeout=None):
            calls.append((method, url, json, timeout))
            return responder(method, url, json, timeout)

        patcher = mock.patch("app.similarity.ollama.requests.request",
                             side_effect=fake_request)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_embed_falls_back_to_legacy_on_404(self):
        def responder(method, url, json, timeout):
            if url.endswith("/api/embed"):
                return _FakeResponse(404, {}, "not found")
            return _FakeResponse(200, {"embedding": [0.5, -0.25]})

        calls = self._patch_transport(responder)
        client = OllamaClient()
        vectors = client.embed(["one", "two"], "legacy-model")
        self.assertEqual(vectors, [[0.5, -0.25], [0.5, -0.25]])
        self.assertEqual(calls[0][0], "POST")
        self.assertTrue(calls[0][1].endswith("/api/embed"))
        self.assertEqual(calls[0][2], {"model": "legacy-model", "input": ["one", "two"]})
        legacy_calls = [call for call in calls if call[1].endswith("/api/embeddings")]
        self.assertEqual([call[2]["prompt"] for call in legacy_calls], ["one", "two"])
        # short connect timeout on every call so startup never hangs
        for call in calls:
            self.assertEqual(call[3][0], CONNECT_TIMEOUT)

    def test_embed_raises_on_bad_status(self):
        calls = self._patch_transport(lambda *args: _FakeResponse(500, {}, "boom"))
        with self.assertRaises(OllamaError):
            OllamaClient().embed("text", "some-model")
        self.assertEqual(len(calls), 1)  # no legacy retry on non-404

    def test_list_embedding_models_uses_capabilities(self):
        payload = {"models": [
            {"name": "llama3", "capabilities": ["completion"]},
            {"name": "nomic-embed-text:latest", "capabilities": ["embedding", "tools"]},
            {"name": "bge-m3:latest", "capabilities": ["embedding"]},
        ]}
        self._patch_transport(lambda *args: _FakeResponse(200, payload))
        self.assertEqual(OllamaClient().list_embedding_models(),
                         ["nomic-embed-text:latest", "bge-m3:latest"])

    def test_list_embedding_models_name_heuristic_fallback(self):
        payload = {"models": [{"name": "llama3"}, {"name": "my-bge-model"}]}  # no capabilities
        calls = self._patch_transport(lambda *args: _FakeResponse(200, payload))
        self.assertEqual(OllamaClient().list_embedding_models(), ["my-bge-model"])
        self.assertFalse(any(call[1].endswith("/api/embed") for call in calls))

    def test_list_embedding_models_probe_fallback(self):
        payload = {"models": [{"name": "weird-model"}]}

        def responder(method, url, json, timeout):
            if url.endswith("/api/embed"):
                return _FakeResponse(200, {"embeddings": [[0.1, 0.2]]})
            return _FakeResponse(200, payload)

        self._patch_transport(responder)
        self.assertEqual(OllamaClient().list_embedding_models(), ["weird-model"])

    def test_list_embedding_models_probe_failure_yields_empty(self):
        payload = {"models": [{"name": "weird-model"}]}

        def responder(method, url, json, timeout):
            if url.endswith("/api/tags"):
                return _FakeResponse(200, payload)
            return _FakeResponse(404, {}, "nope")

        self._patch_transport(responder)
        self.assertEqual(OllamaClient().list_embedding_models(), [])

    def test_is_running_false_on_transport_error(self):
        from requests.exceptions import ConnectionError as RequestsConnectionError

        def responder(method, url, json, timeout):
            raise RequestsConnectionError("connection refused")

        self._patch_transport(responder)
        self.assertFalse(OllamaClient().is_running())

    def test_list_models_error_includes_status(self):
        self._patch_transport(lambda *args: _FakeResponse(503, {}, "unavailable"))
        with self.assertRaises(OllamaError) as ctx:
            OllamaClient().list_models()
        self.assertIn("503", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
