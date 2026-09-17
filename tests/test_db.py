"""Tests for app.db.database and app.db.repo.

Headless unittest suite against a fresh SQLite file in a per-test temporary
directory — no network, no GUI, no real audio files.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.db import repo
from app.db.database import Database, blob_to_vec, vec_to_blob


class _TempDbTestCase(unittest.TestCase):
    """Base class: fresh Database plus one open connection per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="holosmart-test-db-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.db = Database(self.tmp / "library.db")
        self.conn = self.db.connect()
        self.addCleanup(self.conn.close)

    # ------------------------------------------------------------ helpers --
    def add_track(self, folder_path: str = "/music", filename: str = "a.mp3",
                  meta: dict | None = None) -> tuple[int, int]:
        """Create one folder + track; return ``(folder_id, track_id)``."""
        fid = repo.add_folder(self.conn, folder_path)
        path = f"{folder_path.rstrip('/')}/{filename}"
        tid = repo.upsert_track(self.conn, fid, path, dict(meta or {}))
        return fid, tid

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.conn.execute(sql, params).fetchone()["n"])


class DatabaseConnectionTests(_TempDbTestCase):
    """Database.connect/transaction behaviour and idempotent schema setup."""

    def test_connect_applies_pragmas_and_row_factory(self) -> None:
        self.assertIs(self.conn.row_factory, sqlite3.Row)
        self.assertEqual(self.conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(
            self.conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
        self.assertEqual(self.conn.execute("PRAGMA synchronous").fetchone()[0], 1)

    def test_schema_created_and_idempotent_on_reconnect(self) -> None:
        expected = {"folders", "tracks", "chunks", "embeddings", "chunk_tags",
                    "track_embeddings", "playlists", "playlist_items", "settings"}

        def tables(conn: sqlite3.Connection) -> set[str]:
            return {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}

        self.assertTrue(expected <= tables(self.conn))
        repo.add_folder(self.conn, "/music")
        self.conn.commit()

        conn2 = self.db.connect()  # second connect: schema reapplied idempotently
        self.addCleanup(conn2.close)
        self.assertTrue(expected <= tables(conn2))
        self.assertEqual(len(tables(conn2) & expected), len(expected))
        # data written before the reconnect survived
        self.assertIsNotNone(repo.get_folder_by_path(conn2, "/music"))
        conn3 = self.db.connect()  # and a third connect is just as uneventful
        self.addCleanup(conn3.close)
        self.assertEqual(len(repo.list_folders(conn3)), 1)

    def test_database_accepts_str_path(self) -> None:
        db = Database(str(self.tmp / "str-path.db"))
        conn = db.connect()
        self.addCleanup(conn.close)
        self.assertIsInstance(repo.add_folder(conn, "/somewhere"), int)

    def test_transaction_commits(self) -> None:
        with self.db.transaction() as tx:
            self.assertIsNot(tx, self.conn)
            self.assertIs(tx.row_factory, sqlite3.Row)
            repo.add_folder(tx, "/tx/committed")
        conn2 = self.db.connect()
        self.addCleanup(conn2.close)
        self.assertIsNotNone(repo.get_folder_by_path(conn2, "/tx/committed"))
        self.assertIsNotNone(repo.get_folder_by_path(self.conn, "/tx/committed"))

    def test_transaction_rolls_back_on_error(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.db.transaction() as tx:
                repo.add_folder(tx, "/tx/rolled-back")
                raise RuntimeError("boom")
        self.assertIsNone(repo.get_folder_by_path(self.conn, "/tx/rolled-back"))

    def test_transaction_yields_fresh_connection_closed_afterwards(self) -> None:
        with self.db.transaction() as tx:
            repo.add_folder(tx, "/tx/closed")
        with self.assertRaises(sqlite3.ProgrammingError):
            tx.execute("SELECT 1")  # connection is closed once the block ends


class VectorHelperTests(_TempDbTestCase):
    """vec_to_blob / blob_to_vec roundtrips."""

    def test_vec_blob_roundtrip_exact(self) -> None:
        vec = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        blob = vec_to_blob(vec)
        self.assertIsInstance(blob, bytes)
        self.assertEqual(len(blob), 8 * 4)
        out = blob_to_vec(blob)
        np.testing.assert_array_equal(out, vec)
        self.assertEqual(out.dtype, np.dtype("<f4"))
        self.assertEqual(out.shape, (8,))

    def test_vec_to_blob_accepts_lists_and_other_byteorders(self) -> None:
        self.assertEqual(vec_to_blob([1.0, -2.5]),
                         vec_to_blob(np.array([1.0, -2.5], dtype=np.float32)))
        big = np.array([1.5, -2.5], dtype=">f4")  # big-endian input
        np.testing.assert_array_equal(blob_to_vec(vec_to_blob(big)),
                                      big.astype("<f4"))

    def test_blob_to_vec_returns_writable_copy(self) -> None:
        out = blob_to_vec(vec_to_blob(np.ones(3, dtype=np.float32)))
        out[0] = 5.0  # must not raise: np.frombuffer alone is read-only
        self.assertEqual(float(out[0]), 5.0)

    def test_empty_vector_roundtrip(self) -> None:
        self.assertEqual(vec_to_blob(np.array([], dtype=np.float32)), b"")
        self.assertEqual(blob_to_vec(b"").size, 0)


class FolderTests(_TempDbTestCase):
    """add_folder / list_folders / get_folder_by_path."""

    def test_add_folder_returns_id_and_is_idempotent(self) -> None:
        fid1 = repo.add_folder(self.conn, "/music")
        fid2 = repo.add_folder(self.conn, "/music")
        self.assertIsInstance(fid1, int)
        self.assertEqual(fid1, fid2)
        # a trailing slash normalizes onto the same folder row
        self.assertEqual(repo.add_folder(self.conn, "/music/"), fid1)
        self.assertEqual([r["id"] for r in repo.list_folders(self.conn)], [fid1])

    def test_list_folders_ordered_and_get_folder_by_path(self) -> None:
        repo.add_folder(self.conn, "/b-music")
        repo.add_folder(self.conn, "/a-music")
        rows = repo.list_folders(self.conn)
        self.assertEqual([r["path"] for r in rows], ["/a-music", "/b-music"])
        row = repo.get_folder_by_path(self.conn, "/b-music")
        self.assertIsNotNone(row)
        self.assertEqual(row["path"], "/b-music")
        self.assertIn("added_at", row.keys())

    def test_get_folder_by_path_missing_returns_none(self) -> None:
        self.assertIsNone(repo.get_folder_by_path(self.conn, "/nope"))


class TrackTests(_TempDbTestCase):
    """upsert_track, listing, lookups, status and description."""

    def test_upsert_track_inserts_then_updates_same_row(self) -> None:
        fid = repo.add_folder(self.conn, "/music")
        path = "/music/a.mp3"
        tid1 = repo.upsert_track(self.conn, fid, path, {
            "filename": "a.mp3", "size_bytes": 123, "duration_sec": 30.0,
            "title": "A", "artist": "X",
        })
        tid2 = repo.upsert_track(self.conn, fid, path, {
            "duration_sec": 42.5, "bitrate_kbps": 320.0,
        })
        self.assertEqual(tid1, tid2)  # updated, not duplicated
        self.assertEqual(len(repo.list_tracks(self.conn)), 1)
        row = repo.get_track(self.conn, tid1)
        self.assertEqual(row["duration_sec"], 42.5)
        self.assertEqual(row["bitrate_kbps"], 320.0)
        self.assertEqual(row["title"], "A")  # untouched columns survive
        self.assertEqual(row["artist"], "X")
        self.assertEqual(row["size_bytes"], 123)
        self.assertEqual(row["filename"], "a.mp3")
        self.assertEqual(row["status"], "new")
        # re-pointing the track at another folder updates, never duplicates
        fid2 = repo.add_folder(self.conn, "/elsewhere")
        self.assertEqual(repo.upsert_track(self.conn, fid2, path, {}), tid1)
        self.assertEqual(repo.get_track(self.conn, tid1)["folder_id"], fid2)

    def test_upsert_track_filename_defaults_to_basename(self) -> None:
        _, tid = self.add_track("/music", "Song One.flac")
        row = repo.get_track(self.conn, tid)
        self.assertEqual(row["filename"], "Song One.flac")
        self.assertIsNone(row["title"])

    def test_list_tracks_ordered_and_filtered(self) -> None:
        f1 = repo.add_folder(self.conn, "/music")
        f2 = repo.add_folder(self.conn, "/audiobooks")
        repo.upsert_track(self.conn, f1, "/music/b.flac", {})
        repo.upsert_track(self.conn, f1, "/music/a.mp3", {})
        repo.upsert_track(self.conn, f2, "/audiobooks/x.mp3", {})
        self.assertEqual([r["path"] for r in repo.list_tracks(self.conn)],
                         ["/audiobooks/x.mp3", "/music/a.mp3", "/music/b.flac"])
        self.assertEqual([r["path"] for r in repo.list_tracks(self.conn, f1)],
                         ["/music/a.mp3", "/music/b.flac"])
        self.assertEqual(repo.list_tracks(self.conn, 424242), [])

    def test_get_track_and_get_track_by_path(self) -> None:
        _, tid = self.add_track("/music", "a.mp3")
        self.assertIsNotNone(repo.get_track(self.conn, tid))
        self.assertIsNone(repo.get_track(self.conn, 999999))
        row = repo.get_track_by_path(self.conn, "/music/a.mp3")
        self.assertEqual(row["id"], tid)
        self.assertIsNone(repo.get_track_by_path(self.conn, "/music/zzz.mp3"))

    def test_set_track_status_and_description(self) -> None:
        _, tid = self.add_track()
        repo.set_track_status(self.conn, tid, "analyzing")
        self.assertEqual(repo.get_track(self.conn, tid)["status"], "analyzing")
        repo.set_track_status(self.conn, tid, "analyzed", "3 chunks")
        row = repo.get_track(self.conn, tid)
        self.assertEqual(row["status"], "analyzed")
        self.assertEqual(row["status_message"], "3 chunks")
        self.assertIsNotNone(row["last_analyzed_at"])
        repo.set_track_status(self.conn, tid, "error")  # message omitted -> cleared
        row = repo.get_track(self.conn, tid)
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["status_message"])
        repo.set_track_description(self.conn, tid, "upbeat rock with piano")
        self.assertEqual(repo.get_track(self.conn, tid)["description"],
                         "upbeat rock with piano")


class ChunkTests(_TempDbTestCase):
    """replace_chunks / get_chunks / get_chunks_for_tracks."""

    def test_replace_chunks_inserts_ordered_rows(self) -> None:
        _, tid = self.add_track()
        ids = repo.replace_chunks(self.conn, tid,
                                  [(0, 0.0, 20.0), (1, 10.0, 30.0), (2, 20.0, 40.0)])
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)
        chunks = repo.get_chunks(self.conn, tid)
        self.assertEqual([c["idx"] for c in chunks], [0, 1, 2])
        self.assertEqual([c["id"] for c in chunks], ids)
        self.assertAlmostEqual(chunks[1]["start_sec"], 10.0)
        self.assertAlmostEqual(chunks[1]["end_sec"], 30.0)

    def test_replace_chunks_twice_removes_old_rows_and_cascades(self) -> None:
        _, tid = self.add_track()
        ids1 = repo.replace_chunks(self.conn, tid, [(0, 0.0, 20.0), (1, 10.0, 30.0)])
        repo.add_chunk_embedding(self.conn, ids1[0], "clap",
                                 np.ones(4, dtype=np.float32))
        ids2 = repo.replace_chunks(
            self.conn, tid, [(0, 0.0, 20.0), (1, 10.0, 25.0), (2, 15.0, 35.0)])
        chunks = repo.get_chunks(self.conn, tid)
        self.assertEqual([c["idx"] for c in chunks], [0, 1, 2])
        self.assertNotIn(ids1[0], ids2)  # fresh ids after the replace
        self.assertEqual(self.count("chunks", "id = ?", (ids1[0],)), 0)
        # the embedding attached to a deleted chunk cascaded away too
        self.assertEqual(self.count(
            "embeddings",
            "chunk_id IN (SELECT id FROM chunks WHERE track_id = ?)", (tid,)), 0)

    def test_replace_chunks_empty_clears_track(self) -> None:
        _, tid = self.add_track()
        repo.replace_chunks(self.conn, tid, [(0, 0.0, 5.0)])
        self.assertEqual(repo.replace_chunks(self.conn, tid, []), [])
        self.assertEqual(repo.get_chunks(self.conn, tid), [])

    def test_get_chunks_for_tracks(self) -> None:
        _, t1 = self.add_track("/m1", "a.mp3")
        _, t2 = self.add_track("/m2", "b.mp3")
        repo.replace_chunks(self.conn, t1, [(0, 0.0, 10.0), (1, 5.0, 15.0)])
        repo.replace_chunks(self.conn, t2, [(0, 0.0, 10.0)])
        rows = repo.get_chunks_for_tracks(self.conn, [t2, t1])
        self.assertEqual([(r["track_id"], r["idx"]) for r in rows],
                         sorted([(t1, 0), (t1, 1), (t2, 0)]))
        self.assertEqual(repo.get_chunks_for_tracks(self.conn, []), [])


class ChunkEmbeddingTests(_TempDbTestCase):
    """add_chunk_embedding / get_chunk_embeddings (blob <-> vec roundtrip)."""

    def test_add_chunk_embedding_roundtrip_and_norm(self) -> None:
        _, tid = self.add_track()
        (cid,) = repo.replace_chunks(self.conn, tid, [(0, 0.0, 20.0)])
        vec = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        repo.add_chunk_embedding(self.conn, cid, "clap", vec)
        rows = repo.get_chunk_embeddings(self.conn, cid)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        np.testing.assert_array_equal(row["vec"], vec)
        self.assertEqual(row["vec"].dtype, np.dtype("<f4"))
        self.assertEqual(row["vec"].shape, (8,))
        self.assertEqual(row["model"], "clap")
        self.assertEqual(row["dim"], 8)
        self.assertAlmostEqual(row["norm"], float(np.linalg.norm(vec)), places=5)
        self.assertEqual(row["vector"], vec_to_blob(vec))

    def test_add_chunk_embedding_upserts_same_model(self) -> None:
        _, tid = self.add_track()
        (cid,) = repo.replace_chunks(self.conn, tid, [(0, 0.0, 20.0)])
        v1 = np.ones(4, dtype=np.float32)
        v2 = np.full(4, 2.0, dtype=np.float32)
        repo.add_chunk_embedding(self.conn, cid, "clap", v1)
        repo.add_chunk_embedding(self.conn, cid, "clap", v2)
        rows = repo.get_chunk_embeddings(self.conn, cid)
        self.assertEqual(len(rows), 1)  # replaced, not duplicated
        np.testing.assert_array_equal(rows[0]["vec"], v2)
        self.assertAlmostEqual(rows[0]["norm"], float(np.linalg.norm(v2)), places=5)

    def test_get_chunk_embeddings_model_filter(self) -> None:
        _, tid = self.add_track()
        (cid,) = repo.replace_chunks(self.conn, tid, [(0, 0.0, 20.0)])
        repo.add_chunk_embedding(self.conn, cid, "mert", np.zeros(6, dtype=np.float32))
        repo.add_chunk_embedding(self.conn, cid, "clap", np.ones(5, dtype=np.float32))
        clap = repo.get_chunk_embeddings(self.conn, cid, model="clap")
        self.assertEqual([r["model"] for r in clap], ["clap"])
        self.assertEqual(clap[0]["dim"], 5)
        self.assertEqual(
            [r["model"] for r in repo.get_chunk_embeddings(self.conn, cid, model="openl3")],
            [])
        both = repo.get_chunk_embeddings(self.conn, cid)
        self.assertEqual([r["model"] for r in both], ["clap", "mert"])  # stable order

    def test_embedding_rows_behave_like_rows(self) -> None:
        _, tid = self.add_track()
        (cid,) = repo.replace_chunks(self.conn, tid, [(0, 0.0, 20.0)])
        repo.add_chunk_embedding(self.conn, cid, "clap", np.ones(3, dtype=np.float32))
        row = repo.get_chunk_embeddings(self.conn, cid)[0]
        self.assertIn("vec", row.keys())
        self.assertIn("model", row)
        self.assertEqual(row[0], row["id"])  # positional access still works
        self.assertEqual(len(row), len(row.keys()))
        self.assertIn("model", dict(row))
        self.assertEqual(row.get("nope", "dflt"), "dflt")


class TrackEmbeddingTests(_TempDbTestCase):
    """set/get track embeddings (centroids & ollama text embeddings)."""

    def test_set_and_get_track_embedding_upsert(self) -> None:
        _, t1 = self.add_track("/m1", "a.mp3")
        v1 = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        v2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        repo.set_track_embedding(self.conn, t1, "clap", v1)
        repo.set_track_embedding(self.conn, t1, "clap", v2)  # upsert, no duplicate
        np.testing.assert_array_equal(repo.get_track_embedding(self.conn, t1, "clap"), v2)
        self.assertIsNone(repo.get_track_embedding(self.conn, t1, "openl3"))
        self.assertIsNone(repo.get_track_embedding(self.conn, 999999, "clap"))

    def test_get_track_embeddings_filters(self) -> None:
        _, t1 = self.add_track("/m1", "a.mp3")
        _, t2 = self.add_track("/m2", "b.mp3")
        v1 = np.array([1.0, 2.0], dtype=np.float32)
        v2 = np.array([3.0, 4.0], dtype=np.float32)
        v3 = np.array([5.0, 6.0], dtype=np.float32)
        repo.set_track_embedding(self.conn, t1, "clap", v1)
        repo.set_track_embedding(self.conn, t2, "clap", v2)
        repo.set_track_embedding(self.conn, t1, "mert", v3)
        all_clap = repo.get_track_embeddings(self.conn, "clap")
        self.assertEqual(set(all_clap), {t1, t2})
        np.testing.assert_array_equal(all_clap[t1], v1)
        np.testing.assert_array_equal(all_clap[t2], v2)
        only_t2 = repo.get_track_embeddings(self.conn, "clap", track_ids=[t2])
        self.assertEqual(set(only_t2), {t2})
        np.testing.assert_array_equal(only_t2[t2], v2)
        self.assertEqual(repo.get_track_embeddings(self.conn, "clap", track_ids=[]), {})
        self.assertEqual(repo.get_track_embeddings(self.conn, "openl3"), {})

    def test_tracks_with_embeddings(self) -> None:
        _, t1 = self.add_track("/m1", "a.mp3")
        _, t2 = self.add_track("/m2", "b.mp3")
        repo.set_track_embedding(self.conn, t2, "clap", np.ones(2, dtype=np.float32))
        repo.set_track_embedding(self.conn, t1, "clap", np.ones(2, dtype=np.float32))
        repo.set_track_embedding(self.conn, t1, "mert", np.ones(2, dtype=np.float32))
        self.assertEqual(repo.tracks_with_embeddings(self.conn, "clap"),
                         sorted([t1, t2]))
        self.assertEqual(repo.tracks_with_embeddings(self.conn, "mert"), [t1])
        self.assertEqual(repo.tracks_with_embeddings(self.conn, "openl3"), [])


class TagTests(_TempDbTestCase):
    """add_chunk_tags / get_chunk_tags / get_track_tags."""

    def test_get_chunk_tags_ordering_and_filter(self) -> None:
        _, tid = self.add_track()
        c0, c1 = repo.replace_chunks(self.conn, tid,
                                     [(0, 0.0, 20.0), (1, 10.0, 30.0)])
        repo.add_chunk_tags(self.conn, c0, "clap", [("pop", 0.91), ("rock", 0.05)])
        repo.add_chunk_tags(self.conn, c1, "clap", [("jazz", 0.5)])
        repo.add_chunk_tags(self.conn, c1, "mert", [("ambient", 0.7)])
        tags0 = repo.get_chunk_tags(self.conn, c0)
        self.assertEqual([t["text"] for t in tags0], ["pop", "rock"])  # score DESC
        self.assertAlmostEqual(tags0[0]["score"], 0.91, places=6)
        self.assertEqual(
            [t["text"] for t in repo.get_chunk_tags(self.conn, c0, model="mert")],
            [])
        self.assertEqual(
            [t["text"] for t in repo.get_chunk_tags(self.conn, c1, model="mert")],
            ["ambient"])
        self.assertEqual(len(repo.get_chunk_tags(self.conn, c0, model="openl3")), 0)
        # model=None mixes models of the chunk, best score first
        self.assertEqual([t["text"] for t in repo.get_chunk_tags(self.conn, c1)],
                         ["ambient", "jazz"])

    def test_add_chunk_tags_repeats_update_scores(self) -> None:
        _, tid = self.add_track()
        (c0,) = repo.replace_chunks(self.conn, tid, [(0, 0.0, 20.0)])
        repo.add_chunk_tags(self.conn, c0, "clap", [("pop", 0.91), ("rock", 0.05)])
        repo.add_chunk_tags(self.conn, c0, "clap", [("pop", 0.99)])
        tags = repo.get_chunk_tags(self.conn, c0)
        self.assertEqual(len(tags), 2)  # no duplicate row for the same tag text
        self.assertEqual([t["text"] for t in tags], ["pop", "rock"])
        self.assertAlmostEqual(tags[0]["score"], 0.99, places=6)

    def test_get_track_tags_joined_and_ordered(self) -> None:
        _, tid = self.add_track()
        repo.replace_chunks(self.conn, tid, [(0, 0.0, 20.0), (1, 10.0, 30.0)])
        c0, c1 = [c["id"] for c in repo.get_chunks(self.conn, tid)]
        repo.add_chunk_tags(self.conn, c0, "clap", [("pop", 0.91), ("rock", 0.05)])
        repo.add_chunk_tags(self.conn, c1, "clap", [("jazz", 0.5), ("pop", 0.33)])
        repo.add_chunk_tags(self.conn, c1, "mert", [("ambient", 0.7)])
        tags = repo.get_track_tags(self.conn, tid)
        # model=None: every model's tags across all chunks, best score first
        self.assertEqual([(t["text"], t["chunk_idx"]) for t in tags],
                         [("pop", 0), ("ambient", 1), ("jazz", 1),
                           ("pop", 1), ("rock", 0)])
        self.assertAlmostEqual(tags[0]["score"], 0.91, places=6)
        self.assertEqual(len(repo.get_track_tags(self.conn, tid, model="clap")), 4)
        self.assertEqual(
            [t["text"] for t in repo.get_track_tags(self.conn, tid, model="clap")],
            ["pop", "jazz", "pop", "rock"])
        self.assertEqual(
            [t["text"] for t in repo.get_track_tags(self.conn, tid, model="mert")],
            ["ambient"])
        self.assertEqual(repo.get_track_tags(self.conn, 999999), [])


class TrackEmbeddingCountTests(_TempDbTestCase):
    """Per-track per-model chunk-embedding counts for the file tree columns."""

    def test_get_track_embedding_counts(self) -> None:
        _, tid = self.add_track()
        c0, c1 = repo.replace_chunks(self.conn, tid,
                                     [(0, 0.0, 20.0), (1, 10.0, 30.0)])
        self.assertEqual(repo.get_track_embedding_counts(self.conn, tid), {})
        vec = np.ones(4, dtype=np.float32)
        repo.add_chunk_embedding(self.conn, c0, "clap", vec)
        repo.add_chunk_embedding(self.conn, c0, "mert", vec)
        repo.add_chunk_embedding(self.conn, c1, "clap", vec)
        counts = repo.get_track_embedding_counts(self.conn, tid)
        self.assertEqual(counts, {"clap": 2, "mert": 1})
        self.assertEqual(repo.get_track_embedding_counts(self.conn, 999), {})

    def test_get_all_track_embedding_counts(self) -> None:
        _, t1 = self.add_track("/m1", "a.mp3")
        _, t2 = self.add_track("/m2", "b.mp3")
        (c1,) = repo.replace_chunks(self.conn, t1, [(0, 0.0, 20.0)])
        (c2,) = repo.replace_chunks(self.conn, t2, [(0, 0.0, 20.0)])
        vec = np.ones(4, dtype=np.float32)
        repo.add_chunk_embedding(self.conn, c1, "clap", vec)
        repo.add_chunk_embedding(self.conn, c2, "fft", vec)
        repo.add_chunk_embedding(self.conn, c2, "fft", vec)  # upsert, stays 1
        all_counts = repo.get_all_track_embedding_counts(self.conn)
        self.assertEqual(all_counts, {t1: {"clap": 1}, t2: {"fft": 1}})
        # tracks without chunks are simply absent
        _, t3 = self.add_track("/m3", "c.mp3")
        self.assertNotIn(t3, repo.get_all_track_embedding_counts(self.conn))


class PlaylistTests(_TempDbTestCase):
    """create_playlist / add_playlist_items / get_playlist_items / delete."""

    def test_playlist_create_add_items_get_delete(self) -> None:
        _, t1 = self.add_track("/m1", "a.mp3", {"artist": "Alpha", "title": "One"})
        _, t2 = self.add_track("/m2", "b.mp3", {"artist": "Beta", "title": "Two"})
        pid = repo.create_playlist(self.conn, "Roadtrip",
                                   seed_track_id=t1, method="clap")
        pl = repo.get_playlist(self.conn, pid)
        self.assertEqual(pl["name"], "Roadtrip")
        self.assertEqual(pl["seed_track_id"], t1)
        self.assertEqual(pl["method"], "clap")

        repo.add_playlist_items(self.conn, pid, [(t2, 0.95), (t1, 0.8)])
        items = repo.get_playlist_items(self.conn, pid)
        self.assertEqual([i["position"] for i in items], [1, 2])
        self.assertEqual([i["track_id"] for i in items], [t2, t1])
        self.assertEqual([i["similarity"] for i in items], [0.95, 0.8])
        self.assertEqual([i["filename"] for i in items], ["b.mp3", "a.mp3"])
        self.assertEqual(items[0]["artist"], "Beta")
        self.assertEqual(items[1]["title"], "One")
        self.assertEqual(items[0]["path"], "/m2/b.mp3")

        repo.delete_playlist(self.conn, pid)
        self.assertIsNone(repo.get_playlist(self.conn, pid))
        self.assertEqual(repo.get_playlist_items(self.conn, pid), [])
        self.assertEqual(repo.list_playlists(self.conn), [])

    def test_add_playlist_items_replaces_previous(self) -> None:
        _, t1 = self.add_track("/m1", "a.mp3")
        _, t2 = self.add_track("/m2", "b.mp3")
        pid = repo.create_playlist(self.conn, "Mix")
        repo.add_playlist_items(self.conn, pid, [(t1, 0.9), (t2, 0.8)])
        repo.add_playlist_items(self.conn, pid, [(t2, 0.7)])  # re-add renumbers 1..n
        items = repo.get_playlist_items(self.conn, pid)
        self.assertEqual([(i["position"], i["track_id"], i["similarity"])
                          for i in items], [(1, t2, 0.7)])
        repo.add_playlist_items(self.conn, pid, [(t1, None)])  # None similarity ok
        self.assertIsNone(repo.get_playlist_items(self.conn, pid)[0]["similarity"])

    def test_list_playlists_and_missing_playlist(self) -> None:
        self.assertIsNone(repo.get_playlist(self.conn, 1))
        repo.create_playlist(self.conn, "First")
        p2 = repo.create_playlist(self.conn, "Second", method="auto")
        rows = repo.list_playlists(self.conn)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["name"] for r in rows}, {"First", "Second"})
        self.assertIsNone(repo.get_playlist(self.conn, 424242))
        self.assertEqual(repo.get_playlist_items(self.conn, p2), [])


class SettingsTests(_TempDbTestCase):
    """settings key/value store and ollama model list roundtrips."""

    def test_settings_roundtrip_and_defaults(self) -> None:
        self.assertIsNone(repo.get_setting(self.conn, "chunk_seconds"))
        self.assertEqual(repo.get_setting(self.conn, "chunk_seconds", "20.0"), "20.0")
        repo.set_setting(self.conn, "chunk_seconds", "30.0")
        self.assertEqual(repo.get_setting(self.conn, "chunk_seconds"), "30.0")
        repo.set_setting(self.conn, "chunk_seconds", "45.5")  # overwrite, no dup row
        self.assertEqual(repo.get_setting(self.conn, "chunk_seconds"), "45.5")
        self.assertEqual(self.count("settings", "key = 'chunk_seconds'"), 1)

    def test_ollama_models_roundtrip(self) -> None:
        self.assertEqual(repo.get_ollama_models(self.conn), [])
        models = ["nomic-embed-text", "bge-m3"]
        repo.save_ollama_models(self.conn, models)
        self.assertEqual(repo.get_ollama_models(self.conn), models)
        stored = repo.get_setting(self.conn, "ollama_embedding_models")
        self.assertEqual(json.loads(stored), models)  # persisted as a JSON list
        repo.save_ollama_models(self.conn, [])
        self.assertEqual(repo.get_ollama_models(self.conn), [])

    def test_ollama_models_survives_corrupt_json(self) -> None:
        repo.set_setting(self.conn, "ollama_embedding_models", "{not json")
        self.assertEqual(repo.get_ollama_models(self.conn), [])
        repo.set_setting(self.conn, "ollama_embedding_models", "42")  # not a list
        self.assertEqual(repo.get_ollama_models(self.conn), [])


class CleanupTests(_TempDbTestCase):
    """delete_tracks_missing and remove_folder cascade behaviour."""

    def test_delete_tracks_missing(self) -> None:
        f1 = repo.add_folder(self.conn, "/music")
        keep_a = repo.upsert_track(self.conn, f1, "/music/a.mp3", {})
        gone_b = repo.upsert_track(self.conn, f1, "/music/b.mp3", {})
        keep_c = repo.upsert_track(self.conn, f1, "/music/c.mp3", {})
        repo.replace_chunks(self.conn, gone_b, [(0, 0.0, 10.0)])
        f2 = repo.add_folder(self.conn, "/other")
        other_d = repo.upsert_track(self.conn, f2, "/other/d.mp3", {})

        removed = repo.delete_tracks_missing(
            self.conn, f1, {"/music/a.mp3", "/music/c.mp3"})
        self.assertEqual(removed, 1)
        self.assertIsNone(repo.get_track(self.conn, gone_b))
        self.assertIsNotNone(repo.get_track(self.conn, keep_a))
        self.assertIsNotNone(repo.get_track(self.conn, keep_c))
        self.assertIsNotNone(repo.get_track(self.conn, other_d))  # other folder safe
        self.assertEqual(repo.get_chunks(self.conn, gone_b), [])  # chunks cascaded

        self.assertEqual(repo.delete_tracks_missing(self.conn, f1, set()), 2)
        self.assertEqual(repo.list_tracks(self.conn, f1), [])
        self.assertIsNotNone(repo.get_track(self.conn, other_d))

    def test_remove_folder_cascades_everything(self) -> None:
        fid, tid = self.add_track("/music", "a.mp3")
        (cid,) = repo.replace_chunks(self.conn, tid, [(0, 0.0, 10.0)])
        repo.add_chunk_embedding(self.conn, cid, "clap", np.ones(4, dtype=np.float32))
        repo.add_chunk_tags(self.conn, cid, "clap", [("pop", 0.9)])
        repo.set_track_embedding(self.conn, tid, "clap", np.ones(4, dtype=np.float32))

        repo.remove_folder(self.conn, fid)
        repo.remove_folder(self.conn, 999999)  # unknown id: harmless no-op
        self.assertEqual(repo.list_folders(self.conn), [])
        self.assertIsNone(repo.get_track(self.conn, tid))
        for table in ("chunks", "embeddings", "chunk_tags", "track_embeddings"):
            self.assertEqual(self.count(table), 0, table)


if __name__ == "__main__":
    unittest.main()
