"""Tests for app.similarity.pareto (Pareto surface + chunk-level search).

Run:
    cd /Users/apple/Documents/HOLOSMART && .venv/bin/python -m unittest tests.test_pareto -v

Uses a real temp-file ``app.db.database.Database`` plus the real ``app.db.repo``
functions with small synthetic numpy vectors.  No Qt, no torch, no network —
and deliberately NO ``track_embeddings`` rows are ever written, which doubles
as proof that the Pareto search works purely off ``chunks``/``embeddings``.

Hand-computed fixture (all 2-D vectors unless noted):

* seed — clap [[1,0],[0,1],[1,1]], mert [[1,1],[0,1],[1,0]].
  Both centroids are (0.70711, 0.70711), so the chunk objectives are
  c0=[0.70711, 1.0], c1=[0.70711, 0.70711], c2=[1.0, 0.70711]:
  c1 is dominated by both neighbours → Pareto surface = {c0, c2}.
* close  — clap [[1,1],[1,0]]: every surface chunk finds a perfect match → 1.0
* mert_only — mert [[1,1],[1,0]]: same, via the mert model → 1.0
* tricky — clap [[1,0],[0,1]]: matches the seed's NON-surface chunk
  perfectly; surface-based score = (1.0 + 0.70711) / 2 = 0.853553
* both   — clap [[1,0]] + mert [[1,1]]: mean of the two per-model means of
  0.853553 → 0.853553 (ties with tricky; track id breaks the tie)
* dim3   — clap [[1,0,0]]: dimension mismatch → every pair scores 0.0
* far    — clap [[-1,-1]]: ((-0.70711) + (-1.0)) / 2 = -0.853553
* naked  — no chunks at all;  hollow — chunks but no embeddings;
  l3 — a single chunk with an ``openl3`` embedding nobody else shares.

Expected search order: close, mert_only, tricky, both, dim3, far
(1.0 ties between close/mert_only and 0.853553 ties between tricky/both are
broken by ascending track id, which follows the insertion order above).
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.similarity.pareto import (  # noqa: E402
    chunk_objectives,
    pareto_chunk_ids,
    pareto_similar_tracks,
    pareto_surface,
)
from app.similarity.search import similar_tracks  # noqa: E402

INV_SQRT2 = 1.0 / math.sqrt(2.0)
MEAN_BEST = (1.0 + INV_SQRT2) / 2.0  # tricky/both/dim-independent value


class ParetoLibrary:
    """Tiny synthetic library exercising the Pareto search end to end."""

    #: key, filename, artist, title, chunk count, {model: [chunk vectors]}
    PLAN: list[tuple[str, str, str, str, int, dict[str, list[list[float]]]]] = [
        ("seed", "seed_song.wav", "Alpha", "Seed Song", 3, {
            "clap": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "mert": [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        }),
        ("close", "close_song.wav", "Bravo", "Close Song", 2,
         {"clap": [[1.0, 1.0], [1.0, 0.0]]}),
        ("tricky", "tricky_song.wav", "Charlie", "Tricky Song", 2,
         {"clap": [[1.0, 0.0], [0.0, 1.0]]}),
        ("dim3", "dim3_song.wav", "Delta", "Dim3 Song", 1,
         {"clap": [[1.0, 0.0, 0.0]]}),
        ("mert_only", "mert_song.wav", "Echo", "Mert Song", 2,
         {"mert": [[1.0, 1.0], [1.0, 0.0]]}),
        ("both", "both_song.wav", "Foxtrot", "Both Song", 1, {
            "clap": [[1.0, 0.0]],
            "mert": [[1.0, 1.0]],
        }),
        ("far", "far_song.wav", "Golf", "Far Song", 1,
         {"clap": [[-1.0, -1.0]]}),
        ("naked", "naked_song.wav", "Hotel", "Naked Song", 0, {}),
        ("hollow", "hollow_song.wav", "India", "Hollow Song", 2, {}),
        ("l3", "l3_song.wav", "Juliet", "L3 Song", 1,
         {"openl3": [[1.0, 0.0, 0.0, 0.0]]}),
    ]

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.music_dir = self.root / "music"
        self.db = Database(self.root / "library.db")
        self.conn = self.db.connect()
        folder_id = repo.add_folder(self.conn, str(self.music_dir))
        self.ids: dict[str, int] = {}
        for key, filename, artist, title, chunk_count, embeddings in self.PLAN:
            path = str(self.music_dir / filename)
            track_id = repo.upsert_track(self.conn, folder_id, path, {
                "filename": filename, "artist": artist, "title": title,
                "duration_sec": 60.0,
            })
            chunk_ids = repo.replace_chunks(self.conn, track_id, [
                (i, i * 20.0, i * 20.0 + 20.0) for i in range(chunk_count)
            ])
            for model, vectors in embeddings.items():
                for chunk_id, vector in zip(chunk_ids, vectors):
                    repo.add_chunk_embedding(
                        self.conn, chunk_id, model,
                        np.asarray(vector, dtype=np.float32))
            self.ids[key] = track_id
        self.conn.commit()

    @property
    def seed(self) -> int:
        return self.ids["seed"]

    def expected_surface(self) -> list[int]:
        """Seed surface chunk ids in ``idx`` order (hand computed: c0, c2)."""
        chunks = repo.get_chunks(self.conn, self.seed)
        return [int(chunks[0]["id"]), int(chunks[2]["id"])]


class ParetoLibraryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="holosmart-pareto-")
        self.addCleanup(self._tmp.cleanup)
        self.lib = ParetoLibrary(Path(self._tmp.name))
        self.addCleanup(self.lib.conn.close)


# ---------------------------------------------------------------- pure logic
class ParetoSurfaceTests(unittest.TestCase):
    def test_dominated_points_excluded(self):
        points = {1: [3.0, 1.0], 2: [1.0, 3.0], 3: [2.0, 2.0], 4: [1.0, 1.0]}
        self.assertEqual(pareto_surface(points), [1, 2, 3])

    def test_identical_points_both_kept(self):
        points = {1: [3.0, 1.0], 5: [3.0, 1.0], 2: [1.0, 3.0], 4: [1.0, 1.0]}
        self.assertEqual(pareto_surface(points), [1, 2, 5])

    def test_tie_on_one_dimension_still_dominates(self):
        self.assertEqual(pareto_surface({1: [2.0, 1.0], 2: [2.0, 0.0]}), [1])
        self.assertEqual(pareto_surface({1: [2.0, 1.0], 2: [2.0, 1.0]}), [1, 2])

    def test_empty_input(self):
        self.assertEqual(pareto_surface({}), [])

    def test_single_point(self):
        self.assertEqual(pareto_surface({7: [0.5, -0.25]}), [7])

    def test_integer_keys_sort_numerically(self):
        self.assertEqual(pareto_surface({10: [1.0], 2: [1.0]}), [2, 10])

    def test_accepts_numpy_vectors(self):
        points = {1: np.asarray([1.0, 1.0]), 2: np.asarray([0.5, 0.5])}
        self.assertEqual(pareto_surface(points), [1])

    def test_negative_objectives(self):
        # 1 and 2 are identical (both stay); 3 beats them in dim 0 but loses
        # in dim 1, so nothing dominates anything here.
        points = {1: [-1.0, -0.5], 2: [-1.0, -0.5], 3: [-0.9, -0.9]}
        self.assertEqual(pareto_surface(points), [1, 2, 3])

    def test_deterministic_across_calls(self):
        points = {1: [3.0, 1.0], 2: [1.0, 3.0], 3: [2.0, 2.0], 4: [1.0, 1.0]}
        self.assertEqual(pareto_surface(points), pareto_surface(points))


# ---------------------------------------------------------- chunk objectives
class ChunkObjectivesTests(ParetoLibraryTestCase):
    def test_hand_computed_objectives_and_missing_model_rule(self):
        conn = self.lib.conn
        obj_id = repo.upsert_track(conn, 1, str(self.lib.music_dir / "obj.wav"),
                                   {"filename": "obj.wav"})
        chunk_ids = repo.replace_chunks(conn, obj_id, [(0, 0.0, 20.0), (1, 20.0, 40.0)])
        # Insert mert first on purpose: dimensions must still be sorted by
        # model NAME ("clap" before "mert"), not insertion order.
        repo.add_chunk_embedding(conn, chunk_ids[0], "mert",
                                 np.asarray([0.6, 0.8], dtype=np.float32))
        repo.add_chunk_embedding(conn, chunk_ids[0], "clap",
                                 np.asarray([1.0, 0.0], dtype=np.float32))
        repo.add_chunk_embedding(conn, chunk_ids[1], "clap",
                                 np.asarray([0.0, 1.0], dtype=np.float32))
        conn.commit()

        objectives = chunk_objectives(conn, obj_id)
        self.assertEqual(sorted(objectives), sorted(chunk_ids))
        # clap centroid = normalize([0.5, 0.5]) = (0.70711, 0.70711);
        # mert centroid = [0.6, 0.8] (already unit length).
        expected = {
            chunk_ids[0]: [INV_SQRT2, 1.0],   # both embeddings present
            chunk_ids[1]: [INV_SQRT2, -1.0],  # missing mert -> -1.0
        }
        for chunk_id, values in expected.items():
            self.assertEqual(len(objectives[chunk_id]), 2)
            for got, want in zip(objectives[chunk_id], values):
                self.assertAlmostEqual(got, want, places=6)
        # chunk1 is dominated by chunk0 -> surface holds only chunk0
        self.assertEqual(pareto_chunk_ids(conn, obj_id), [chunk_ids[0]])

    def test_chunk_without_any_embedding_gets_minus_one_everywhere(self):
        conn = self.lib.conn
        obj_id = repo.upsert_track(conn, 1, str(self.lib.music_dir / "obj2.wav"),
                                   {"filename": "obj2.wav"})
        chunk_ids = repo.replace_chunks(conn, obj_id, [
            (0, 0.0, 20.0), (1, 20.0, 40.0), (2, 40.0, 60.0)])
        repo.add_chunk_embedding(conn, chunk_ids[0], "clap",
                                 np.asarray([1.0, 0.0], dtype=np.float32))
        conn.commit()

        objectives = chunk_objectives(conn, obj_id)
        self.assertEqual(objectives[chunk_ids[0]], [1.0])  # centroid is itself
        self.assertEqual(objectives[chunk_ids[1]], [-1.0])
        self.assertEqual(objectives[chunk_ids[2]], [-1.0])
        self.assertEqual(pareto_chunk_ids(conn, obj_id), [chunk_ids[0]])

    def test_dimensions_only_cover_models_present_on_this_track(self):
        objectives = chunk_objectives(self.lib.conn, self.lib.seed)
        # "openl3" exists in the library (track l3) but not on the seed's
        # chunks, so every seed objective vector has exactly 2 dimensions.
        for values in objectives.values():
            self.assertEqual(len(values), 2)
        surface = self.lib.expected_surface()
        for chunk_id, want in zip(surface, ([INV_SQRT2, 1.0], [1.0, INV_SQRT2])):
            for got, expected in zip(objectives[chunk_id], want):
                self.assertAlmostEqual(got, expected, places=6)

    def test_chunks_without_embeddings_yield_empty_objective_vectors(self):
        conn = self.lib.conn
        hollow = self.lib.ids["hollow"]
        objectives = chunk_objectives(conn, hollow)
        self.assertEqual(len(objectives), 2)  # two chunks, zero models
        for values in objectives.values():
            self.assertEqual(values, [])
        # identical (empty) vectors never dominate each other
        self.assertEqual(sorted(pareto_chunk_ids(conn, hollow)),
                         sorted(objectives))

    def test_track_without_chunks_returns_empty(self):
        self.assertEqual(chunk_objectives(self.lib.conn, self.lib.ids["naked"]), {})


# -------------------------------------------------------------- chunk ids
class ParetoChunkIdsTests(ParetoLibraryTestCase):
    def test_surface_ordered_by_chunk_idx_not_by_id(self):
        conn = self.lib.conn
        tid = repo.upsert_track(conn, 1, str(self.lib.music_dir / "scrambled.wav"),
                                {"filename": "scrambled.wav"})
        # Insertion order deliberately differs from idx order.
        chunk_ids = repo.replace_chunks(conn, tid, [
            (2, 40.0, 60.0), (0, 0.0, 20.0), (1, 20.0, 40.0)])
        # One model per chunk: every chunk is the unit centroid of its own
        # model, so all three objective vectors are mutually non-dominated.
        for chunk_id, model in zip(chunk_ids, ("ma", "mb", "mc")):
            repo.add_chunk_embedding(conn, chunk_id, model,
                                     np.asarray([1.0, 0.0], dtype=np.float32))
        conn.commit()

        by_idx = [int(c["id"]) for c in repo.get_chunks(conn, tid)]
        self.assertEqual(by_idx, [chunk_ids[1], chunk_ids[2], chunk_ids[0]])
        self.assertEqual(pareto_chunk_ids(conn, tid), by_idx)
        self.assertNotEqual(pareto_chunk_ids(conn, tid), chunk_ids)  # not id order

    def test_seed_surface_is_c0_and_c2_in_idx_order(self):
        self.assertEqual(pareto_chunk_ids(self.lib.conn, self.lib.seed),
                         self.lib.expected_surface())


# ------------------------------------------------------------- end-to-end
class ParetoSimilarTracksTests(ParetoLibraryTestCase):
    def test_similar_tracks_pareto_dispatch(self):
        results = similar_tracks(self.lib.conn, self.lib.seed,
                                 method="pareto", limit=10)
        # The seed itself leads the list at 100 %, then the best matches.
        self.assertEqual(results[0].track_id, self.lib.seed)
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual([r.track_id for r in results[1:]], [
            self.lib.ids["close"],     # 1.0
            self.lib.ids["mert_only"],  # 1.0 (tie broken by track id)
            self.lib.ids["tricky"],    # 0.853553
            self.lib.ids["both"],      # 0.853553 (tie broken by track id)
            self.lib.ids["dim3"],      # 0.0 (dimension mismatch -> zeros)
            self.lib.ids["far"],       # -0.853553
        ])
        for excluded in ("naked", "hollow", "l3"):
            self.assertNotIn(self.lib.ids[excluded],
                             [r.track_id for r in results[1:]])
        self.assertTrue(all(r.method == "pareto" for r in results))
        scores = [r.score for r in results[1:]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0], 1.0, places=6)
        self.assertAlmostEqual(scores[2], MEAN_BEST, places=6)
        self.assertAlmostEqual(scores[4], 0.0, places=6)
        self.assertAlmostEqual(scores[5], -MEAN_BEST, places=6)
        top = results[1]
        self.assertEqual(top.filename, "close_song.wav")
        self.assertEqual(top.artist, "Bravo")
        self.assertEqual(top.title, "Close Song")
        self.assertEqual(top.path, str(self.lib.music_dir / "close_song.wav"))

    def test_works_without_track_embeddings(self):
        # the fixture never writes track_embeddings — assert it, then search
        self.assertEqual(repo.get_track_embeddings(self.lib.conn, "clap"), {})
        results = similar_tracks(self.lib.conn, self.lib.seed, method="pareto")
        self.assertTrue(results)
        self.assertTrue(all(r.method == "pareto" for r in results))

    def test_close_beats_tricky_thanks_to_surface(self):
        # "tricky" matches the seed's non-surface chunk perfectly, yet the
        # surface-based score keeps "close" (all-surface matches) on top.
        results = similar_tracks(self.lib.conn, self.lib.seed, method="pareto")
        by_id = {r.track_id: r.score for r in results}
        self.assertGreater(by_id[self.lib.ids["close"]],
                           by_id[self.lib.ids["tricky"]])

    def test_direct_call_matches_dispatch(self):
        via_search = similar_tracks(self.lib.conn, self.lib.seed, method="pareto")
        direct = pareto_similar_tracks(self.lib.conn, self.lib.seed)
        # via_search = seed row (100 %) + the direct results
        self.assertEqual([(r.track_id, r.score, r.method) for r in via_search],
                         [(self.lib.seed, 1.0, "pareto")]
                         + [(r.track_id, r.score, r.method) for r in direct])

    def test_progress_cb_receives_coarse_messages(self):
        messages: list[str] = []
        pareto_similar_tracks(self.lib.conn, self.lib.seed,
                              progress_cb=messages.append)
        self.assertEqual(len(messages), 6)  # one per candidate track
        self.assertTrue(all(m.startswith("comparing track") for m in messages))
        self.assertTrue(messages[-1].endswith("(6/6)"))

    def test_limit_truncates(self):
        results = similar_tracks(self.lib.conn, self.lib.seed,
                                 method="pareto", limit=2)
        self.assertEqual([r.track_id for r in results],
                         [self.lib.seed, self.lib.ids["close"],
                          self.lib.ids["mert_only"]])

    def test_deterministic_across_calls(self):
        first = similar_tracks(self.lib.conn, self.lib.seed, method="pareto")
        second = similar_tracks(self.lib.conn, self.lib.seed, method="pareto")
        self.assertEqual([(r.track_id, r.score, r.method) for r in first],
                         [(r.track_id, r.score, r.method) for r in second])

    def test_seed_without_chunks_raises_friendly_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            similar_tracks(self.lib.conn, self.lib.ids["naked"], method="pareto")
        message = str(ctx.exception).lower()
        self.assertIn("chunk", message)
        self.assertIn("analyz", message)

    def test_seed_without_shared_model_raises(self):
        # chunks but no embeddings at all -> no model to share
        with self.assertRaises(RuntimeError):
            similar_tracks(self.lib.conn, self.lib.ids["hollow"], method="pareto")
        # openl3 on l3's chunks, but nobody else carries openl3
        with self.assertRaises(RuntimeError):
            similar_tracks(self.lib.conn, self.lib.ids["l3"], method="pareto")

    def test_unknown_seed_raises(self):
        with self.assertRaises(RuntimeError):
            similar_tracks(self.lib.conn, 424242, method="pareto")


# ------------------------------------------------------------ repo helper
class GetTrackChunkModelsTests(ParetoLibraryTestCase):
    def test_maps_tracks_to_sorted_models(self):
        ids = self.lib.ids
        models = repo.get_track_chunk_models(self.lib.conn)
        self.assertEqual(models[ids["seed"]], ["clap", "mert"])
        self.assertEqual(models[ids["close"]], ["clap"])
        self.assertEqual(models[ids["mert_only"]], ["mert"])
        self.assertEqual(models[ids["both"]], ["clap", "mert"])
        self.assertEqual(models[ids["l3"]], ["openl3"])
        self.assertNotIn(ids["naked"], models)   # no chunks
        self.assertNotIn(ids["hollow"], models)  # chunks but no embeddings


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
