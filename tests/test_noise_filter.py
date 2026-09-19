"""Tests for HDBSCAN/OPTICS noise filtering of chunk datasets.

The noise detection runs WITHIN each song: a chunk is flagged when its
density score (HDBSCAN mutual reachability / OPTICS ordering
reachability) exceeds ``NOISE_REACH_FACTOR`` x its own song's median
score.  Junk must therefore be a minority of a track's chunks — the
same spectrum inside a noise-collage track is not noise.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.similarity.chunk_distances import (  # noqa: E402
    similar_tracks_chunk_distance)
from app.similarity.noise_filter import (  # noqa: E402
    OPTICS_MAX_POINTS, availability, fit_noise_filter, filtered_centroids,
    filters_for_dataset, noise_ids_for)
from app.similarity.search import similar_tracks  # noqa: E402


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _unit(vector) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class _NoiseLibraryTestCase(unittest.TestCase):
    """Temp library: real texture clusters + a planted junk minority.

    * texture A (e1 direction, jittered): 12 chunks on the seed, 10 on
      cand_a, 2 on the tiny track
    * texture C (e2 direction): 10 chunks on cand_c
    * junk: 4 chunks in two orthogonal directions (e3/e4, twice each) on
      the seed, 1 on the tiny track — isolated points that stand far
      outside their own song's density, which the per-song detectors
      must flag.  The seed keeps 25 % junk mass, so unfiltered
      comparisons score cand_a noticeably lower than after filtering.
    """

    DIM = 8

    def setUp(self) -> None:
        self._app = _app()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(Path(self._tmp.name) / "lib.db")
        rng = np.random.default_rng(7)
        cluster_a = [_unit(np.r_[1.0, rng.normal(scale=0.02, size=7)])
                     for _ in range(24)]
        cluster_c = [_unit(np.r_[0.0, 1.0, rng.normal(scale=0.02, size=6)])
                     for _ in range(10)]
        junk = [_unit(np.eye(self.DIM, dtype=float)[2 + (i // 2)])
                for i in range(4)]
        plans = {
            "seed.wav": cluster_a[:12] + junk,
            "cand_a.wav": cluster_a[12:22],
            "cand_c.wav": cluster_c,
            # too few chunks for the per-song detector: never clustered
            "tiny.wav": cluster_a[22:24] + [_unit(np.eye(self.DIM)[5])],
        }
        self.chunk_ids: dict[str, list[int]] = {}
        with self.db.transaction() as conn:
            folder = repo.add_folder(conn, "/music")
            for name, vectors in plans.items():
                track_id = repo.upsert_track(
                    conn, folder, f"/music/{name}",
                    {"filename": name, "extension": ".wav",
                     "duration_sec": 30.0})
                spans = [(i, float(i * 10.0), float((i + 1) * 10.0))
                         for i in range(len(vectors))]
                chunk_ids = repo.replace_chunks(conn, track_id, spans)
                for chunk_id, vec in zip(chunk_ids, vectors):
                    repo.add_chunk_embedding(
                        conn, chunk_id, "fft",
                        np.asarray(vec, dtype=np.float32))
                self.chunk_ids[name] = [int(c) for c in chunk_ids]
        self.track_ids = {}
        conn = self.db.connect()
        for row in conn.execute(
                "SELECT id, filename FROM tracks").fetchall():
            self.track_ids[str(row["filename"])] = int(row["id"])
        # per-track centroids (analysis normally stores these; the centroid
        # algorithm's unfiltered path needs them)
        directions = {"seed.wav": _unit([1.0, 0.10] + [0.0] * (self.DIM - 2)),
                      "cand_a.wav": _unit([1.0] + [0.0] * (self.DIM - 1)),
                      "cand_c.wav": _unit([0.0, 1.0] + [0.0] * 6),
                      "tiny.wav": _unit([1.0] + [0.0] * (self.DIM - 1))}
        with self.db.transaction() as tx:
            for name, track_id in self.track_ids.items():
                repo.set_track_embedding(tx, track_id, "fft",
                                         directions[name])
        self.seed = self.track_ids["seed.wav"]
        self.cand_a = self.track_ids["cand_a.wav"]
        self.cand_c = self.track_ids["cand_c.wav"]
        self.tiny = self.track_ids["tiny.wav"]
        self._conn = self.db.connect()

    def tearDown(self) -> None:
        self._conn.close()

    # the seed's planted junk chunk ids (its last 4 chunks)
    @property
    def seed_junk(self) -> set[int]:
        return set(self.chunk_ids["seed.wav"][12:])


class NoiseFilterFitTests(_NoiseLibraryTestCase):
    def test_hdbscan_flags_planted_junk(self) -> None:
        result = fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        self.assertEqual(result["n_vectors"], 39)
        stored = noise_ids_for(self._conn, "fft", ("hdbscan",))
        self.assertLessEqual(self.seed_junk, stored)     # every junk chunk
        # texture and homogeneous tracks survive untouched
        self.assertFalse(stored & set(self.chunk_ids["seed.wav"][:12]))
        self.assertFalse(stored & set(self.chunk_ids["cand_a.wav"]))
        self.assertFalse(stored & set(self.chunk_ids["cand_c.wav"]))
        # tracks below MIN_TRACK_CHUNKS are skipped entirely
        self.assertFalse(stored & set(self.chunk_ids["tiny.wav"]))
        self.assertEqual(result["n_noise"], len(self.seed_junk))
        summary = filters_for_dataset(self._conn, "fft")["hdbscan"]
        self.assertEqual(summary["n_noise"], result["n_noise"])

    def test_optics_flags_planted_junk(self) -> None:
        result = fit_noise_filter(self.db.db_path, "fft", "optics")
        stored = noise_ids_for(self._conn, "fft", ("optics",))
        self.assertLessEqual(self.seed_junk, stored)
        self.assertFalse(stored & set(self.chunk_ids["seed.wav"][:12]))
        self.assertFalse(stored & set(self.chunk_ids["cand_a.wav"]))
        self.assertFalse(stored & set(self.chunk_ids["cand_c.wav"]))
        self.assertFalse(stored & set(self.chunk_ids["tiny.wav"]))
        self.assertEqual(result["n_noise"], len(self.seed_junk))

    def test_unknown_method_and_tiny_dataset_raise(self) -> None:
        with self.assertRaises(ValueError):
            fit_noise_filter(self.db.db_path, "fft", "kmeans")
        # a dataset with fewer than 20 vectors is refused
        with self.db.transaction() as conn:
            lonely = repo.add_folder(conn, "/lonely")
            track_id = repo.upsert_track(conn, lonely, "/lonely/x.wav",
                                         {"filename": "x.wav"})
            chunk_ids = repo.replace_chunks(conn, track_id,
                                            [(0, 0.0, 10.0)])
            repo.add_chunk_embedding(
                conn, chunk_ids[0], "mert", np.ones(4, dtype=np.float32))
        with self.assertRaises(RuntimeError) as ctx:
            fit_noise_filter(self.db.db_path, "mert", "hdbscan")
        self.assertIn("at least 20", str(ctx.exception))

    def test_point_cap_is_enforced(self) -> None:
        import app.similarity.noise_filter as nf

        original = nf.OPTICS_MAX_POINTS
        nf.OPTICS_MAX_POINTS = 10
        try:
            with self.assertRaises(RuntimeError) as ctx:
                fit_noise_filter(self.db.db_path, "fft", "optics")
            self.assertIn("limited to 10", str(ctx.exception))
        finally:
            nf.OPTICS_MAX_POINTS = original
        self.assertEqual(OPTICS_MAX_POINTS, 1_000_000)  # restored

    def test_refit_replaces_previous_run(self) -> None:
        first = fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        second = fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        self.assertNotEqual(first["filter_id"], second["filter_id"])
        rows = repo.list_noise_filters(self._conn)
        self.assertEqual([str(r["method"]) for r in rows
                          if str(r["dataset"]) == "fft"], ["hdbscan"])

    def test_worker_signals(self) -> None:
        from app.ui.workers import NoiseFilterWorker

        worker = NoiseFilterWorker(self.db.db_path, "fft", "hdbscan")
        seen: dict = {}
        direct = Qt.ConnectionType.DirectConnection   # no event loop in tests
        worker.finished_ok.connect(
            lambda ds, m, nv, nn: seen.update(ds=ds, m=m, nv=nv, nn=nn),
            direct)
        worker.failed.connect(lambda msg: seen.update(error=msg), direct)
        worker.start()
        worker.wait()
        self.assertNotIn("error", seen)
        self.assertEqual((seen["ds"], seen["m"]), ("fft", "hdbscan"))
        self.assertEqual(seen["nv"], 39)
        self.assertGreaterEqual(seen["nn"], 4)

    def test_availability_reports_missing_sklearn(self) -> None:
        import sys

        table = availability()
        if table.get("hdbscan") is None:
            self.skipTest("scikit-learn installed in this environment")
        self.assertIn("pip install", table["hdbscan"])


class NoiseFilteredSearchTests(_NoiseLibraryTestCase):
    """End-to-end: the seed's junk must stop dragging the true match down."""

    def test_filtered_emd_boosts_the_texture_match(self) -> None:
        unfiltered = {r.track_id: r.score for r in
                      similar_tracks_chunk_distance(
                          self._conn, self.seed, method="emd",
                          dataset="fft")}
        fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        filtered = {r.track_id: r.score for r in
                    similar_tracks_chunk_distance(
                        self._conn, self.seed, method="emd", dataset="fft",
                        discard_noise=("hdbscan",))}
        # dropping the seed's junk removes the cost of dragging 25 % of
        # its mass across the space -> the real match scores higher
        self.assertGreater(filtered[self.cand_a], unfiltered[self.cand_a])
        self.assertIn(self.cand_a, filtered)
        self.assertIn(self.cand_c, filtered)

    def test_filtered_chamfer_and_psvi_boost_it_too(self) -> None:
        for method in ("chamfer", "optics-fit"):
            if method == "chamfer":
                fit_noise_filter(self.db.db_path, "fft", "hdbscan")
                algorithm, noise = "chamfer", ("hdbscan",)
            else:
                fit_noise_filter(self.db.db_path, "fft", "optics")
                algorithm, noise = "pareto", ("optics",)
            with self.subTest(algorithm=algorithm):
                unfiltered = {r.track_id: r.score for r in similar_tracks(
                    self._conn, self.seed, dataset="fft",
                    algorithm=algorithm)}
                filtered = {r.track_id: r.score for r in similar_tracks(
                    self._conn, self.seed, dataset="fft",
                    algorithm=algorithm, discard_noise=noise)}
                self.assertGreater(filtered[self.cand_a],
                                   unfiltered[self.cand_a])

    def test_filtered_centroid_recomputes_from_surviving_chunks(self) -> None:
        fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        unfiltered = {r.track_id: r.score for r in similar_tracks(
            self._conn, self.seed, dataset="fft", algorithm="centroid")}
        filtered = {r.track_id: r.score for r in similar_tracks(
            self._conn, self.seed, dataset="fft", algorithm="centroid",
            discard_noise=("hdbscan",))}
        # the seed's centroids are now junk-free -> textures score higher
        self.assertGreater(filtered[self.cand_a], unfiltered[self.cand_a])

    def test_union_of_both_methods(self) -> None:
        fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        fit_noise_filter(self.db.db_path, "fft", "optics")
        union = noise_ids_for(self._conn, "fft", ("hdbscan", "optics"))
        hdbscan_only = noise_ids_for(self._conn, "fft", ("hdbscan",))
        optics_only = noise_ids_for(self._conn, "fft", ("optics",))
        self.assertLessEqual(hdbscan_only, union)
        self.assertLessEqual(optics_only, union)
        results = similar_tracks(
            self._conn, self.seed, dataset="fft", algorithm="emd",
            discard_noise=("hdbscan", "optics"))
        self.assertIn(self.cand_a, [r.track_id for r in results])

    def test_missing_filter_is_a_noop(self) -> None:
        with_filter = similar_tracks(
            self._conn, self.seed, dataset="fft", algorithm="emd",
            discard_noise=("hdbscan",))
        without = similar_tracks(
            self._conn, self.seed, dataset="fft", algorithm="emd")
        self.assertEqual([r.track_id for r in with_filter],
                         [r.track_id for r in without])

    def test_seed_fully_noisy_raises_friendly_error(self) -> None:
        # the per-song detector only ever flags a minority, so drive the
        # search-side guard directly: store a filter that covers every
        # chunk of the seed track
        all_seed = self.chunk_ids["seed.wav"]
        with self.db.transaction() as conn:
            repo.set_noise_filter_result(
                conn, "fft", "hdbscan",
                '{"scope": "test"}', 39, len(all_seed), all_seed)
        for algorithm in ("centroid", "emd", "pareto"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaises(RuntimeError) as ctx:
                    similar_tracks(self._conn, self.seed,
                                   dataset="fft", algorithm=algorithm,
                                   discard_noise=("hdbscan",))
                self.assertIn("every chunk of the seed",
                              str(ctx.exception))

    def test_filtered_centroids_helper(self) -> None:
        fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        noise = noise_ids_for(self._conn, "fft", ("hdbscan",))
        self.assertTrue(noise)
        centroids = filtered_centroids(self._conn, "fft", noise)
        for name in ("seed.wav", "cand_a.wav", "cand_c.wav"):
            self.assertIn(self.track_ids[name], centroids)   # keeps texture
        self.assertEqual(centroids[self.cand_a].size, self.DIM)


class NoiseFilterRepoTests(_NoiseLibraryTestCase):
    def test_reduction_delete_cascades_noise_filters(self) -> None:
        with self.db.transaction() as conn:
            red_id = repo.create_reduction(conn, "PCA-2 of fft", "fft",
                                           "pca", "{}", 2)
            all_chunks = [chunk_id for ids in self.chunk_ids.values()
                          for chunk_id in ids]
            repo.replace_reduced_embeddings(
                conn, red_id, [(chunk_id, np.ones(2, dtype=np.float32))
                               for chunk_id in all_chunks])
        fit_noise_filter(self.db.db_path, f"red:{red_id}", "hdbscan")
        with self.db.transaction() as conn:
            stored = repo.get_noise_filter(conn, f"red:{red_id}", "hdbscan")
            self.assertIsNotNone(stored)
            repo.delete_reduction(conn, red_id)
            self.assertIsNone(
                repo.get_noise_filter(conn, f"red:{red_id}", "hdbscan"))
        # assert OUTSIDE the write transaction: another connection cannot
        # see uncommitted deletes
        self.assertEqual(repo.list_noise_filters(self._conn), [])


if __name__ == "__main__":
    unittest.main()
