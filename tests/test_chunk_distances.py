"""Chamfer / Earth Mover's Distance between chunk sets + the search on top.

Pure-math tests use small hand-built unit vectors (no Qt, no DB); the
integration tests drive ``similar_tracks_chunk_distance`` against a real
temp Database like tests/test_fft_model.py's SimilarityIntegrationTests.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.db import repo
from app.db.database import Database
from app.similarity.chunk_distances import (
    chamfer_distance,
    cosine_cost_matrix,
    earth_movers_distance,
    similar_tracks_chunk_distance,
)

E1 = np.array([[1.0, 0.0]])          # single-chunk sets for compact cases
E2 = np.array([[0.0, 1.0]])


def _vec(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float32)


class CosineCostMatrixTests(unittest.TestCase):
    def test_identical_rows_cost_zero_and_orthogonal_cost_one(self) -> None:
        cost = cosine_cost_matrix([_vec(1, 0), _vec(0, 1)], [_vec(1, 0)])
        self.assertAlmostEqual(float(cost[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(cost[1, 0]), 1.0, places=6)

    def test_input_scaling_does_not_matter(self) -> None:
        # Rows are normalized internally: 2*e1 has the same cost profile.
        cost = cosine_cost_matrix([_vec(2, 0)], [_vec(1, 0)])
        self.assertAlmostEqual(float(cost[0, 0]), 0.0, places=6)

    def test_costs_are_clipped_into_unit_interval_of_cosine(self) -> None:
        cost = cosine_cost_matrix([_vec(1, 0)], [_vec(-1, 0)])
        self.assertAlmostEqual(float(cost[0, 0]), 2.0, places=6)  # cos = -1

    def test_empty_or_ragged_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            cosine_cost_matrix([], [E1[0]])
        with self.assertRaises(ValueError):
            cosine_cost_matrix([E1[0]], [np.ones(3)])


class ChamferTests(unittest.TestCase):
    def test_identical_sets_cost_zero(self) -> None:
        vecs = [_vec(1, 0), _vec(0.6, 0.8), _vec(-0.2, 0.98)]
        self.assertAlmostEqual(chamfer_distance(vecs, vecs), 0.0, places=6)

    def test_known_pairwise_value(self) -> None:
        self.assertAlmostEqual(chamfer_distance(E1, E2), 1.0, places=6)

    def test_one_to_many_averages_both_directions(self) -> None:
        # e1's nearest in B is itself (0); e2's nearest in A is 1 away.
        # The symmetric Chamfer mean is (mean_A->B + mean_B->A) / 2 =
        # (0 + 0.5) / 2 = 0.25, independent of argument order.
        a, b = E1, np.vstack([E1, E2])
        self.assertAlmostEqual(chamfer_distance(a, b), 0.25, places=6)
        self.assertAlmostEqual(chamfer_distance(b, a), 0.25, places=6)

    def test_zero_vectors_cost_one_against_everything(self) -> None:
        # Silent-chunk zero vectors sit at the noncommittal mid distance.
        self.assertAlmostEqual(
            chamfer_distance([np.zeros(2)], E1), 1.0, places=6)


class EarthMoversTests(unittest.TestCase):
    def test_identical_sets_cost_zero(self) -> None:
        vecs = [_vec(1, 0), _vec(0.6, 0.8), _vec(-0.2, 0.98)]
        # Entropic regularization leaks a little mass across near-diagonal
        # cells, so identical sets land ~1e-4, not exactly 0 — far below the
        # ~1.0 spread between unrelated textures.
        self.assertLess(earth_movers_distance(vecs, vecs), 0.002)

    def test_single_chunk_disjoint_costs_one(self) -> None:
        # All of the mass must travel across a cost-1 edge.
        self.assertAlmostEqual(earth_movers_distance(E1, E2), 1.0, places=4)

    def test_half_the_mass_travels(self) -> None:
        # A holds e1 + e2, B holds only e1: half of A's mass stays, half
        # moves across the cost-1 edge -> 0.5 (and symmetric).
        a, b = np.vstack([E1, E2]), E1
        self.assertAlmostEqual(earth_movers_distance(a, b), 0.5, places=3)
        self.assertAlmostEqual(earth_movers_distance(b, a), 0.5, places=3)

    def test_matched_transposition_costs_almost_zero(self) -> None:
        a = np.vstack([E1, E2])
        b = np.vstack([E2, E1])   # same chunks, different order
        self.assertAlmostEqual(earth_movers_distance(a, b), 0.0, places=4)

    def test_regularization_strength_changes_little(self) -> None:
        a = np.vstack([E1, E2])
        b = np.array([[0.7071, 0.7071]])
        loose = earth_movers_distance(a, b, reg=1.0)
        tight = earth_movers_distance(a, b, reg=0.02)
        self.assertAlmostEqual(loose, tight, delta=0.05)

    def test_result_is_finite_for_zero_vectors(self) -> None:
        value = earth_movers_distance([np.zeros(2)], E2)
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, 1.0, places=3)  # cost from zero row = 1


class _ChunkDistanceDbTestCase(unittest.TestCase):
    """Temp-Database fixture: three tracks with 2-dim chunk embeddings."""

    V_NEAR = _vec(1.0, 0.0)        # seed's texture
    V_NEAR2 = _vec(0.98, 0.199)    # a slightly rotated near-duplicate
    V_FAR = _vec(0.0, 1.0)         # orthogonal texture

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "lib.db")
        with self.db.transaction() as conn:
            folder = repo.add_folder(conn, str(self._tmp.name))
            self.seed_id = self._track(conn, folder, "seed.wav")
            self.near_id = self._track(conn, folder, "near.wav")
            self.far_id = self._track(conn, folder, "far.wav")
            self._embed(conn, self.seed_id, [self.V_NEAR, self.V_NEAR2])
            self._embed(conn, self.near_id, [self.V_NEAR])
            self._embed(conn, self.far_id, [self.V_FAR])

    @staticmethod
    def _track(conn, folder_id: int, name: str) -> int:
        return repo.upsert_track(conn, folder_id, f"/music/{name}",
                                 {"filename": name, "extension": ".wav",
                                  "duration_sec": 40.0})

    @staticmethod
    def _embed(conn, track_id: int, vectors: list[np.ndarray]) -> None:
        spans = [(i, float(i * 20.0), float((i + 1) * 20.0))
                 for i in range(len(vectors))]
        ids = repo.replace_chunks(conn, track_id, spans)
        for chunk_id, vec in zip(ids, vectors):
            repo.add_chunk_embedding(conn, chunk_id, "fft", vec)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class ChunkDistanceSearchTests(_ChunkDistanceDbTestCase):
    def _search(self, seed_id: int, method: str = "emd", limit: int = 20):
        conn = self.db.connect()
        try:
            return similar_tracks_chunk_distance(conn, seed_id,
                                                 method=method, limit=limit)
        finally:
            conn.close()

    def test_emd_ranks_matching_texture_first(self) -> None:
        results = self._search(self.seed_id, "emd")
        self.assertEqual([r.track_id for r in results][0], self.near_id)
        self.assertEqual(results[0].method, "emd")
        self.assertGreater(results[0].score,
                           [r for r in results
                            if r.track_id == self.far_id][0].score)

    def test_chamfer_ranks_matching_texture_first(self) -> None:
        results = self._search(self.seed_id, "chamfer")
        self.assertEqual(results[0].track_id, self.near_id)
        self.assertEqual(results[0].method, "chamfer")

    def test_similarity_is_one_over_one_plus_distance(self) -> None:
        results = self._search(self.seed_id, "chamfer")
        near = [r for r in results if r.track_id == self.near_id][0]
        far = [r for r in results if r.track_id == self.far_id][0]
        # near: nearest-neighbour distance ~0.005 -> score ~0.995; far: the
        # seed's chunks are 11.5° apart, so its mean distance to the
        # orthogonal texture is ~0.85 -> score ~0.54. Both in (0, 1], and
        # the matching texture clearly wins.
        self.assertGreater(near.score, 0.9)
        self.assertGreater(far.score, 0.45)
        self.assertLess(far.score, 0.65)
        self.assertGreater(near.score, far.score)

    def test_seed_excluded_and_limit_respected(self) -> None:
        results = self._search(self.seed_id, "emd", limit=1)
        self.assertEqual([r.track_id for r in results], [self.near_id])
        self.assertNotIn(self.seed_id, [r.track_id for r in results])

    def test_seed_without_chunks_is_a_friendly_error(self) -> None:
        with self.db.transaction() as conn:
            empty_id = self._track(conn, repo.add_folder(conn, "/x"),
                                   "empty.wav")
        with self.assertRaises(RuntimeError) as ctx:
            self._search(empty_id, "emd")
        self.assertIn("no chunks yet", str(ctx.exception))

    def test_mismatched_dimensions_drop_the_model(self) -> None:
        with self.db.transaction() as conn:
            cand = self._track(conn, repo.add_folder(conn, "/y"), "7d.wav")
            ids = repo.replace_chunks(conn, cand, [(0, 0.0, 20.0)])
            repo.add_chunk_embedding(conn, ids[0], "fft",
                                     np.ones(7, dtype=np.float32))
        results = self._search(self.seed_id, "emd")
        # The 7-dim candidate shares the model NAME but not the vector
        # dimension: it must be skipped, not crash or score garbage.
        self.assertNotIn(cand, [r.track_id for r in results])

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._search(self.seed_id, "cosine")


class RestrictedDatasetSearchTests(_ChunkDistanceDbTestCase):
    """``dataset=``-restricted EMD/Chamfer must compare vectors.

    Regression: the restricted path once iterated the ``{chunk_id: vec}``
    dict itself, so it compared chunk-id *scalars* — every cosine cost
    collapsed to 0 and both algorithms reported 100 % for every candidate.
    """

    def _restricted(self, method: str) -> dict[int, float]:
        conn = self.db.connect()
        try:
            results = similar_tracks_chunk_distance(
                conn, self.seed_id, method=method, dataset="fft", limit=20)
        finally:
            conn.close()
        return {r.track_id: r.score for r in results}

    def test_restricted_search_scores_are_real_similarities(self) -> None:
        emd = self._restricted("emd")
        chamfer = self._restricted("chamfer")
        self.assertEqual(set(emd), {self.near_id, self.far_id})
        # Honest scores: nothing collapses to the broken 100 %.
        for score in (*emd.values(), *chamfer.values()):
            self.assertLess(score, 1.0)
            self.assertGreater(score, 0.4)

    def test_restricted_emd_and_chamfer_match_exact_values(self) -> None:
        with self.db.transaction() as conn:
            mix_id = self._track(conn, repo.add_folder(conn, "/mix"),
                                 "mix.wav")
            self._embed(conn, mix_id, [self.V_NEAR, self.V_FAR])
        emd = self._restricted("emd")
        chamfer = self._restricted("chamfer")
        # seed [V_NEAR, V_NEAR2] vs mix [V_NEAR, V_FAR], uniform masses:
        # EMD transports half the mass per row: (0 + 0.801) / 2 = 0.4005
        # (Sinkhorn's entropic smoothing at reg=0.05 leaves ~0.006 bias).
        self.assertAlmostEqual(emd[mix_id], 1.0 / 1.4005, delta=0.01)
        # Chamfer: row mins (0, 0.02) and col mins (0, 0.801) averaged
        # = 0.20525 — clearly NOT the EMD value.
        self.assertAlmostEqual(chamfer[mix_id], 1.0 / 1.20525, places=3)
        self.assertLess(emd[mix_id], chamfer[mix_id])
        # single-chunk candidate {V_NEAR}: EMD transports half the seed
        # mass for free and half at cost d -> d/2.  Chamfer's symmetric
        # mean halves it again: row-mean d/2 and col-mean 0 -> d/4.
        d = float(1.0 - 0.98 / np.sqrt(0.98 ** 2 + 0.199 ** 2))
        self.assertAlmostEqual(emd[self.near_id], 1.0 / (1.0 + d / 2),
                               delta=0.005)
        self.assertAlmostEqual(chamfer[self.near_id], 1.0 / (1.0 + d / 4),
                               places=3)
        self.assertGreater(chamfer[self.near_id], emd[self.near_id])

    def test_restricted_ranks_matching_texture_first(self) -> None:
        scores = self._restricted("chamfer")
        self.assertGreater(scores[self.near_id], scores[self.far_id])


if __name__ == "__main__":
    unittest.main()
