"""Learning tab backend: pair storage, weight optimizer, weighted distances,
per-component FFT normalization, and the analysis skip fix."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.normalization import (  # noqa: E402
    component_stats, normalize_model, normalize_model_components,
    rebuild_model_centroids)
from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.learning import weights as lw  # noqa: E402
from app.models.registry import get_plugin  # noqa: E402
from app.similarity.search import cosine, similar_tracks  # noqa: E402

try:  # Qt-backed tests reuse the UI base fixtures
    from tests.test_ui import UiTestBase  # noqa: E402
except Exception:  # pragma: no cover
    UiTestBase = unittest.TestCase
import soundfile as sf  # noqa: E402


class _TempDbTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="holosmart-learning-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db = Database(self.root / "library.db")
        self.conn = self.db.connect()
        self.addCleanup(self.conn.close)
        (self.root / "music").mkdir(exist_ok=True)
        self.folder_id = repo.add_folder(self.conn, str(self.root / "music"))
        self.conn.commit()

    def add_track(self, filename: str, vectors: list[list[float]],
                  model: str = "fft", sine: bool = False) -> int:
        path = str(self.root / "music" / filename)
        # a real wav so the pipeline can actually decode it
        import soundfile as sf

        if sine:
            t = np.linspace(0, 1.0, 8000, endpoint=False)
            samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        else:
            samples = np.zeros(8000, dtype=np.float32)
        sf.write(path, samples, 8000)
        track_id = repo.upsert_track(self.conn, self.folder_id, path, {
            "filename": filename, "artist": "A", "title": "T",
            "extension": ".wav", "duration_sec": 1.0,
        })
        chunk_ids = repo.replace_chunks(self.conn, track_id, [
            (i, i * 10.0, i * 10.0 + 10.0) for i in range(len(vectors))])
        for chunk_id, vector in zip(chunk_ids, vectors):
            repo.add_chunk_embedding(self.conn, chunk_id, model,
                                     np.asarray(vector, dtype=np.float32))
        from app.similarity.search import ensure_track_embeddings
        ensure_track_embeddings(self.conn, [track_id])
        self.conn.commit()
        return track_id


# ------------------------------------------------------------------- pairs --
class PairStorageTests(_TempDbTestCase):
    def test_add_pair_dedupes_both_orders(self) -> None:
        a = self.add_track("a.wav", [[1.0, 0.0]])
        b = self.add_track("b.wav", [[0.9, 0.1]])
        first = lw.add_pair(self.conn, a, b)
        self.assertIsNotNone(first)
        self.assertIsNone(lw.add_pair(self.conn, b, a))   # reverse duplicate
        self.assertIsNone(lw.add_pair(self.conn, a, a))   # same track twice
        self.assertEqual(len(lw.list_pairs(self.conn)), 1)

    def test_remove_pair(self) -> None:
        a = self.add_track("a.wav", [[1.0, 0.0]])
        b = self.add_track("b.wav", [[0.9, 0.1]])
        pair_id = lw.add_pair(self.conn, a, b)
        lw.remove_pair(self.conn, pair_id)
        self.assertEqual(lw.list_pairs(self.conn), [])

    def test_list_pairs_carries_filenames(self) -> None:
        a = self.add_track("alpha.wav", [[1.0, 0.0]])
        b = self.add_track("beta.wav", [[0.9, 0.1]])
        lw.add_pair(self.conn, a, b)
        pairs = lw.list_pairs(self.conn)
        self.assertEqual((pairs[0]["name_a"], pairs[0]["name_b"]),
                         ("alpha.wav", "beta.wav"))


# --------------------------------------------------------------- optimizer --
class LearnWeightsTests(unittest.TestCase):
    def _pairs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(11)
        pairs = []
        for _ in range(25):
            base = rng.normal(size=40) * 0.05
            a, b = base.copy(), base + rng.normal(size=40) * 0.02
            a[5:] += rng.normal(size=35) * 3.0
            b[5:] += rng.normal(size=35) * 3.0
            pairs.append((a, b))
        return pairs

    def test_agreeing_components_dominate(self) -> None:
        weights = lw.learn_weights(self._pairs())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=9)
        top5 = set(np.argsort(weights)[::-1][:5])
        self.assertEqual(top5, {0, 1, 2, 3, 4})
        # noise components carry no similarity → weight 0
        self.assertEqual(float(weights[5:].max()), 0.0)

    def test_degenerate_pairs_give_uniform_weights(self) -> None:
        zeros = [np.zeros(4), np.zeros(4)]
        weights = lw.learn_weights([(np.array([1.0, 2.0, 3.0, 4.0]),
                                     np.array([1.0, 2.0, 3.0, 4.0]))])
        np.testing.assert_allclose(weights, np.ones(4))

    def test_unusable_pairs_raise(self) -> None:
        with self.assertRaises(RuntimeError):
            lw.learn_weights([])

    def test_learn_and_store_persists(self) -> None:
        db = Database(Path(tempfile.mkdtemp(prefix="lw-")) / "l.db")
        conn = db.connect()
        a = repo.upsert_track(conn, repo.add_folder(conn, "/tmp"), "/a.mp3",
                              {"filename": "a.mp3"})
        b = repo.upsert_track(conn, 1, "/b.mp3", {"filename": "b.mp3"})
        rng = np.random.default_rng(3)
        base = rng.normal(size=6) * 0.1
        repo.set_track_embedding(conn, a, "fft",
                                 np.concatenate([base, rng.normal(size=6) * 5]))
        repo.set_track_embedding(conn, b, "fft",
                                 np.concatenate([base, rng.normal(size=6) * 5]))
        conn.commit()
        lw.add_pair(conn, a, b)
        conn.commit()
        conn.close()
        summary = lw.learn_and_store(db, "fft")
        self.assertEqual(summary["pairs_used"], 1)
        weights = summary["weights"]
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=9)
        # components 0..5 agree; 6..11 are noise → top weights at 0..5
        self.assertGreater(weights[:6].min(), weights[6:].max())
        conn = db.connect()
        stored = lw.load_weight_vector(conn, "fft")
        conn.close()
        self.assertIsNotNone(stored)
        np.testing.assert_allclose(stored, weights)


# ------------------------------------------------------- weighted distance --
class WeightedDistanceTests(_TempDbTestCase):
    def test_weights_change_search_scores(self) -> None:
        """The search must apply learned weights to every comparison."""
        seed = self.add_track("seed.wav", [[1.0, 0.0], [0.9, 0.1]])
        other = self.add_track("other.wav", [[0.8, 0.2], [0.7, 0.3]])
        p1 = self.add_track("p1.wav", [[0.1, 1.0], [0.2, 0.9]])
        p2 = self.add_track("p2.wav", [[0.3, 1.0], [0.0, 1.0]])
        lw.add_pair(self.conn, p1, p2)   # pairs agree on component 1
        self.conn.commit()
        before = {r.track_id: r.score for r in similar_tracks(
            self.conn, seed, dataset="fft", limit=10)}
        lw.learn_and_store(self.db, "fft")
        weights = lw.load_weight_vector(self.conn, "fft")
        self.assertGreater(weights[1], weights[0])
        after = {r.track_id: r.score for r in similar_tracks(
            self.conn, seed, dataset="fft", limit=10)}
        self.assertIn(other, after)
        self.assertNotEqual(round(before[other], 6), round(after[other], 6))
        # clearing restores the unweighted scores exactly
        lw.clear_weights(self.conn, "fft")
        self.conn.commit()
        cleared = {r.track_id: r.score for r in similar_tracks(
            self.conn, seed, dataset="fft", limit=10)}
        self.assertAlmostEqual(cleared[other], before[other], places=6)

    def test_weighted_cosine_math(self) -> None:
        """cos(sqrt(w)∘a, sqrt(w)∘b) is the weight-weighted cosine."""
        a = np.array([0.2, 1.0])
        b = np.array([-0.2, 1.0])   # near-orthogonal first components
        unweighted = cosine(a, b)
        weights = np.array([1.0, 20.0])   # component 1 carries similarity
        # emphasizing the agreeing component pulls the pair closer together
        weighted = cosine(lw.apply_weight_vector(a, weights),
                          lw.apply_weight_vector(b, weights))
        self.assertGreater(weighted, unweighted)
        self.assertGreater(weighted, 0.99)

    def test_weight_vector_applies_sqrt_scaling(self) -> None:
        vec = np.array([3.0, 4.0])
        out = lw.apply_weight_vector(vec, np.array([4.0, 1.0]))
        np.testing.assert_allclose(out, [6.0, 4.0])
        # no weights / mismatched dims → unchanged
        np.testing.assert_allclose(lw.apply_weight_vector(vec, None), vec)
        np.testing.assert_allclose(
            lw.apply_weight_vector(vec, np.ones(5)), vec)

    def test_reduced_and_text_datasets_never_get_weights(self) -> None:
        self.assertIsNone(lw.weight_vector_for_dataset(self.conn, "red:1"))
        self.assertIsNone(lw.weight_vector_for_dataset(self.conn,
                                                       "ollama:x"))

    def test_scale_model_vectors_handles_both_containers(self) -> None:
        lw.save_weights(self.conn, "fft", np.array([4.0, 1.0]))
        lw.save_weights(self.conn, "clap", np.array([4.0, 1.0]))
        self.conn.commit()
        scaled = lw.scale_model_vectors(self.conn, {
            "fft": {"c1": np.array([3.0, 4.0])},
            "clap": [np.array([1.0, 1.0])],
            "red:1": {"c2": np.array([9.0, 9.0])},
        })
        np.testing.assert_allclose(scaled["fft"]["c1"], [6.0, 4.0])
        np.testing.assert_allclose(scaled["clap"][0], [2.0, 1.0])
        np.testing.assert_allclose(scaled["red:1"]["c2"], [9.0, 9.0])


# ------------------------------------------------- FFT per-component norm. --
class FftNormalizationTests(_TempDbTestCase):
    def test_zscore_per_component_and_centroid_rebuild(self) -> None:
        self.add_track("a.wav", [[10.0, 100.0], [12.0, 100.0]])
        self.add_track("b.wav", [[8.0, 104.0], [10.0, 96.0]])
        n = normalize_model(self.db, "fft")
        self.assertGreater(n[0], 0)
        self.assertGreater(n[1], 0)
        mean, std, count = component_stats(self.conn, "fft")
        np.testing.assert_allclose(mean, np.zeros(2), atol=1e-5)
        np.testing.assert_allclose(std, np.ones(2), atol=1e-3)
        # centroid rows were rebuilt from the normalized vectors
        for track_id in (1, 2):
            centroid = repo.get_track_embedding(self.conn, track_id, "fft")
            self.assertIsNotNone(centroid)
            self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0,
                                   places=5)
        # idempotent: a second pass rewrites nothing
        self.assertEqual(normalize_model_components(self.conn, "fft"), 0)

    def test_fewer_than_two_vectors_is_a_noop(self) -> None:
        self.add_track("solo.wav", [[5.0, 5.0]])
        self.assertEqual(normalize_model(self.db, "fft"), (0, 0))

    def test_zero_variance_component_becomes_flat_zero(self) -> None:
        self.add_track("a.wav", [[1.0, 7.0], [1.0, 9.0]])
        self.add_track("b.wav", [[1.0, 3.0], [1.0, 5.0]])
        normalize_model(self.db, "fft")
        vecs = [repo.get_chunk_embeddings(self.conn, cid, "fft")[0]["vec"]
                for cid in (1, 2, 3, 4)]
        for vec in vecs:
            self.assertAlmostEqual(float(vec[0]), 0.0)  # zero-var → flat 0
        comp1 = np.array([v[1] for v in vecs])
        np.testing.assert_allclose(comp1.mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(comp1.std(), 1.0, atol=1e-2)


class AnalysisWorkerNormalizationTest(_TempDbTestCase):
    """The analysis run's post-pass normalizes FFT vectors automatically."""

    def test_worker_run_normalizes_fft_after_analysis(self) -> None:
        """FFT-only run: the post-pass standardizes the whole FFT dataset."""
        import app.analysis.pipeline as pipeline_mod
        import app.ui.workers as workers_mod
        from unittest import mock

        from tests.test_chunk_progress import FakeChunkPlugin

        class VarFake(FakeChunkPlugin):
            """Fake whose vectors depend on the chunk loudness."""

            def embed(self, chunks, sr):
                return [np.asarray(
                    [0.1 + float(np.sqrt(np.mean(c ** 2))) * (i + 1)
                     for i in range(4)], dtype=np.float32)
                    for c in chunks]

        plugin = VarFake(name="fft", provides_text=False)
        a = self.add_track("a.wav", [], sine=False)
        b = self.add_track("b.wav", [], sine=True)   # different content
        config = AppConfig()
        config.models = ["fft"]
        from app.ui.workers import AnalysisWorker

        worker = AnalysisWorker(self.db.db_path, config, [a, b])
        with mock.patch.object(pipeline_mod, "get_plugin", lambda name: plugin), \
                mock.patch.object(workers_mod, "get_plugin",
                                  lambda name: plugin):
            worker.run()   # synchronous: normalization happens inside run()
        mean, std, count = component_stats(self.conn, "fft")
        self.assertGreaterEqual(count, 2)
        np.testing.assert_allclose(mean, np.zeros(mean.size), atol=1e-4)
        np.testing.assert_allclose(std, np.ones(std.size), atol=1e-2)

    def test_non_fft_run_still_normalizes_fft_dataset(self) -> None:
        """Normalization is database-wise and unconditional: a run that did
        NOT use FFT (e.g. Analyse Selected with MERT only) still ends with
        the stored FFT vectors re-standardized per component."""
        import app.analysis.pipeline as pipeline_mod
        import app.ui.workers as workers_mod
        from unittest import mock

        from tests.test_chunk_progress import FakeChunkPlugin

        # pre-existing, un-normalized FFT dataset (written earlier)
        a = self.add_track("a.wav", [[10.0, 100.0], [12.0, 100.0]])
        b = self.add_track("b.wav", [[8.0, 104.0], [10.0, 96.0]])

        from app.ui.workers import AnalysisWorker

        class VarFake(FakeChunkPlugin):
            def embed(self, chunks, sr):
                return [np.asarray([0.5, 0.5, 0.5, 0.5],
                                   dtype=np.float32) for _ in chunks]

        mert = VarFake(name="mert330", provides_text=False)
        config = AppConfig()
        config.models = ["mert330"]           # NOT fft
        worker = AnalysisWorker(self.db.db_path, config, [a])
        with mock.patch.object(pipeline_mod, "get_plugin",
                               lambda name: mert), \
                mock.patch.object(workers_mod, "get_plugin",
                                  lambda name: mert):
            worker.run()
        # the MERT run re-chunked track a (its vectors cascade away), but
        # the REMAINING fft dataset came out re-standardized — proof the
        # post-pass ran even though the run itself was MERT-only
        mean, std, count = component_stats(self.conn, "fft")
        self.assertGreaterEqual(count, 2)
        np.testing.assert_allclose(mean, np.zeros(mean.size), atol=1e-4)
        np.testing.assert_allclose(std, np.ones(std.size), atol=1e-2)

    def test_run_without_fft_vectors_skips_normalization(self) -> None:
        """A library without FFT vectors pays no normalization cost."""
        import app.analysis.pipeline as pipeline_mod
        import app.ui.workers as workers_mod
        from unittest import mock

        from tests.test_chunk_progress import FakeChunkPlugin

        from app.ui.workers import AnalysisWorker

        a = self.add_track("a.wav", [], model="clap", sine=True)
        plugin = FakeChunkPlugin(name="clap", provides_text=False)
        config = AppConfig()
        config.models = ["clap"]
        worker = AnalysisWorker(self.db.db_path, config, [a])
        calls = []
        with mock.patch.object(pipeline_mod, "get_plugin",
                               lambda name: plugin), \
                mock.patch.object(workers_mod, "get_plugin",
                                  lambda name: plugin), \
                mock.patch("app.analysis.normalization.normalize_model",
                           side_effect=lambda db, m: calls.append(m) or (0, 0)):
            worker.run()
        self.assertEqual(calls, [])   # normalize_model never invoked


class SkipPolicyTests(_TempDbTestCase):
    """A track analyzed with FFT only must NOT be skipped after enabling
    MERT-330M — the incremental pipeline fills the new model in."""

    def test_track_model_coverage(self) -> None:
        a = self.add_track("a.wav", [[1.0], [2.0]], model="fft")
        n_chunks, coverage = repo.track_model_coverage(self.conn, a)
        self.assertEqual(n_chunks, 2)
        self.assertEqual(coverage, {"fft": 2})

    def test_fully_analyzed_requires_every_enabled_model(self) -> None:
        from tests.test_chunk_progress import FakeChunkPlugin

        a = self.add_track("a.wav", [[1.0], [2.0]], model="fft")
        self.add_track("other.wav", [[1.0], [2.0]], model="fft")

        from app.ui.workers import AnalysisWorker

        config = AppConfig()
        config.models = ["fft", "mert330"]
        worker = AnalysisWorker(self.db.db_path, config, [a])
        # only fft available → track counts as fully analyzed
        available = {"fft": True, "mert330": False}
        self.assertTrue(worker._fully_analyzed(self.conn, a, available))
        # MERT-330M becomes available → no longer fully analyzed
        available["mert330"] = True
        self.assertFalse(worker._fully_analyzed(self.conn, a, available))
        # no chunks at all → never fully analyzed
        self.assertFalse(worker._fully_analyzed(self.conn, 999999,
                                                available))

    def test_worker_fills_in_new_model_with_skip_analyzed(self) -> None:
        """The reported bug: FFT-analyzed tracks were skipped after enabling
        MERT-330M.  The second run must fill the new model in and keep the
        stored FFT vectors byte-identical."""
        from unittest import mock

        import app.analysis.pipeline as pipeline_mod
        import app.ui.workers as workers_mod
        from tests.test_chunk_progress import FakeChunkPlugin

        a = self.add_track("a.wav", [], sine=True)

        from app.ui.workers import AnalysisWorker

        fft_config = AppConfig()
        fft_config.models = ["fft"]
        fft_plugin = FakeChunkPlugin(name="fft", provides_text=False)
        with mock.patch.object(pipeline_mod, "get_plugin",
                               lambda name: fft_plugin), \
                mock.patch.object(workers_mod, "get_plugin",
                                  lambda name: fft_plugin):
            AnalysisWorker(self.db.db_path, fft_config, [a]).run()
        _n, coverage = repo.track_model_coverage(self.conn, a)
        self.assertEqual(coverage.get("fft"), 1)
        fft_blob = self.conn.execute(
            "SELECT vector FROM embeddings e JOIN chunks c "
            "ON c.id = e.chunk_id WHERE c.track_id = ? AND e.model = 'fft'",
            (a,)).fetchone()[0]

        # MERT-330M gets enabled; a plain (skip_analyzed) run must visit
        # the analyzed track and add the missing model.
        mert_plugin = FakeChunkPlugin(name="mert330", provides_text=False)
        plugins = {"fft": fft_plugin, "mert330": mert_plugin}
        mert_config = AppConfig()
        mert_config.models = ["mert330"]
        worker = AnalysisWorker(self.db.db_path, mert_config, [a],
                                skip_analyzed=True)
        with mock.patch.object(pipeline_mod, "get_plugin",
                               lambda name: plugins[name]), \
                mock.patch.object(workers_mod, "get_plugin",
                                  lambda name: plugins[name]):
            worker.run()
        _n, coverage = repo.track_model_coverage(self.conn, a)
        self.assertEqual(coverage.get("mert330"), 1)   # filled in
        blob_now = self.conn.execute(
            "SELECT vector FROM embeddings e JOIN chunks c "
            "ON c.id = e.chunk_id WHERE c.track_id = ? AND e.model = 'fft'",
            (a,)).fetchone()[0]
        self.assertEqual(blob_now, fft_blob)           # FFT untouched



# ------------------------------------------------------------------- UI ----
class LearningTabTests(UiTestBase):
    """The Learning tab: pair flow, weight display, persistence, clear."""

    def _add_extra_track(self, filename: str) -> int:
        path = self.dir / filename
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.2 * np.sin(2 * np.pi * 300 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": filename, "extension": ".wav",
                 "duration_sec": 1.0})

    def _pane(self):
        from app.ui.detail_pane import DetailPane

        config = AppConfig()
        config.use_ollama = False
        return DetailPane(self.db, config)

    def test_pair_flow_add_duplicate_remove(self) -> None:
        pane = self._pane()
        other = self._add_extra_track("second.wav")
        pane.show_track(self.track_id)
        pane._on_learn_set_a()
        pane.show_track(other)
        pane._on_learn_set_b()
        pane._on_learn_add_pair()
        self.assertEqual(pane._pairs_list.count(), 1)
        # same pair again (both slots on one track) → rejected, not added
        pane.show_track(self.track_id)
        pane._on_learn_set_a()
        pane._on_learn_set_b()
        pane._on_learn_add_pair()
        self.assertEqual(pane._pairs_list.count(), 1)
        self.assertIn("already", pane._learn_summary.text())
        # remove
        pane._pairs_list.setCurrentRow(0)
        pane._on_learn_remove_pair()
        self.assertEqual(pane._pairs_list.count(), 0)

    def test_weights_display_and_persistence(self) -> None:
        from app.models.fft_model import FEATURE_DIM

        pane = self._pane()
        other = self._add_extra_track("second.wav")
        pane.show_track(self.track_id)
        pane._on_learn_set_a()
        pane.show_track(other)
        pane._on_learn_set_b()
        pane._on_learn_add_pair()

        weights = np.linspace(1.0, 2.0, FEATURE_DIM)   # ascending
        # the worker persists the weights and then emits learned(...)
        with self.db.transaction() as conn:
            lw.save_weights(conn, "fft", weights)
        pane._on_learned("fft", 1, weights)
        self.assertEqual(pane._weights_table.rowCount(), FEATURE_DIM)
        # row 0 = the strongest component (last index), descending order
        self.assertEqual(pane._weights_table.item(0, 0).text(),
                         pane._weight_component_name(FEATURE_DIM - 1))
        self.assertIn("MOST", pane._learn_summary.text())
        self.assertIn("LEAST", pane._learn_summary.text())
        # a fresh pane shows the persisted weights
        pane2 = self._pane()
        self.assertEqual(pane2._weights_table.rowCount(), FEATURE_DIM)
        # clearing wipes table and persistence
        pane2._on_learn_clear()
        self.assertEqual(pane2._weights_table.rowCount(), 0)
        pane3 = self._pane()
        self.assertEqual(pane3._weights_table.rowCount(), 0)

    def test_learn_failure_reenables_button(self) -> None:
        pane = self._pane()
        pane._learn_button.setEnabled(False)
        pane._on_learn_failed("No song pairs to learn from.")
        self.assertTrue(pane._learn_button.isEnabled())
        self.assertIn("No song pairs", pane._learn_summary.text())

    def test_learn_button_runs_real_worker(self) -> None:
        """The Learn button's worker round-trip on a tiny real dataset."""
        from app.learning.weights import learn_and_store
        from app.models.fft_model import FEATURE_DIM

        pane = self._pane()
        other = self._add_extra_track("second.wav")
        # two tracks with fft centroids (one chunk each) + a pair
        with self.db.transaction() as conn:
            repo.set_track_embedding(conn, self.track_id, "fft",
                                     np.linspace(0.1, 0.4, FEATURE_DIM))
            repo.set_track_embedding(conn, other, "fft",
                                     np.linspace(0.1, 0.4, FEATURE_DIM))
        pane.show_track(self.track_id)
        pane._on_learn_set_a()
        pane.show_track(other)
        pane._on_learn_set_b()
        pane._on_learn_add_pair()
        summary = learn_and_store(self.db, "fft")   # the worker body, sync
        self.assertEqual(summary["pairs_used"], 1)
        pane._on_learned("fft", summary["pairs_used"], summary["weights"])
        self.assertEqual(pane._weights_table.rowCount(), FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()
