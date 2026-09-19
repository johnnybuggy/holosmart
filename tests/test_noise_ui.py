"""Tests for this feature round: unconditional noise tinting, the
post-analysis HDBSCAN/OPTICS clustering, the chunk-tags dialog, and
click-to-play on the visualization scatter."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app.db import repo
from app.db.database import Database
from app.similarity import noise_filter


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class NoisePipelineTests(unittest.TestCase):
    """Post-run clustering + unconditional Chunks-tab tinting."""

    def setUp(self) -> None:
        self._app = _app()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(Path(self._tmp.name) / "lib.db")
        rng = np.random.default_rng(3)
        with self.db.transaction() as conn:
            folder = repo.add_folder(conn, "/music")
            for t in range(3):
                track_id = repo.upsert_track(
                    conn, folder, f"/music/s{t}.wav",
                    {"filename": f"s{t}.wav", "extension": ".wav",
                     "duration_sec": 120.0})
                specs = [(i, float(i * 10), float((i + 1) * 10))
                         for i in range(12)]
                ids = repo.replace_chunks(conn, track_id, specs)
                for cid, i in zip(ids, range(12)):
                    vec = rng.normal(size=8).astype(np.float32)
                    if i == 11:
                        vec = np.zeros(8, dtype=np.float32)
                    repo.add_chunk_embedding(conn, cid, "fft", vec)

    def test_full_fit_marks_every_track_as_clustered(self) -> None:
        """After a full fit the sweep finds NOTHING pending."""
        for method in noise_filter.METHODS:
            noise_filter.fit_noise_filter(self.db.db_path, "fft", method)
        with self.db.transaction() as conn:
            for method in noise_filter.METHODS:
                self.assertEqual(
                    repo.pending_noise_tracks(
                        conn, "fft", method,
                        noise_filter.MIN_TRACK_CHUNKS),
                    [])

    def test_sweep_clusters_only_new_and_reanalyzed_tracks(self) -> None:
        """The button-driven pass judges ONLY pending songs."""
        for method in noise_filter.METHODS:
            noise_filter.fit_noise_filter(self.db.db_path, "fft", method)
        # a NEW track appears (as an analysis run would leave it)
        rng = np.random.default_rng(8)
        with self.db.transaction() as conn:
            new_id = repo.upsert_track(
                conn, 1, "/music/s3.wav",
                {"filename": "s3.wav", "extension": ".wav",
                 "duration_sec": 120.0})
            ids = repo.replace_chunks(
                conn, new_id, [(i, float(i * 10), float((i + 1) * 10))
                               for i in range(12)])
            for cid in ids:
                repo.add_chunk_embedding(
                    conn, cid, "fft",
                    rng.normal(size=8).astype(np.float32))
        with self.db.transaction() as conn:
            self.assertEqual(
                repo.pending_noise_tracks(
                    conn, "fft", "hdbscan",
                    noise_filter.MIN_TRACK_CHUNKS),
                [new_id])
        results = noise_filter.fit_noise_filter_pending(
            self.db.db_path, ["fft"])
        self.assertEqual(len(results), 2)   # one per method
        self.assertEqual({r["method"] for r in results},
                         {"hdbscan", "optics"})
        self.assertTrue(all(r["n_refit_tracks"] == 1 for r in results))
        with self.db.transaction() as conn:
            self.assertEqual(
                repo.pending_noise_tracks(
                    conn, "fft", "hdbscan",
                    noise_filter.MIN_TRACK_CHUNKS),
                [])
            datasets = {str(r["dataset"])
                        for r in repo.list_noise_filters(conn)}
        self.assertEqual(datasets, {"fft"})

    def test_reanalyzed_track_becomes_pending_by_signature(self) -> None:
        """Changed chunk ids (same count) mark a track pending again."""
        noise_filter.fit_noise_filter(self.db.db_path, "fft", "hdbscan")   # noqa: E501 (only hdbscan is asserted)
        rng = np.random.default_rng(9)
        with self.db.transaction() as conn:
            ids = repo.replace_chunks(
                conn, 2, [(i, float(i * 10), float((i + 1) * 10))
                          for i in range(12)])
            for cid in ids:
                repo.add_chunk_embedding(
                    conn, cid, "fft",
                    rng.normal(size=8).astype(np.float32))
        with self.db.transaction() as conn:
            self.assertIn(2, repo.pending_noise_tracks(
                conn, "fft", "hdbscan", noise_filter.MIN_TRACK_CHUNKS))

    def test_sweep_skips_datasets_without_new_tracks(self) -> None:
        """Nothing pending → the sweep fits nothing and returns [].."""
        for method in noise_filter.METHODS:
            noise_filter.fit_noise_filter(self.db.db_path, "fft", method)
        results = noise_filter.fit_noise_filter_pending(
            self.db.db_path, ["fft"])
        self.assertEqual(results, [])

    def test_incremental_refit_touches_only_given_songs(self) -> None:
        """A one-song refit leaves every other song's flags byte-identical."""
        noise_filter.fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        with self.db.transaction() as conn:
            before = noise_filter.noise_ids_for(conn, "fft", ("hdbscan",))
            track1_before = {c for c in before
                             if c in {int(r["id"]) for r in conn.execute(
                                 "SELECT id FROM chunks WHERE track_id = 1")}}
            # give track 1 fresh vectors with obvious junk
            cids = repo.replace_chunks(
                conn, 1, [(i, float(i * 10), float((i + 1) * 10))
                          for i in range(12)])
            junk_id = cids[-1]
        rng = np.random.default_rng(5)
        with self.db.transaction() as conn:
            for i in range(11):
                vec = np.r_[1.0, rng.normal(scale=0.02, size=7)]
                vec = vec / np.linalg.norm(vec)
                repo.add_chunk_embedding(conn, cids[i], "fft",
                                         np.asarray(vec, dtype=np.float32))
            # orthogonal outlier: far from the random-texture blob
            junk_vec = np.zeros(8, dtype=np.float32)
            junk_vec[2] = 1.0
            repo.add_chunk_embedding(conn, junk_id, "fft", junk_vec)
        noise_filter.fit_noise_filter_tracks(self.db.db_path, "fft",
                                             "hdbscan", [1])
        with self.db.transaction() as conn:
            after = noise_filter.noise_ids_for(conn, "fft", ("hdbscan",))
            others_before = before - track1_before
        self.assertEqual(after & others_before, others_before)
        self.assertIn(junk_id, after - before)   # the new junk is flagged

    def test_incremental_refit_creates_filter_when_missing(self) -> None:
        result = noise_filter.fit_noise_filter_tracks(
            self.db.db_path, "fft", "optics", [1, 2, 3])
        self.assertEqual(result["n_refit_tracks"], 3)
        with self.db.transaction() as conn:
            row = repo.get_noise_filter(conn, "fft", "optics")
        self.assertIsNotNone(row)
        self.assertEqual(int(row["n_vectors"]), 36)

    def test_incremental_refit_replaces_reanalyzed_track_flags(self) -> None:
        """Old chunk ids of a re-analyzed song cannot survive as flags."""
        noise_filter.fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        with self.db.transaction() as conn:
            old_ids = [int(r["id"]) for r in conn.execute(
                "SELECT id FROM chunks WHERE track_id = 2")]
        # full re-analysis: all chunks replaced, vectors re-created
        with self.db.transaction() as conn:
            cids = repo.replace_chunks(
                conn, 2, [(i, float(i * 10), float((i + 1) * 10))
                          for i in range(12)])
            rng = np.random.default_rng(6)
            for cid in cids:
                repo.add_chunk_embedding(
                    conn, cid, "fft",
                    rng.normal(size=8).astype(np.float32))
        noise_filter.fit_noise_filter_tracks(self.db.db_path, "fft",
                                             "hdbscan", [2])
        with self.db.transaction() as conn:
            stored = noise_filter.noise_ids_for(conn, "fft", ("hdbscan",))
        self.assertEqual(stored & set(old_ids), set())

    def test_tinting_is_unconditional(self) -> None:
        """Flagged chunks are tinted even with every checkbox unchecked."""
        noise_filter.fit_noise_filter(self.db.db_path, "fft", "hdbscan")
        from app.ui.detail_pane import DetailPane

        from app.config import AppConfig

        pane = DetailPane(self.db, AppConfig())
        pane.show()
        self.addCleanup(pane.close)
        track_id = 1
        pane.show_track(track_id)
        self._app.processEvents()
        with self.db.transaction() as conn:
            pane._refresh_chunks(conn, track_id)
        tinted = [
            c for c in range(pane._chunks_table.rowCount())
            if pane._chunks_table.item(c, 0).background().color() != pane
            ._chunks_table.palette().window().color()]
        # some junk chunk was flagged and tinted with NO checkbox checked
        self.assertGreater(len(tinted), 0)
        for method in ("hdbscan", "optics"):
            pane._noise_checks[method].setChecked(False)
        with self.db.transaction() as conn:
            pane._refresh_chunks(conn, track_id)
        tinted_again = [
            c for c in range(pane._chunks_table.rowCount())
            if pane._chunks_table.item(c, 0).background().color() != pane
            ._chunks_table.palette().window().color()]
        self.assertEqual(tinted_again, tinted)

    def test_noise_checkboxes_live_on_the_similar_tab(self) -> None:
        from app.ui.detail_pane import DetailPane

        from app.config import AppConfig

        pane = DetailPane(self.db, AppConfig())
        pane.show()
        self.addCleanup(pane.close)
        self._app.processEvents()
        # Tab order: Overview(0), Chunks(1), Similar(2), Learning(3).
        similar_page = pane.widget(2)
        for method in ("hdbscan", "optics"):
            checkbox = pane._noise_checks[method]
            ancestor = checkbox.parent()
            found = False
            while ancestor is not None:
                if ancestor is similar_page:
                    found = True
                    break
                ancestor = ancestor.parentWidget()
            self.assertTrue(found, method)


if __name__ == "__main__":
    unittest.main()

class NoiseSweepButtonTests(unittest.TestCase):
    """The Chunks tab's "Find outliers" button drives the sweep worker."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = _app()

    def test_button_exists_and_emits(self) -> None:
        from PySide6.QtWidgets import QPushButton

        from app.config import AppConfig
        from app.ui.detail_pane import DetailPane

        pane = DetailPane(Database(Path(tempfile.mkdtemp()) / "x.db"),
                          AppConfig())
        try:
            self.assertIsInstance(pane._noise_sweep_button, QPushButton)
            self.assertIn("newly analyzed",
                          pane._noise_sweep_button.toolTip())
            hits: list[bool] = []
            pane.noise_sweep_requested.connect(lambda: hits.append(True))
            pane._noise_sweep_button.click()
            self.assertEqual(hits, [True])
        finally:
            pane.setParent(None)
            del pane

    def test_analysis_run_touches_no_noise_tables(self) -> None:
        """Analysis runs carry no noise clustering any more."""
        from app.ui.workers import AnalysisWorker

        self.assertFalse(hasattr(AnalysisWorker,
                                 "_post_run_noise_clustering"))
        self.assertFalse(hasattr(AnalysisWorker, "_vector_snapshot"))
