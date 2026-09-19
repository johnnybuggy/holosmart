"""Tests for reduction datasets + the reworked Dataset/Algorithm search UI."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.similarity.search import similar_tracks  # noqa: E402


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _ReductionLibraryTestCase(unittest.TestCase):
    """Temp library: three tracks with distinct FFT chunk vectors."""

    def setUp(self) -> None:
        self._app = _app()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(Path(self._tmp.name) / "lib.db")
        with self.db.transaction() as conn:
            fid = repo.add_folder(conn, str(Path(self._tmp.name) / "music"))
            plans = [
                ("seed.wav", [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]),
                ("close.wav", [[0.95, 0.05, 0.0, 0.0], [0.85, 0.1, 0.05, 0.0]]),
                ("far.wav", [[-1.0, 0.0, 0.0, 0.0], [-0.9, 0.0, 0.1, 0.0]]),
            ]
            self.track_ids: dict[str, int] = {}
            self.chunk_ids: list[int] = []
            for filename, chunks in plans:
                tid = repo.upsert_track(conn, fid, f"/music/{filename}", {
                    "filename": filename, "extension": ".wav",
                    "codec": "pcm_s16le", "sample_rate": 8000,
                    "channels": 1, "duration_sec": 20.0, "size_bytes": 100})
                self.track_ids[filename.split(".")[0]] = tid
                chunk_ids = repo.replace_chunks(
                    conn, tid, [(i, i * 10.0, (i + 1) * 10.0)
                                for i in range(len(chunks))])
                self.chunk_ids.extend(int(c) for c in chunk_ids)
                for chunk_id, vec in zip(chunk_ids, chunks):
                    repo.add_chunk_embedding(
                        conn, chunk_id, "fft",
                        np.asarray(vec, dtype=np.float32))
        self.seed = self.track_ids["seed"]

    def tearDown(self) -> None:
        pass

    def _conn(self):
        return self.db.connect()

    def _make_reduction(self, name: str = "PCA-3 of fft") -> int:
        from app.ui.workers import ReductionWorker

        worker = ReductionWorker(self.db.db_path, "fft", "pca", 3, name)
        seen: dict = {}
        direct = Qt.ConnectionType.DirectConnection   # no event loop in tests
        worker.finished_ok.connect(
            lambda rid, name_, n: seen.update(id=rid), direct)
        worker.failed.connect(lambda msg: seen.update(error=msg), direct)
        worker.start()
        worker.wait()
        assert "id" in seen, seen
        return seen["id"]


class ReductionRepoTests(_ReductionLibraryTestCase):
    def test_reduction_crud_roundtrip(self) -> None:
        with self.db.transaction() as conn:
            red_id = repo.create_reduction(conn, "PCA-2 of fft", "fft",
                                           "pca", {"n_neighbors": 15}, 2)
            # one chunk from each of the first two tracks
            rows = [(chunk_id, np.ones(2, dtype=np.float32))
                    for chunk_id in self.chunk_ids[::2][:2]]
            repo.replace_reduced_embeddings(conn, red_id, rows)
            repo.set_reduction_result(conn, red_id, len(rows), [0.7, 0.2])
            stored = repo.get_reduction(conn, red_id)
            listed = repo.list_reductions(conn)
        self.assertEqual(stored["name"], "PCA-2 of fft")
        self.assertEqual(stored["source_model"], "fft")
        self.assertEqual(stored["method"], "pca")
        self.assertEqual(stored["n_vectors"], 2)
        self.assertEqual(stored["n_components"], 2)
        self.assertEqual([r["id"] for r in listed], [red_id])
        with self.db.transaction() as conn:
            vecs = repo.get_reduced_chunk_embeddings(conn, red_id)
            tracks = repo.tracks_with_reduced_chunks(conn, red_id)
            # re-running replaces the stored vector set
            repo.replace_reduced_embeddings(conn, red_id, rows[:1])
            after = repo.get_reduced_chunk_embeddings(conn, red_id)
            repo.delete_reduction(conn, red_id)
            self.assertIsNone(repo.get_reduction(conn, red_id))
            self.assertEqual(repo.list_reductions(conn), [])
        self.assertEqual([v["vec"].size for v in vecs], [2, 2])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(after), 1)


class ReductionRefreshTests(_ReductionLibraryTestCase):
    """Rerun/refresh mechanism: staleness + in-place re-fit."""

    def test_update_reduction_keeps_id_and_metadata(self) -> None:
        with self.db.transaction() as conn:
            red_id = repo.create_reduction(conn, "PCA-2 of fft", "fft",
                                           "pca", {}, 2)
            repo.update_reduction(conn, red_id, "PCA-3 of fft", "pca",
                                  {"n_neighbors": 15}, 3)
            row = repo.get_reduction(conn, red_id)
        self.assertEqual(row["name"], "PCA-3 of fft")
        self.assertEqual(row["n_components"], 3)
        self.assertEqual(row["source_model"], "fft")

    def test_coverage_flags_stale_and_dead_chunk_ids(self) -> None:
        red_id = self._make_reduction("PCA-3 of fft")
        with self.db.transaction() as conn:
            covered, total = repo.reduction_coverage(conn, red_id, "fft")
            self.assertEqual((covered, total), (total, total))  # up to date
            # a NEW track with chunks makes the reduction outdated
            tid = repo.upsert_track(conn, 1, "/music/new.wav",
                                    {"filename": "new.wav"})
            new_chunks = repo.replace_chunks(conn, tid, [(0, 0.0, 10.0)])
            repo.add_chunk_embedding(conn, new_chunks[0], "fft",
                                     np.ones(4, dtype=np.float32))
            covered, total = repo.reduction_coverage(conn, red_id, "fft")
            self.assertLess(covered, total)
            # re-analysis replaces chunk ids: dangling rows stop counting
            repo.replace_chunks(conn, self.seed, [(0, 0.0, 10.0)])
            covered, total = repo.reduction_coverage(conn, red_id, "fft")
            self.assertLess(covered, total)

    def test_worker_refresh_replaces_reduction_in_place(self) -> None:
        from app.ui.workers import ReductionWorker

        red_id = self._make_reduction("PCA-3 of fft")
        # a fourth track appears after the fit
        with self.db.transaction() as conn:
            tid = repo.upsert_track(conn, 1, "/music/new.wav",
                                    {"filename": "new.wav"})
            chunks = repo.replace_chunks(conn, tid, [(0, 0.0, 10.0),
                                                     (1, 10.0, 20.0)])
            for chunk_id in chunks:
                repo.add_chunk_embedding(conn, chunk_id, "fft",
                                         np.ones(4, dtype=np.float32))
        worker = ReductionWorker(self.db.db_path, "fft", "pca", 3,
                                 "PCA-3 of fft",
                                 replace_reduction_id=red_id)
        seen: dict = {}
        direct = Qt.ConnectionType.DirectConnection
        worker.finished_ok.connect(
            lambda rid, name_, n: seen.update(id=rid, n=n), direct)
        worker.failed.connect(lambda msg: seen.update(error=msg), direct)
        worker.start()
        worker.wait()
        self.assertNotIn("error", seen)
        self.assertEqual(seen["id"], red_id)             # SAME dataset id
        with self.db.transaction() as conn:
            row = repo.get_reduction(conn, red_id)
            vecs = repo.get_reduced_chunk_embeddings(conn, red_id)
            centroids = repo.get_track_embeddings(conn, f"red:{red_id}")
        self.assertEqual(row["n_vectors"], seen["n"])
        self.assertEqual(len(vecs), 8)                   # 3*2 old + 2 new
        # centroids rebuilt for every track, none stale
        red_tracks = set(centroids)
        with self.db.transaction() as conn:
            live = {int(r["track_id"]) for r in conn.execute(
                "SELECT DISTINCT track_id FROM chunks").fetchall()}
        self.assertLessEqual(red_tracks, live)


class ReductionWorkerTests(_ReductionLibraryTestCase):
    def _run_worker(self, **overrides) -> dict:
        from app.ui.workers import ReductionWorker

        kwargs = dict(source_model="fft", method="pca", n_components=3,
                      name="PCA-3 of fft")
        kwargs.update(overrides)
        worker = ReductionWorker(self.db.db_path, **kwargs)
        seen: dict = {}
        direct = Qt.ConnectionType.DirectConnection   # no event loop in tests
        worker.stage.connect(
            lambda msg: seen.setdefault("stages", []).append(msg), direct)
        worker.progress.connect(
            lambda v: seen.setdefault("progress", []).append(v), direct)
        worker.finished_ok.connect(
            lambda rid, name, n: seen.update(id=rid, name=name, n=n), direct)
        worker.failed.connect(lambda msg: seen.update(error=msg), direct)
        worker.start()
        worker.wait()
        return seen

    def test_worker_creates_dataset_with_centroids(self) -> None:
        seen = self._run_worker()
        self.assertNotIn("error", seen)
        self.assertEqual(seen["name"], "PCA-3 of fft")
        self.assertEqual(seen["n"], 6)          # 2 chunks x 3 tracks
        with self.db.transaction() as conn:
            red = repo.get_reduction(conn, seen["id"])
            vectors = repo.get_reduced_chunk_embeddings(conn, seen["id"])
            centroids = {tid: repo.get_track_embedding(conn, tid,
                                                       f"red:{seen['id']}")
                         for tid in self.track_ids.values()}
        self.assertEqual(red["method"], "pca")
        self.assertEqual(red["n_components"], 3)
        self.assertTrue(red["explained_variance"])   # PCA ratios stored
        self.assertEqual(len(vectors), 6)
        self.assertTrue(all(v["vec"].size == 3 for v in vectors))
        # every analyzed track got a red:N centroid for centroid search
        for tid, centroid in centroids.items():
            self.assertIsNotNone(centroid, tid)
            self.assertEqual(centroid.size, 3)

    def test_worker_rejects_duplicate_name(self) -> None:
        self._run_worker()
        seen = self._run_worker(name="PCA-3 of fft")
        self.assertIn("already exists", seen["error"])

    def test_worker_reports_missing_data(self) -> None:
        seen = self._run_worker(source_model="clap")
        self.assertIn("analyze more tracks", seen["error"])

    def test_progress_and_stages_flow(self) -> None:
        seen = self._run_worker()
        self.assertTrue(any("Loading" in s for s in seen["stages"]))
        self.assertTrue(any("Fitting" in s for s in seen["stages"]))
        self.assertTrue(any("Storing" in s for s in seen["stages"]))
        self.assertEqual(seen["progress"][-1], 100)


class ReducedDatasetSearchTests(_ReductionLibraryTestCase):
    def test_centroid_search_on_reduced_dataset(self) -> None:
        red_id = self._make_reduction()
        conn = self._conn()
        results = similar_tracks(conn, self.seed, dataset=f"red:{red_id}",
                                 algorithm="centroid")
        # seed first at 100 %, then close before far
        self.assertEqual(results[0].track_id, self.seed)
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual([r.track_id for r in results[1:]],
                         [self.track_ids["close"], self.track_ids["far"]])
        self.assertTrue(all(r.method == f"red:{red_id}"
                            for r in results[1:]))

    def test_chunk_algorithms_on_reduced_dataset(self) -> None:
        red_id = self._make_reduction()
        conn = self._conn()
        for algorithm in ("pareto", "emd", "chamfer", "psvi"):
            with self.subTest(algorithm=algorithm):
                results = similar_tracks(conn, self.seed,
                                         dataset=f"red:{red_id}",
                                         algorithm=algorithm)
                self.assertEqual(results[0].track_id, self.seed)
                self.assertGreaterEqual(len(results), 2)

    def test_psvi_alias_maps_to_pareto(self) -> None:
        conn = self._conn()
        results = similar_tracks(conn, self.seed, dataset="fft",
                                 algorithm="psvi")
        self.assertEqual(results[0].track_id, self.seed)
        self.assertTrue(all(r.method == "pareto" for r in results))

    def test_model_dataset_restricts_candidates(self) -> None:
        from app.similarity.pareto import pareto_similar_tracks

        conn = self._conn()
        # Only tracks with fft chunk vectors are candidates (all three);
        # the direct call keeps its documented seed-excluded contract.
        results = pareto_similar_tracks(conn, self.seed, dataset="fft")
        self.assertEqual([r.track_id for r in results],
                         [self.track_ids["close"], self.track_ids["far"]])

    def test_dataset_validation_errors(self) -> None:
        conn = self._conn()
        with self.assertRaises(RuntimeError) as ctx:
            similar_tracks(conn, self.seed, dataset="red:999")
        self.assertIn("red:999", str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            similar_tracks(conn, self.seed, dataset="clap")
        self.assertIn("not comparable", str(ctx.exception))




class DatasetPickerTests(_ReductionLibraryTestCase):
    """The Similar tab's Dataset + Algorithm pickers."""

    def _pane(self):
        from app.ui.detail_pane import DetailPane
        pane = DetailPane(self.db, AppConfig())
        self.addCleanup(pane.deleteLater)
        return pane

    def test_dataset_combo_lists_models_and_reductions(self) -> None:
        self._make_reduction()
        pane = self._pane()
        data = [pane._dataset_combo.itemData(i)
                for i in range(pane._dataset_combo.count())]
        self.assertIn("fft", data)
        self.assertIn("red:1", data)
        # FFT is usable (three analyzed tracks) …
        fft_item = pane._dataset_combo.model().item(data.index("fft"))
        self.assertTrue(fft_item.isEnabled())
        # … CLAP is not (no chunk vectors in this fixture).
        clap_item = pane._dataset_combo.model().item(data.index("clap"))
        self.assertFalse(clap_item.isEnabled())

    def test_refresh_button_tracks_staleness(self) -> None:
        red_id = self._make_reduction()
        pane = self._pane()
        index = pane._dataset_combo.findData(f"red:{red_id}")
        pane._dataset_combo.setCurrentIndex(index)
        # up to date: enabled with an "up to date" tooltip
        self.assertTrue(pane._refresh_reduce_button.isEnabled())
        self.assertNotIn("⚠", pane._refresh_reduce_button.text())
        self.assertIn("Up to date", pane._refresh_reduce_button.toolTip())
        # new analysis arrives -> the button flags the staleness
        with self.db.transaction() as conn:
            tid = repo.upsert_track(conn, 1, "/music/new.wav",
                                    {"filename": "new.wav"})
            chunk_ids = repo.replace_chunks(conn, tid, [(0, 0.0, 10.0)])
            repo.add_chunk_embedding(conn, chunk_ids[0], "fft",
                                     np.ones(4, dtype=np.float32))
        pane.refresh_datasets()
        self.assertIn("⚠", pane._refresh_reduce_button.text())
        self.assertIn("Outdated", pane._refresh_reduce_button.toolTip())
        self.assertIn("outdated",
                      [pane._dataset_combo.itemText(i)
                       for i in range(pane._dataset_combo.count())
                       if pane._dataset_combo.itemData(i) == f"red:{red_id}"
                       ][0])
        # model datasets never offer refresh
        fft_index = pane._dataset_combo.findData("fft")
        pane._dataset_combo.setCurrentIndex(fft_index)
        self.assertFalse(pane._refresh_reduce_button.isEnabled())

    def test_refresh_dialog_refits_in_place(self) -> None:
        from app.ui.reduction_dialog import ReductionDialog

        red_id = self._make_reduction()
        with self.db.transaction() as conn:
            row = repo.get_reduction(conn, red_id)
        dialog = ReductionDialog(self.db, AppConfig(), rerun_of=row)
        # identity locked, everything else prefilled from the row
        self.assertFalse(dialog._source_combo.isEnabled())
        self.assertFalse(dialog._name_edit.isEnabled())
        self.assertEqual(dialog._name_edit.text(), "PCA-3 of fft")
        self.assertEqual(dialog._source_combo.currentData(), "fft")
        self.assertEqual(dialog._method_combo.currentData(), "pca")
        self.assertEqual(dialog._components_spin.value(), 3)
        self.assertEqual(dialog._run_button.text(), "&Refresh")
        dialog._run()
        dialog._worker.wait()
        self.assertEqual(dialog._worker._replace_reduction_id, red_id)
        self.assertGreaterEqual(dialog._worker._n_components, 3)

    def test_search_requested_carries_dataset_algorithm_noise(self) -> None:
        pane = self._pane()
        pane._current_track_id = self.seed
        seen = []
        pane.search_requested.connect(
            lambda s, d, a, l, n: seen.append((s, d, a, l, n)))
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData("fft"))
        pane._algorithm_combo.setCurrentIndex(
            pane._algorithm_combo.findData("emd"))
        pane._on_search()
        self.assertEqual(seen, [([self.seed], "fft", "emd",
                                 pane._limit_spin.value(), ())])
        # checking a noise filter adds it to the emitted methods
        pane._noise_checks["hdbscan"].setChecked(True)
        pane._on_search()
        self.assertEqual(seen[-1][4], ("hdbscan",))

    def test_noise_toggled_signal_and_tooltips(self) -> None:
        pane = self._pane()
        seen = []
        pane.noise_filter_toggled.connect(
            lambda d, m, on: seen.append((d, m, on)))
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData("fft"))
        # no cached run yet: enabled, unchecked, hint tooltip
        checkbox = pane._noise_checks["hdbscan"]
        self.assertTrue(checkbox.isEnabled())
        self.assertIn("one-time", checkbox.toolTip())
        checkbox.setChecked(True)
        self.assertEqual(seen, [("fft", "hdbscan", True)])
        # after a run is stored the tooltip reports the cached stats
        with self.db.transaction() as conn:
            repo.set_noise_filter_result(conn, "fft", "hdbscan",
                                         "{}", 6, 2, [])
        pane._sync_noise_checkboxes()
        self.assertIn("noise data ready", checkbox.toolTip())
        self.assertIn("2 of 6 chunks", checkbox.toolTip())

    def test_ollama_dataset_locks_algorithm_to_centroid(self) -> None:
        with self.db.transaction() as conn:
            repo.set_track_embedding(conn, self.seed, "ollama:m1",
                                     np.asarray([1.0, 0.0], dtype=np.float32))
            repo.set_track_embedding(conn, self.track_ids["close"], "ollama:m1",
                                     np.asarray([0.9, 0.1], dtype=np.float32))
        pane = self._pane()
        index = pane._dataset_combo.findData("ollama:m1")
        self.assertGreaterEqual(index, 0)
        pane._dataset_combo.setCurrentIndex(index)
        self.assertFalse(pane._algorithm_combo.isEnabled())
        self.assertEqual(pane._algorithm_combo.currentData(), "centroid")
        self.assertEqual(pane._current_search_label(), "ollama:m1")

    def test_chunks_highlight_flagged_outliers_per_method(self) -> None:
        """The Chunks tab tints outlier rows with the method's color."""
        from PySide6.QtGui import QColor

        pane = self._pane()
        pane.show_track(self.seed)
        seed_chunks = self.chunk_ids[0:2]      # seed.wav's two chunks
        with self.db.transaction() as conn:
            repo.set_noise_filter_result(
                conn, "fft", "hdbscan", '{"scope": "test"}', 6, 1,
                [seed_chunks[0]])
            repo.set_noise_filter_result(
                conn, "fft", "optics", '{"scope": "test"}', 6, 2,
                seed_chunks)
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData("fft"))
        # Highlighting is UNCONDITIONAL (post-run clustering keeps the
        # flags fresh): chunk 0 is flagged by both methods -> the blend
        # color, chunk 1 is OPTICS-only -> blue.  No checkbox is checked.
        pane.refresh_chunks()
        table = pane._chunks_table
        self.assertEqual(table.item(0, 0).background().color().name(),
                         "#f48fb1")
        self.assertEqual(table.item(1, 0).background().color().name(),
                         "#90caf9")
        self.assertIn("HDBSCAN", table.item(0, 0).toolTip())
        self.assertIn("OPTICS", table.item(0, 0).toolTip())
        # toggling the (search-only) checkboxes never changes the tint
        pane._noise_checks["hdbscan"].setChecked(True)
        pane._noise_checks["optics"].setChecked(True)
        self.assertEqual(table.item(0, 0).background().color().name(),
                         "#f48fb1")
        pane._noise_checks["hdbscan"].setChecked(False)
        pane._noise_checks["optics"].setChecked(False)
        self.assertEqual(table.item(0, 0).background().color().name(),
                         "#f48fb1")

    def test_multi_reference_seed_list_flow(self) -> None:
        """The Similar tab's reference list: auto-fill, add, remove, emit."""
        pane = self._pane()
        # selecting a file in the tree auto-fills the list
        pane.show_track(self.seed)
        self.assertEqual(pane._seed_ids(), [self.seed])
        # selecting another file REPLACES the auto entry (list follows)
        other = self.track_ids["close"]
        pane.show_track(other)
        self.assertEqual(pane._seed_ids(), [other])
        # "Add selected" pins the list against auto-following (a duplicate
        # add changes nothing but still pins)
        pane._on_add_seed()
        self.assertEqual(pane._seed_ids(), [other])
        pane.show_track(self.seed)
        self.assertEqual(pane._seed_ids(), [other])
        pane._on_add_seed()
        self.assertEqual(pane._seed_ids(), [other, self.seed])
        # search emits ALL references
        seen = []
        pane.search_requested.connect(
            lambda s, d, a, l, n: seen.append((s, d, a, l, n)))
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData("fft"))
        pane._on_search()
        self.assertEqual(seen[-1][0], [other, self.seed])
        # removing the highlighted entry works; clearing restores the auto one
        pane._seed_list.setCurrentRow(1)
        pane._on_remove_seed()
        self.assertEqual(pane._seed_ids(), [other])
        pane._on_clear_seeds()
        # clearing empties the list and the auto-fill stays suppressed for
        # the still-selected file; picking another file resumes it
        self.assertEqual(pane._seed_ids(), [])
        pane.show_track(other)
        self.assertEqual(pane._seed_ids(), [other])
        self.assertEqual(pane._auto_seed_id, other)

    def test_noise_checkbox_survives_dataset_switch(self) -> None:
        """Checked filters stay checked: they are method toggles, not
        dataset-bound (the old auto-uncheck reset the user's choice)."""
        red_id = self._make_reduction()
        pane = self._pane()
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData("fft"))
        pane._noise_checks["hdbscan"].setChecked(True)
        self.assertTrue(pane._noise_checks["hdbscan"].isChecked())
        # switch the Similar dataset to a reduction with no cached noise
        # run: the checkbox must NOT silently reset
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData(f"red:{red_id}"))
        self.assertTrue(pane._noise_checks["hdbscan"].isChecked())

    def test_noise_highlight_ignores_similar_dataset(self) -> None:
        """Grid highlighting pools every stored run of a method."""
        red_id = self._make_reduction()
        pane = self._pane()
        pane.show_track(self.seed)
        seed_chunk = self.chunk_ids[0]
        with self.db.transaction() as conn:
            repo.set_noise_filter_result(
                conn, "fft", "hdbscan", '{"scope": "test"}', 6, 1,
                [seed_chunk])
        # Similar combo points at the reduction (no cached noise run there)
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData(f"red:{red_id}"))
        pane._noise_checks["hdbscan"].setChecked(True)
        table = pane._chunks_table
        self.assertEqual(table.item(0, 0).background().color().name(),
                         "#ffd54f")

    def test_noise_run_dataset_resolution(self) -> None:
        pane = self._pane()
        # a model dataset selected -> used as-is
        pane._dataset_combo.setCurrentIndex(
            pane._dataset_combo.findData("fft"))
        self.assertEqual(pane._resolve_noise_run_dataset(), "fft")
        # no dataset selected, but a track with fft chunks is displayed ->
        # falls back to the track's first vector model
        pane._dataset_combo.setCurrentIndex(-1)
        pane.show_track(self.seed)
        self.assertEqual(pane._resolve_noise_run_dataset(), "fft")

    def test_chunk_double_click_shows_vector_dialog(self) -> None:
        from unittest import mock

        from PySide6.QtWidgets import QDialog, QTableWidget

        from app.models.fft_model import FEATURE_DIM

        pane = self._pane()
        pane.show_track(self.seed)
        col = next(c for c, m in pane._chunks_model_columns.items()
                   if m == "fft")
        seen: list[tuple[int, str]] = []
        tags_opened: list[int] = []
        with mock.patch.object(pane, "_show_chunk_vector_dialog",
                               lambda cid, model: seen.append((cid, model))), \
                mock.patch.object(pane, "_show_chunk_tags_dialog",
                                  lambda cid: tags_opened.append(cid)):
            pane._on_chunk_cell_double_clicked(0, col)
            # a non-model cell opens the ALL-TAGS dialog instead
            pane._on_chunk_cell_double_clicked(0, 0)
        self.assertEqual(seen, [(self.chunk_ids[0], "fft")])
        self.assertEqual(tags_opened, [self.chunk_ids[0]])
        # the real dialog: FEATURE_DIM rows with band-stat names
        dialogs: list[QDialog] = []
        with mock.patch.object(QDialog, "exec",
                               lambda self: dialogs.append(self)):
            pane._show_chunk_vector_dialog(self.chunk_ids[0], "fft")
        table = dialogs[0].findChild(QTableWidget)
        # fixture fft vectors are 4-dim; dimension names still come from
        # the plugin's band-stat table (real vectors are FEATURE_DIM wide)
        self.assertEqual(table.rowCount(), 4)
        self.assertLess(4, FEATURE_DIM)
        self.assertEqual(table.item(0, 0).text(), "0–50 Hz mean")
        self.assertEqual(table.columnCount(), 2)

    def test_playlist_label_combines_algorithm_and_dataset(self) -> None:
        pane = self._pane()
        pane._dataset_combo.setCurrentIndex(pane._dataset_combo.findData("fft"))
        pane._algorithm_combo.setCurrentIndex(
            pane._algorithm_combo.findData("pareto"))
        self.assertEqual(pane._current_search_label(), "pareto:fft")


class ReductionDialogTests(_ReductionLibraryTestCase):
    def test_dialog_runs_worker_and_emits_dataset(self) -> None:
        from app.ui.reduction_dialog import ReductionDialog
        dialog = ReductionDialog(self.db, AppConfig())
        self.addCleanup(dialog.close)
        created = []
        dialog.reduction_created.connect(created.append,
                                         Qt.ConnectionType.DirectConnection)
        dialog._components_spin.setValue(3)
        self.assertEqual(dialog._name_edit.text(), "PCA-3 of fft")
        # A user-typed name is preserved when parameters change.
        dialog._name_edit.setText("My mix space")
        dialog._mark_name_dirty()
        dialog._components_spin.setValue(2)
        self.assertEqual(dialog._name_edit.text(), "My mix space")
        dialog._components_spin.setValue(3)

        dialog._run()
        dialog._worker.wait()
        QApplication.processEvents()   # deliver the queued finished_ok chain
        self.assertEqual(created, ["red:1"])
        self.assertIn("✓", dialog._stage_label.text())
        self.assertEqual(dialog._progress.value(), 100)
        with self.db.transaction() as conn:
            red = repo.get_reduction(conn, 1)
        self.assertEqual(red["n_vectors"], 6)
        self.assertEqual(red["name"], "My mix space")

    def test_method_switch_toggles_parameter_widgets(self) -> None:
        from app.ui.reduction_dialog import ReductionDialog
        dialog = ReductionDialog(self.db, AppConfig())
        self.addCleanup(dialog.close)
        # PCA default
        self.assertTrue(dialog._neighbors_spin.isEnabled() is False)
        dialog._method_combo.setCurrentIndex(
            dialog._method_combo.findData("umap"))
        self.assertTrue(dialog._neighbors_spin.isEnabled())
        self.assertTrue(dialog._min_dist_spin.isEnabled())
        self.assertFalse(dialog._perplexity_spin.isEnabled())
        self.assertEqual(dialog._components_spin.maximum(), 32)
        # t-SNE: components locked to 2, perplexity editable
        tsne_index = dialog._method_combo.findData("tsne")
        if not dialog._method_combo.model().item(tsne_index).isEnabled():
            self.skipTest("scikit-learn installed in this environment")
        dialog._method_combo.setCurrentIndex(tsne_index)
        self.assertEqual(dialog._components_spin.maximum(), 2)
        self.assertEqual(dialog._components_spin.value(), 2)
        self.assertTrue(dialog._perplexity_spin.isEnabled())




class MethodAvailabilityTests(_ReductionLibraryTestCase):
    """The dialog greys out methods the dataset size rules out."""

    def _dialog(self):
        from app.ui.reduction_dialog import ReductionDialog
        dialog = ReductionDialog(self.db, AppConfig())
        self.addCleanup(dialog.close)
        return dialog

    def test_chunk_vector_counts_helper(self) -> None:
        conn = self._conn()
        counts = repo.chunk_vector_counts(conn)
        self.assertEqual(counts.get("fft"), 6)   # 2 chunks x 3 tracks
        self.assertNotIn("clap", counts)

    def test_methods_over_dataset_cap_are_greyed_out(self) -> None:
        from app.analysis import dim_reduction
        if dim_reduction.availability().get("tsne"):
            self.skipTest("scikit-learn installed in this environment")
        dialog = self._dialog()
        tsne_index = dialog._method_combo.findData("tsne")
        # fixture: 6 fft vectors — below every real cap, so enabled …
        self.assertTrue(dialog._method_combo.model().item(tsne_index)
                        .isEnabled())
        # … but with the cap patched down, t-SNE is disabled with a hint
        original = dim_reduction.TSNE_MAX_POINTS
        dim_reduction.TSNE_MAX_POINTS = 5
        try:
            dialog._sync_method_availability()
            item = dialog._method_combo.model().item(tsne_index)
            self.assertFalse(item.isEnabled())
            self.assertIn("limited to 5", item.toolTip())
            self.assertIn("6 chunk vectors", item.toolTip())
        finally:
            dim_reduction.TSNE_MAX_POINTS = original
        dialog._sync_method_availability()
        self.assertTrue(dialog._method_combo.model().item(tsne_index)
                        .isEnabled())

    def test_over_cap_selection_falls_back_to_usable_method(self) -> None:
        from app.analysis import dim_reduction
        if dim_reduction.availability().get("tsne"):
            self.skipTest("scikit-learn installed in this environment")
        dialog = self._dialog()
        tsne_index = dialog._method_combo.findData("tsne")
        pca_index = dialog._method_combo.findData("pca")
        dialog._method_combo.setCurrentIndex(tsne_index)
        original = dim_reduction.TSNE_MAX_POINTS
        dim_reduction.TSNE_MAX_POINTS = 5
        try:
            dialog._sync_method_availability()
            # the stale over-cap selection is replaced by a usable method
            self.assertNotEqual(dialog._current_method(), "tsne")
            self.assertEqual(dialog._current_method(), "pca")
            self.assertGreaterEqual(pca_index, 0)
        finally:
            dim_reduction.TSNE_MAX_POINTS = original


if __name__ == "__main__":
    unittest.main()
