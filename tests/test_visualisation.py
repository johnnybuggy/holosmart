"""Tests for the Visualisation dialog and its QPainter scatter canvas."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.db.database import Database  # noqa: E402
from app.db import repo  # noqa: E402


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _VizDbTestCase(unittest.TestCase):
    """Temp database with two tracks and fft (+ partial mert) vectors."""

    def setUp(self) -> None:
        self._app = _app()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(Path(self._tmp.name) / "lib.db")
        with self.db.transaction() as conn:
            fid = repo.add_folder(conn, str(Path(self._tmp.name) / "music"))
            self.t1 = repo.upsert_track(conn, fid, "/music/a.wav", {
                "filename": "a.wav", "extension": ".wav", "codec": "pcm_s16le",
                "sample_rate": 8000, "channels": 1, "duration_sec": 20.0,
                "size_bytes": 100})
            self.t2 = repo.upsert_track(conn, fid, "/music/b.wav", {
                "filename": "b.wav", "extension": ".wav", "codec": "pcm_s16le",
                "sample_rate": 8000, "channels": 1, "duration_sec": 20.0,
                "size_bytes": 100})
            self.c1 = repo.replace_chunks(
                conn, self.t1, [(0, 0.0, 10.0), (1, 10.0, 20.0)])
            self.c2 = repo.replace_chunks(
                conn, self.t2, [(0, 0.0, 10.0), (1, 10.0, 20.0)])
            # fft (dim 4) on all chunks; mert (dim 3) only on track 1.
            for base, chunk_ids in ((0.0, self.c1), (100.0, self.c2)):
                for idx, chunk_id in enumerate(chunk_ids):
                    vec = np.arange(4, dtype=np.float32) + base + idx
                    repo.add_chunk_embedding(conn, chunk_id, "fft", vec)
            for chunk_id in self.c1:
                repo.add_chunk_embedding(conn, chunk_id, "mert",
                                         np.array([.5, -.5, 1.0],
                                                  dtype=np.float32))
        self._track_colors = {}

    def make_dialog(self):
        from app.config import AppConfig
        from app.ui.visualisation import VisualisationDialog

        dialog = VisualisationDialog(self.db, AppConfig())
        self.addCleanup(dialog.close)
        return dialog


class RepoChunkEmbeddingRowsTests(_VizDbTestCase):
    def test_rows_are_joined_with_track_and_chunk_info(self) -> None:
        with self.db.transaction() as conn:
            rows = repo.get_chunk_embedding_rows(conn)
        # 4 fft rows + 2 mert rows.
        self.assertEqual(len(rows), 6)
        by_model = {}
        for row in rows:
            by_model.setdefault(row["model"], []).append(row)
        self.assertEqual(sorted(by_model), ["fft", "mert"])
        fft = by_model["fft"]
        self.assertEqual([r["chunk_idx"] for r in fft], [0, 1, 0, 1])
        self.assertEqual(fft[0]["track_filename"], "a.wav")
        self.assertEqual(fft[0]["track_path"], "/music/a.wav")
        self.assertAlmostEqual(fft[0]["start_sec"], 0.0)
        self.assertEqual(fft[0]["vec"].size, 4)
        self.assertEqual(fft[0]["dim"], 4)

    def test_rows_can_be_filtered_by_model(self) -> None:
        with self.db.transaction() as conn:
            rows = repo.get_chunk_embedding_rows(conn, models=["mert"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["model"] == "mert" for r in rows))


class VisualisationDialogTests(_VizDbTestCase):
    def test_raw_component_plot(self) -> None:
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        # fft is the only model with data on every chunk; both combos show it.
        self.assertEqual(dialog._x_model.currentData(), "fft")
        self.assertEqual(dialog._y_model.currentData(), "fft")
        self.assertEqual(dialog._canvas.point_count, 4)
        self.assertEqual(dialog._canvas._x_label, "FFT[0]")
        self.assertEqual(dialog._canvas._y_label, "FFT[1]")
        self.assertIn("4 chunks", dialog._status_label.text())

    def test_cross_model_raw_plot(self) -> None:
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        # X = fft[0], Y = mert[0] → only the chunks carrying BOTH models.
        dialog._y_model.setCurrentIndex(dialog._y_model.findData("mert"))
        dialog._y_comp.setValue(0)
        dialog._plot()
        self.assertEqual(dialog._canvas.point_count, 2)   # track 1 only
        self.assertEqual(dialog._canvas._y_label, "MERT[0]")
        self.assertIn("MERT", dialog._status_label.text())

    def test_component_range_follows_model_dim(self) -> None:
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog._y_model.setCurrentIndex(dialog._y_model.findData("mert"))
        dialog._sync_component_range()
        self.assertEqual(dialog._y_comp.maximum(), 2)     # dim 3 → 0..2
        dialog._y_model.setCurrentIndex(dialog._y_model.findData("fft"))
        dialog._sync_component_range()
        self.assertEqual(dialog._x_comp.maximum(), 3)     # dim 4 → 0..3

    def test_plot_click_changes_the_drawn_points(self) -> None:
        """The reported bug: labels updated but the drawn point cloud never
        changed — set_plot left stale cached QPolygons in place while the
        widget size stayed the same.  Components with genuinely different
        point patterns must produce different pixels."""
        # three chunks on one track: comp0=[0,5,10], comp3=[0,10,5] — same
        # axis range, DIFFERENT pattern (auto-scaling would hide the
        # difference for plain ramps).
        with self.db.transaction() as conn:
            c3 = repo.replace_chunks(
                conn, self.t1, [(0, 0.0, 10.0), (1, 10.0, 20.0),
                                (2, 20.0, 30.0)])
            for chunk_id, x3 in zip(c3, (0.0, 10.0, 5.0)):
                repo.add_chunk_embedding(conn, chunk_id, "fft",
                                         np.array([0.0, 7.0, 0.0, x3],
                                                  dtype=np.float32))
        dialog = self.make_dialog()
        dialog.show()
        self.addCleanup(dialog.hide)
        QApplication.processEvents()
        dialog.refresh_and_plot()
        dialog._canvas.repaint()           # force a real paint pass
        self.assertIsNotNone(dialog._canvas._polys)
        before = {cid: [(p.x(), p.y()) for p in poly]
                  for cid, poly in dialog._canvas._polys.items()}

        dialog._x_comp.setValue(3)         # FFT[0] → FFT[3]
        dialog._plot()
        dialog._canvas.repaint()
        after = {cid: [(p.x(), p.y()) for p in poly]
                 for cid, poly in dialog._canvas._polys.items()}

        self.assertNotEqual(before, after)   # the DRAWN points changed
        self.assertTrue(dialog._canvas._x_label.endswith("[3]"))

    def test_canvas_rebuilds_polygons_for_new_data(self) -> None:
        """Canvas-level: set_plot invalidates the polygon cache, so the next
        paint shows the new data even at an unchanged widget size."""
        from app.ui.scatter_plot import ScatterCanvas

        canvas = ScatterCanvas()
        canvas.resize(600, 400)
        canvas.show()
        self.addCleanup(canvas.hide)
        QApplication.processEvents()        # realize the widget on offscreen

        def drawn():
            # repaint() paints synchronously, but ONLY when shown
            canvas.repaint()
            self.assertIsNotNone(canvas._polys)
            return [(p.x(), p.y()) for p in canvas._polys[0]]

        canvas.set_plot(np.array([0.0, 0.0, 0.0]),
                        np.array([0.0, 5.0, 10.0]))
        first = drawn()
        # same axis range, different pattern — pixels must differ
        canvas.set_plot(np.array([0.0, 0.0, 0.0]),
                        np.array([0.0, 10.0, 5.0]))
        second = drawn()
        self.assertNotEqual(first, second)
        # and clearing works too: empty plot paints the note, no crash
        canvas.clear_plot("gone")
        canvas.repaint()
        self.assertEqual(canvas.point_count, 0)

    def test_parameter_changes_update_the_plot_live(self) -> None:
        """Changing X/Y components re-plots without pressing Plot (debounced
        live updates — instant methods only)."""
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog.show()
        self.addCleanup(dialog.hide)

        # user-driven spin change schedules a debounced re-plot …
        dialog._x_comp.setValue(2)
        self.assertTrue(dialog._replot_timer.isActive())
        dialog._replot_timer.stop()
        dialog._plot()
        self.assertEqual(dialog._canvas._x_label, "FFT[2]")

        # … and firing the debounce re-plots to the changed parameter
        dialog._x_comp.setValue(3)
        dialog._replot_timer.stop()          # control the timer directly
        dialog._auto_plot()
        self.assertEqual(dialog._canvas._x_label, "FFT[3]")

        # model switch: the component range follows, the component value is
        # preserved (still valid for MERT's dim 3) and the plot re-derives
        dialog._y_model.setCurrentIndex(dialog._y_model.findData("mert"))
        self.assertEqual(dialog._y_comp.maximum(), 2)
        self.assertEqual(dialog._y_comp.value(), 1)
        dialog._replot_timer.stop()
        dialog._auto_plot()
        self.assertEqual(dialog._canvas._y_label, "MERT[1]")

        # reduction switch to PCA: instant too
        dialog._reduce_combo.setCurrentIndex(
            dialog._reduce_combo.findData("pca"))
        dialog._replot_timer.stop()
        dialog._auto_plot()
        self.assertTrue(dialog._canvas._x_label.startswith("PC1"))

    def test_tsne_change_waits_for_the_plot_button(self) -> None:
        """Slow methods never auto-run; the status line tells the user."""
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog.show()
        self.addCleanup(dialog.hide)
        index = dialog._reduce_combo.findData("tsne")
        if dialog._reduce_combo.model().item(index).isEnabled():
            self.skipTest("scikit-learn installed in this environment")
        dialog._reduce_combo.setCurrentIndex(index)
        dialog._replot_timer.stop()
        dialog._auto_plot()
        self.assertIn("press Plot", dialog._status_label.text())

    def test_restore_does_not_schedule_a_replot(self) -> None:
        """Programmatic restores (reload path) stay signal-silent — no
        double-plot after a Plot/refresh click."""
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        self.assertFalse(dialog._replot_timer.isActive())
        # a reload with preserved selections must not arm the timer either
        dialog._x_comp.setValue(3)
        dialog._replot_timer.stop()
        dialog.refresh_and_plot()
        self.assertFalse(dialog._replot_timer.isActive())
        self.assertEqual(dialog._x_comp.value(), 3)

    def test_selections_survive_a_plot_click(self) -> None:
        """The reported bug: clicking "Plot" after changing components reset
        everything to defaults instead of plotting the new selection."""
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        # user picks X = fft[3], Y = fft[2] and unchecks MERT
        dialog._x_comp.setValue(3)
        dialog._y_comp.setValue(2)
        dialog._reduce_checks["mert"].setChecked(False)
        dialog.refresh_and_plot()          # the "Plot" button path

        # the selection is still in place …
        self.assertEqual(dialog._x_model.currentData(), "fft")
        self.assertEqual(dialog._y_model.currentData(), "fft")
        self.assertEqual(dialog._x_comp.value(), 3)
        self.assertEqual(dialog._y_comp.value(), 2)
        self.assertFalse(dialog._reduce_checks["mert"].isChecked())
        # … and the plot shows the CHOSEN components, not the defaults
        self.assertEqual(dialog._canvas._x_label, "FFT[3]")
        self.assertEqual(dialog._canvas._y_label, "FFT[2]")

    def test_selections_survive_after_new_data_appears(self) -> None:
        """Analyze more tracks mid-session: models/components stay put."""
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog._x_comp.setValue(2)
        dialog._y_model.setCurrentIndex(dialog._y_model.findData("mert"))
        dialog._y_comp.setValue(1)
        # new analysis results land in the database
        with self.db.transaction() as conn:
            c3 = repo.replace_chunks(conn, self.t2, [(2, 20.0, 30.0)])
            repo.add_chunk_embedding(conn, c3[0], "mert",
                                     np.array([.1, .2, .3],
                                              dtype=np.float32))
        dialog.refresh_and_plot()
        self.assertEqual(dialog._x_model.currentData(), "fft")
        self.assertEqual(dialog._x_comp.value(), 2)
        self.assertEqual(dialog._y_model.currentData(), "mert")
        self.assertEqual(dialog._y_comp.value(), 1)

    def test_selection_clamps_when_model_data_disappears(self) -> None:
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog._x_comp.setValue(3)
        dialog._y_model.setCurrentIndex(dialog._y_model.findData("mert"))
        # all MERT vectors vanish (e.g. analysis cleared)
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM embeddings WHERE model = 'mert'")
        dialog.refresh_and_plot()
        # Y falls back to the first available model, X keeps fft + comp
        self.assertEqual(dialog._x_model.currentData(), "fft")
        self.assertEqual(dialog._x_comp.value(), 3)
        self.assertEqual(dialog._y_model.currentData(), "fft")
        self.assertLessEqual(dialog._y_comp.value(),
                             dialog._y_comp.maximum())
        self.assertFalse(dialog._reduce_checks["mert"].isEnabled())
        self.assertEqual(dialog._canvas.point_count, 4)

    def test_pca_reduction_plot(self) -> None:
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog._reduce_combo.setCurrentIndex(
            dialog._reduce_combo.findData("pca"))
        dialog._plot()
        # Both data-bearing models are checked by default, so only the
        # chunks carrying BOTH models (track 1) are projected.
        self.assertEqual(dialog._canvas.point_count, 2)
        self.assertTrue(dialog._canvas._x_label.startswith("PC1"))
        self.assertIn("PCA", dialog._status_label.text())
        # Points really were reduced: both axes carry the PCA labels.
        self.assertTrue(dialog._canvas._y_label.startswith("PC2"))

    def test_pca_rejects_empty_model_selection(self) -> None:
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog._reduce_combo.setCurrentIndex(
            dialog._reduce_combo.findData("pca"))
        for check in dialog._reduce_checks.values():
            check.setChecked(False)
        dialog._plot()
        self.assertEqual(dialog._canvas.point_count, 0)
        self.assertIn("at least one model", dialog._status_label.text())

    def test_reduction_with_partially_covered_models(self) -> None:
        # mert covers only track 1's chunks; selecting both models must
        # restrict the plot to the chunks that carry both.
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        dialog._reduce_combo.setCurrentIndex(
            dialog._reduce_combo.findData("pca"))
        dialog._plot()          # both enabled models are checked by default
        self.assertEqual(dialog._canvas.point_count, 2)
        self.assertIn("MERT + FFT", dialog._status_label.text())

    def test_tsne_selected_without_library_reports_hint(self) -> None:
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        index = dialog._reduce_combo.findData("tsne")
        if dialog._reduce_combo.model().item(index).isEnabled():
            self.skipTest("scikit-learn installed in this environment")
        dialog._reduce_combo.setCurrentIndex(index)
        dialog.refresh_and_plot()      # method stays on the (disabled) item
        dialog._plot()
        self.assertIn("pip install", dialog._status_label.text())

    def test_no_data_shows_hint(self) -> None:
        with self.db.transaction() as conn:
            repo.clear_track_analysis(conn, self.t1)
            repo.clear_track_analysis(conn, self.t2)
        dialog = self.make_dialog()
        dialog.refresh_and_plot()
        self.assertEqual(dialog._canvas.point_count, 0)
        self.assertIn("Analyze", dialog._status_label.text())

    def test_subsample_is_deterministic_and_annotated(self) -> None:
        from app.ui.visualisation import VisualisationDialog

        n, cap = 10, 4
        keep1 = VisualisationDialog._subsample_indices(n, cap)
        keep2 = VisualisationDialog._subsample_indices(n, cap)
        np.testing.assert_array_equal(keep1, keep2)
        self.assertEqual(keep1.size, cap)

    def test_raw_plot_subsamples_before_building_labels(self) -> None:
        """Rows are thinned to the cap first: hover labels exist only for
        the plotted points (keeps re-plots fast on real libraries)."""
        from unittest import mock

        from app.ui.visualisation import RAW_MAX_POINTS as _cap

        dialog = self.make_dialog()
        with mock.patch("app.ui.visualisation.RAW_MAX_POINTS",
                        min(_cap, 4)):
            dialog.refresh_and_plot()
        # 4 fft rows in the fixture > cap 4? No: cap == 4 → no subsample.
        # Drop one chunk to force the cap strictly below the row count.
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM embeddings WHERE model = 'fft' "
                         "AND chunk_id = ?", (self.c1[1],))
        with mock.patch("app.ui.visualisation.RAW_MAX_POINTS", 2):
            dialog.refresh_and_plot()
        self.assertEqual(dialog._canvas.point_count, 2)
        self.assertEqual(len(dialog._canvas._labels), 2)
        self.assertIn("subsampled 2 of 3", dialog._status_label.text())


class ScatterCanvasTests(_VizDbTestCase):
    def test_plot_and_paint(self) -> None:
        from app.ui.scatter_plot import ScatterCanvas

        canvas = ScatterCanvas()
        canvas.resize(400, 300)
        canvas.show()
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.array([0.0, 5.0, -5.0, 0.5])
        canvas.set_plot(xs, ys, "x axis", "y axis",
                        labels=[f"p{i}" for i in range(4)],
                        colors=[0, 1, 0, 1], status="note")
        self.assertEqual(canvas.point_count, 4)
        canvas.grab()          # forces a paint pass; must not raise
        self.assertIsNotNone(canvas._px)
        self.assertEqual(canvas._px.shape, (4, 2))

    def test_flat_and_empty_data(self) -> None:
        from app.ui.scatter_plot import ScatterCanvas

        canvas = ScatterCanvas()
        canvas.resize(400, 300)
        canvas.show()
        # All identical values (flat axis) and an empty plot must paint.
        canvas.set_plot([2.0, 2.0], [7.0, 7.0], "x", "y")
        canvas.grab()
        canvas.clear_plot("nothing here")
        self.assertEqual(canvas.point_count, 0)
        canvas.grab()

    def test_length_mismatch_raises(self) -> None:
        from app.ui.scatter_plot import ScatterCanvas

        canvas = ScatterCanvas()
        with self.assertRaises(ValueError):
            canvas.set_plot([1.0, 2.0], [1.0], "x", "y")


if __name__ == "__main__":
    unittest.main()
