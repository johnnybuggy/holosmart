"""Interaction tests: scatter click-to-play and the chunk-tags dialog."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication, QDialog, QTableWidget

from app.db import repo
from app.db.database import Database


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class ScatterClickTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = _app()

    def test_click_emits_point_index(self) -> None:
        from PySide6.QtGui import QMouseEvent

        from app.ui.scatter_plot import ScatterCanvas

        canvas = ScatterCanvas()
        canvas.resize(400, 300)
        canvas.show()
        self.addCleanup(canvas.close)
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 1.0, 2.0])
        canvas.set_plot(xs, ys, labels=["a", "b", "c"])
        self._app.processEvents()
        seen: list[int] = []
        canvas.point_clicked.connect(seen.append)
        # click exactly on the second point's pixel position
        px, py = canvas._px[1]
        event = QMouseEvent(QEvent.Type.MouseButtonPress,
                            QPointF(px, py), QPointF(px, py),
                            Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(event)
        self.assertEqual(seen, [1])
        # a click in empty space emits nothing
        event = QMouseEvent(QEvent.Type.MouseButtonPress,
                            QPointF(10, 290), QPointF(10, 290),
                            Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(event)
        self.assertEqual(seen, [1])

    def test_dialog_maps_clicked_point_to_track_path(self) -> None:
        from app.ui import visualisation as viz

        db_path = Path(tempfile.mkdtemp()) / "v.db"
        db = Database(db_path)
        self.addCleanup(db_path.unlink)
        with db.transaction() as conn:
            folder = repo.add_folder(conn, "/music")
            track_id = repo.upsert_track(
                conn, folder, "/music/song.wav",
                {"filename": "song.wav", "extension": ".wav"})
            (cid,) = repo.replace_chunks(conn, track_id, [(0, 0.0, 10.0)])
            repo.add_chunk_embedding(
                conn, cid, "fft", np.ones(8, dtype=np.float32) / 4)

        dialog = viz.VisualisationDialog(db, None, None)
        self.addCleanup(dialog.close)
        dialog.refresh_and_plot()
        self._app.processEvents()
        self.assertGreater(dialog._canvas.point_count, 0)
        paths: list[str] = []
        dialog.play_track_path_requested.connect(paths.append)
        dialog._on_point_clicked(0)
        self.assertEqual(paths, ["/music/song.wav"])
        # out-of-range indices are ignored
        dialog._on_point_clicked(999)
        self.assertEqual(paths, ["/music/song.wav"])


class ChunkTagsDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._app = _app()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(Path(self._tmp.name) / "lib.db")
        with self.db.transaction() as conn:
            folder = repo.add_folder(conn, "/music")
            self.track_id = repo.upsert_track(
                conn, folder, "/music/song.wav",
                {"filename": "song.wav", "extension": ".wav"})
            ids = repo.replace_chunks(conn, self.track_id,
                                      [(0, 0.0, 10.0), (1, 10.0, 20.0)])
            self.chunk_id = ids[0]
            repo.add_chunk_tags(conn, self.chunk_id, "clap",
                                [("organ", 0.87), ("choir", 0.01)])
            repo.add_chunk_tags(conn, self.chunk_id, "muqlan",
                                [("a warm jazz loop", 1.0)])

    def _pane(self):
        from app.config import AppConfig
        from app.ui.detail_pane import DetailPane

        pane = DetailPane(self.db, AppConfig())
        pane.show()
        self.addCleanup(pane.close)
        pane.show_track(self.track_id)
        self._app.processEvents()
        return pane

    def test_double_click_non_model_cell_opens_all_tags(self) -> None:
        pane = self._pane()
        opened: list[int] = []
        with mock.patch.object(
                type(pane), "_show_chunk_tags_dialog",
                lambda self, chunk_id: opened.append(chunk_id)):
            # Tags column (3) is NOT a model column
            pane._on_chunk_cell_double_clicked(0, 3)
        self.assertEqual(opened, [self.chunk_id])

    def test_tags_dialog_lists_every_model_with_log10_scores(self) -> None:
        pane = self._pane()
        captured: list[QDialog] = []

        def fake_exec(dialog_self):
            captured.append(dialog_self)
            return QDialog.DialogCode.Accepted

        with mock.patch.object(QDialog, "exec", fake_exec):
            pane._show_chunk_tags_dialog(self.chunk_id)
        self.assertEqual(len(captured), 1)
        table = captured[0].findChildren(QTableWidget)[0]
        self.assertEqual(table.rowCount(), 3)
        # rows sorted best score first; log10 formatting of the weights
        rows = [(table.item(r, 1).text(), table.item(r, 2).text())
                for r in range(table.rowCount())]
        by_text = dict(rows)
        self.assertAlmostEqual(float(by_text["organ"]), math.log10(0.87),
                               places=2)
        self.assertAlmostEqual(float(by_text["choir"]), -2.0, places=2)
        self.assertAlmostEqual(float(by_text["a warm jazz loop"]), 0.0,
                               places=2)

    def test_empty_tags_dialog_stays_friendly(self) -> None:
        pane = self._pane()
        captured: list[QDialog] = []

        def fake_exec(dialog_self):
            captured.append(dialog_self)
            return QDialog.DialogCode.Accepted

        with mock.patch.object(QDialog, "exec", fake_exec):
            pane._show_chunk_tags_dialog(self.chunk_id + 1)   # no tags
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].findChildren(QTableWidget), [])


if __name__ == "__main__":
    unittest.main()
