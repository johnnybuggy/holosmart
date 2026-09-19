"""Tests for the Clear Analysis feature: repo cleanup function and the
main-window toolbar/context-menu flow.

The repo-level tests are plain unittest against a fresh SQLite file (no Qt
widgets involved); the UI-level tests run the real MainWindow offscreen
(QT_QPA_PLATFORM=offscreen, same fixture pattern as tests/test_ui.py).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402


class _TempDbTestCase(unittest.TestCase):
    """Base class for the repo-level tests: fresh Database + one connection."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="holosmart-test-clear-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.db = Database(self.tmp / "library.db")
        self.conn = self.db.connect()
        self.addCleanup(self.conn.close)

    def count(self, table: str, where: str = "", params: tuple = ()) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return int(self.conn.execute(sql, params).fetchone()["n"])


class ClearTrackAnalysisRepoTests(_TempDbTestCase):
    """repo.clear_track_analysis removes every analysis artefact of a track."""

    def _analyzed_track(self, folder: str, filename: str) -> int:
        """Insert one track carrying a full set of fake analysis results."""
        fid = repo.add_folder(self.conn, folder)
        tid = repo.upsert_track(self.conn, fid, f"{folder}/{filename}",
                                {"filename": filename, "duration_sec": 30.0})
        c0, c1 = repo.replace_chunks(
            self.conn, tid, [(0, 0.0, 10.0), (1, 10.0, 20.0)])
        repo.add_chunk_embedding(self.conn, c0, "clap",
                                 np.ones(4, dtype=np.float32))
        repo.add_chunk_embedding(self.conn, c1, "clap",
                                 np.linspace(0.0, 1.0, 4, dtype=np.float32))
        repo.add_chunk_tags(self.conn, c0, "clap", [("pop", 0.91), ("rock", 0.4)])
        repo.set_track_embedding(self.conn, tid, "clap",
                                 np.ones(4, dtype=np.float32))
        repo.set_track_embedding(self.conn, tid, "ollama:test",
                                 np.ones(4, dtype=np.float32))
        repo.set_track_description(self.conn, tid, "upbeat rock with piano")
        repo.set_track_status(self.conn, tid, "analyzed", "2 chunks")
        return tid

    def _assert_fully_cleared(self, tid: int) -> None:
        self.assertEqual(self.count("chunks", "track_id = ?", (tid,)), 0)
        self.assertEqual(self.count(
            "embeddings",
            "chunk_id IN (SELECT id FROM chunks WHERE track_id = ?)", (tid,)), 0)
        self.assertEqual(self.count(
            "chunk_tags",
            "chunk_id IN (SELECT id FROM chunks WHERE track_id = ?)", (tid,)), 0)
        self.assertEqual(
            self.count("track_embeddings", "track_id = ?", (tid,)), 0)
        row = repo.get_track(self.conn, tid)
        self.assertEqual(row["status"], "new")
        self.assertIsNone(row["status_message"])
        self.assertIsNone(row["description"])
        # re-analysis readiness: batch runs skip only status == 'analyzed'
        self.assertNotEqual(row["status"], "analyzed")

    def test_clear_track_analysis_removes_every_result(self):
        tid = self._analyzed_track("/music", "a.mp3")
        self.assertEqual(self.count("chunks", "track_id = ?", (tid,)), 2)
        self.assertEqual(
            self.count("track_embeddings", "track_id = ?", (tid,)), 2)

        repo.clear_track_analysis(self.conn, tid)

        self._assert_fully_cleared(tid)
        # last_analyzed_at is intentionally kept as a historical record
        self.assertIsNotNone(
            repo.get_track(self.conn, tid)["last_analyzed_at"])

    def test_clear_track_analysis_leaves_other_tracks_untouched(self):
        cleared = self._analyzed_track("/music", "a.mp3")
        kept = self._analyzed_track("/other", "b.mp3")

        repo.clear_track_analysis(self.conn, cleared)

        self._assert_fully_cleared(cleared)
        self.assertEqual(self.count("chunks", "track_id = ?", (kept,)), 2)
        self.assertEqual(
            self.count("track_embeddings", "track_id = ?", (kept,)), 2)
        row = repo.get_track(self.conn, kept)
        self.assertEqual(row["status"], "analyzed")
        self.assertEqual(row["description"], "upbeat rock with piano")

    def test_clear_track_analysis_works_without_foreign_keys_pragma(self):
        """The explicit sub-select deletes do not rely on ON DELETE CASCADE."""
        tid = self._analyzed_track("/music", "a.mp3")
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys = OFF")   # cascades now dormant
        try:
            repo.clear_track_analysis(self.conn, tid)
            self.conn.commit()
        finally:
            self.conn.execute("PRAGMA foreign_keys = ON")
        self._assert_fully_cleared(tid)


class _FakeMenu:
    """Stands in for QMenu when driving _show_context_menu headlessly.

    Keeps every added action so tests can assert the menu order; ``exec``
    pretends the user clicked the "Clear analysis results" entry.
    """

    last: _FakeMenu | None = None

    def __init__(self, *args, **kwargs) -> None:
        self._actions: list = []
        _FakeMenu.last = self

    def addAction(self, action) -> None:
        self._actions.append(action)

    def addSeparator(self) -> None:
        self._actions.append(None)   # sentinel: separator position marker

    def exec(self, pos):
        for action in self._actions:
            if action is not None and action.text() == "Clear analysis results":
                return action
        return None


class UiTestBase(unittest.TestCase):
    """Fixture pattern of tests/test_ui.py: temp dir, real wav, real Database."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.db_path = self.dir / "lib.db"
        self.db = Database(self.db_path)

        self.wav = self.dir / "song.wav"
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(self.wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)

        with self.db.transaction() as conn:
            self.folder_id = repo.add_folder(conn, str(self.dir))
            self.track_id = repo.upsert_track(
                conn, self.folder_id, str(self.wav),
                {"filename": "song.wav", "extension": ".wav",
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": self.wav.stat().st_size})


class ClearAnalysisUiTests(UiTestBase):
    """Offscreen MainWindow tests for the Clear Analysis toolbar action."""

    def _add_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": path.stat().st_size})

    def _fake_results(self, track_id: int) -> None:
        """Persist a small fake analysis result set for *track_id*."""
        with self.db.transaction() as conn:
            (cid,) = repo.replace_chunks(conn, track_id, [(0, 0.0, 0.5)])
            repo.add_chunk_embedding(conn, cid, "clap",
                                     np.ones(8, dtype=np.float32))
            repo.add_chunk_tags(conn, cid, "clap", [("rock", 0.9)])
            repo.set_track_embedding(conn, track_id, "clap",
                                     np.ones(8, dtype=np.float32))
            repo.set_track_embedding(conn, track_id, "ollama:test",
                                     np.ones(8, dtype=np.float32))
            repo.set_track_description(conn, track_id, "upbeat rock")
            repo.set_track_status(conn, track_id, "analyzed", "fake done")

    def _window(self):
        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        config.analyze_wav = True   # these fixtures deliberately use .wav
        win = MainWindow(config, db_path=self.db_path)
        self.addCleanup(win.close)
        return win

    def _row(self, track_id: int):
        conn = self.db.connect()
        try:
            return repo.get_track(conn, track_id)
        finally:
            conn.close()

    def _chunk_count(self, track_id: int) -> int:
        conn = self.db.connect()
        try:
            return int(conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE track_id = ?",
                (track_id,)).fetchone()[0])
        finally:
            conn.close()

    def _track_item(self, tree, track_id: int):
        from app.ui.folder_tree import TRACK_ROLE
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.data(0, TRACK_ROLE) == track_id:
                return item
            stack.extend(item.child(i) for i in range(item.childCount()))
        return None

    def test_clear_analysis_results_clears_selected_track_only(self):
        second_id = self._add_track("sub/second.wav")
        self._fake_results(self.track_id)
        self._fake_results(second_id)
        win = self._window()
        win._tree.select_track(self.track_id)
        self.assertEqual(win._details.current_track_id(), self.track_id)

        yes = QMessageBox.StandardButton.Yes
        with mock.patch("app.ui.main_window.QMessageBox.question",
                        return_value=yes) as question:
            win.clear_analysis_results()

        question.assert_called_once()
        self.assertIn("1 track(s)", question.call_args.args[2])
        # the selected track lost every analysis artefact
        row = self._row(self.track_id)
        self.assertEqual(row["status"], "new")
        self.assertIsNone(row["status_message"])
        self.assertIsNone(row["description"])
        self.assertEqual(self._chunk_count(self.track_id), 0)
        # the other track is untouched
        other = self._row(second_id)
        self.assertEqual(other["status"], "analyzed")
        self.assertEqual(other["description"], "upbeat rock")
        self.assertEqual(self._chunk_count(second_id), 1)
        # status bar, tree status columns and details pane reflect the clear
        self.assertIn("Cleared analysis results for 1 track(s).",
                      win.statusBar().currentMessage())
        self.assertEqual(win._tree.selected_track_id(), self.track_id)
        # the cleared track's row shows no glyph/status anywhere (its model
        # column is empty again); the untouched track keeps its check mark
        cleared = self._track_item(win._tree, self.track_id)
        self.assertEqual(cleared.text(1), "")
        self.assertTrue(cleared.toolTip(0).splitlines()[-1].startswith("new"))
        self.assertEqual(self._track_item(win._tree, second_id).text(1), "✓")
        root = win._tree.topLevelItem(0)
        self.assertEqual(root.text(1), "50%")   # CLAP column, 1 of 2 files
        self.assertEqual(win._details._chunks_table.rowCount(), 0)

    def test_clear_analysis_declined_keeps_everything(self):
        self._fake_results(self.track_id)
        win = self._window()
        win._tree.select_track(self.track_id)

        no = QMessageBox.StandardButton.No
        with mock.patch("app.ui.main_window.QMessageBox.question",
                        return_value=no) as question:
            win.clear_analysis_results()

        question.assert_called_once()
        row = self._row(self.track_id)
        self.assertEqual(row["status"], "analyzed")
        self.assertEqual(row["description"], "upbeat rock")
        self.assertEqual(self._chunk_count(self.track_id), 1)
        self.assertNotIn("Cleared", win.statusBar().currentMessage())

    def test_clear_analysis_without_selection_is_a_noop(self):
        win = self._window()   # nothing selected right after construction
        self.assertIsNone(win._tree.selected_track_id())
        with mock.patch("app.ui.main_window.QMessageBox.question") as question:
            win.clear_analysis_results()
        question.assert_not_called()
        self.assertIn("Select a track or folder to clear.",
                      win.statusBar().currentMessage())

    def test_clear_analysis_refuses_while_analysis_is_running(self):
        self._fake_results(self.track_id)
        win = self._window()
        win._tree.select_track(self.track_id)

        class _RunningWorker:
            def isRunning(self) -> bool:
                return True

            def wait(self, msecs: int = 0) -> bool:   # closeEvent() may call
                return True

        win._analysis_worker = _RunningWorker()
        with mock.patch("app.ui.main_window.QMessageBox.question") as question:
            win.clear_analysis_results()

        question.assert_not_called()
        self.assertIn("Stop the running analysis",
                      win.statusBar().currentMessage())
        row = self._row(self.track_id)
        self.assertEqual(row["status"], "analyzed")
        self.assertEqual(self._chunk_count(self.track_id), 1)

    def test_clear_action_disabled_while_busy(self):
        from PySide6.QtWidgets import QToolBar
        win = self._window()
        self.assertEqual(win._act_clear.text(), "Clear Analysis")
        self.assertIn("Clear analysis results for the selected file or folder",
                      win._act_clear.toolTip())
        bar = win.findChild(QToolBar)
        texts = [action.text() for action in bar.actions()]
        # placed between "Analyze All" and "Stop Analysis"
        self.assertLess(texts.index("Analyze All"),
                        texts.index("Clear Analysis"))
        self.assertLess(texts.index("Clear Analysis"),
                        texts.index("Stop Analysis"))

        class _IdleWorker:
            def isRunning(self) -> bool:
                return False

            def wait(self, msecs: int = 0) -> bool:   # closeEvent() may call
                return True

        win._analysis_worker = _IdleWorker()
        win._set_busy(True)
        self.assertFalse(win._act_clear.isEnabled())
        win._set_busy(False)
        self.assertTrue(win._act_clear.isEnabled())

    def test_context_menu_clear_dispatches_to_main_window(self):
        from app.ui import folder_tree
        from app.ui.main_window import MainWindow
        self.assertTrue(hasattr(MainWindow, "clear_analysis_results"))
        second_id = self._add_track("sub/second.wav")
        self._fake_results(self.track_id)
        self._fake_results(second_id)
        win = self._window()
        win._tree.select_track(self.track_id)

        with mock.patch.object(folder_tree, "QMenu", _FakeMenu), \
                mock.patch("app.ui.main_window.QMessageBox.question",
                           return_value=QMessageBox.StandardButton.Yes):
            win._tree._show_context_menu(QPoint(4, 4))

        # menu order: Analyze, Find similar…, |, Clear analysis results, |,
        # Reveal in Finder
        menu = _FakeMenu.last
        self.assertEqual(
            [None if action is None else action.text()
             for action in menu._actions],
            ["Analyze", "Find similar…", None,
             "Clear analysis results", None, "Reveal in Finder"])
        row = self._row(self.track_id)
        self.assertEqual(row["status"], "new")
        self.assertEqual(self._chunk_count(self.track_id), 0)
        self.assertEqual(self._chunk_count(second_id), 1)
        self.assertIn("Cleared analysis results for 1 track(s).",
                      win.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
