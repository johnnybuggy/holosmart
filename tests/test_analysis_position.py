"""Analysis batch position: worker signal + main-window status line.

Two levels, mirroring the established fixture styles:
- worker: AnalysisWorker.run() driven synchronously with a faked
  analyze_track — like tests/test_ui.py::TestAnalysisStop and
  tests/test_chunk_progress.py::TestWorkerChunkProgress;
- main window: offscreen MainWindow with a patched analyze_track and a
  bounded qWait loop — like
  tests/test_chunk_progress.py::TestMainWindowChunkProgress.

Covered here: the worker's track_position (position, total, filename)
signal — with skipped (missing) rows consuming no position — and the
main-window status line "Analyzing 3/12 — song.mp3 — <suffix>" for both
coarse steps and chunk ticks.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402


def _make_db_with_wavs(tmp, names: list[str]) -> tuple[Database, list[int]]:
    """Real temp Database with one small sine wav per *name*, indexed in order.

    The analysis pipeline is faked in every test here, so the wav content
    only has to exist — the rows must, though, or the worker skips them.
    """
    dir_path = Path(tmp.name)
    db = Database(dir_path / "lib.db")
    ids: list[int] = []
    with db.transaction() as conn:
        folder_id = repo.add_folder(conn, str(dir_path))
        for name in names:
            wav = dir_path / name
            sr = 8000
            t = np.linspace(0, 0.5, sr // 2, endpoint=False)
            sf.write(wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
            ids.append(repo.upsert_track(
                conn, folder_id, str(wav),
                {"filename": name, "extension": wav.suffix}))
    return db, ids


class _WorkerEvents(dict):
    """Event recorder wired to every AnalysisWorker signal."""

    def __init__(self) -> None:
        super().__init__(position=[], started=[], progress=[], chunk=[],
                         finished=[], stopped=[], all=[], failed=[])


def _wire(worker, ev: _WorkerEvents) -> None:
    worker.track_position.connect(
        lambda pos, tot, name: ev["position"].append((pos, tot, name)))
    worker.track_started.connect(
        lambda tid, path: ev["started"].append((tid, path)))
    worker.track_chunk_progress.connect(
        lambda tid, cur, tot, label: ev["chunk"].append((tid, cur, tot, label)))
    worker.track_progress.connect(
        lambda tid, msg: ev["progress"].append((tid, msg)))
    worker.track_finished.connect(
        lambda tid, ok, msg: ev["finished"].append((tid, ok, msg)))
    worker.stopped.connect(lambda count: ev["stopped"].append(count))
    worker.all_finished.connect(lambda: ev["all"].append(True))
    worker.failed.connect(lambda msg: ev["failed"].append(msg))


class TestWorkerTrackPosition(unittest.TestCase):
    """AnalysisWorker emits track_position (position, total, filename)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.ids = _make_db_with_wavs(
            self._tmp, ["a.wav", "b.wav", "c.wav"])
        self.db_path = Path(self._tmp.name) / "lib.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _make_worker(self, track_ids):
        from app.ui.workers import AnalysisWorker
        config = AppConfig()
        # These tests pin the sequential mode: they assert the exact position
        # order, which parallel starts (the new default of 2) do not
        # guarantee. Parallel ordering tolerance is covered by
        # tests/test_parallel_analysis.py.
        config.analysis_parallelism = 1
        return AnalysisWorker(self.db_path, config, list(track_ids))

    def _row(self, track_id):
        conn = self.db.connect()
        try:
            return repo.get_track(conn, track_id)
        finally:
            conn.close()

    def _expected_positions(self, batch: list[int]) -> list[tuple[int, int, str]]:
        """(1-based position, len(batch), filename) over the existing rows."""
        rows = [self._row(tid) for tid in self.ids]
        return [(i + 1, len(batch), Path(row["path"]).name)
                for i, row in enumerate(rows)]

    def test_position_counts_started_tracks(self):
        worker = self._make_worker(self.ids)
        ev = _WorkerEvents()
        _wire(worker, ev)

        def fake(db, track_id, config, progress_cb=None,
                force=False):
            notify = progress_cb or (lambda msg, cur=None, tot=None: None)
            notify("Decoding audio")
            with db.transaction() as conn:
                repo.set_track_status(conn, track_id, "analyzed", "fake done")

        with mock.patch("app.analysis.pipeline.analyze_track", fake):
            worker.run()  # synchronous, like TestAnalysisStop's fakes

        # (1, 3, 'a.wav'), (2, 3, 'b.wav'), (3, 3, 'c.wav')
        self.assertEqual(ev["position"], self._expected_positions(self.ids))
        # Every start is preceded by its position announcement, in order.
        self.assertEqual([tid for tid, _ in ev["started"]], self.ids)
        self.assertEqual(ev["failed"], [])

    def test_missing_row_does_not_consume_position(self):
        missing_id = max(self.ids) + 1000          # not in the database
        batch = [self.ids[0], missing_id, self.ids[1], self.ids[2]]
        worker = self._make_worker(batch)
        ev = _WorkerEvents()
        _wire(worker, ev)

        def fake(db, track_id, config, progress_cb=None,
                force=False):
            pass

        with mock.patch("app.analysis.pipeline.analyze_track", fake):
            worker.run()

        # The missing row is skipped silently: positions stay 1..3 over the
        # tracks that actually started, while the total still reflects the
        # requested batch (the nonexistent id included).
        self.assertEqual(ev["position"], self._expected_positions(batch))
        self.assertEqual([tid for tid, _ in ev["started"]], self.ids)
        self.assertEqual(ev["failed"], [])
        self.assertEqual(len(ev["finished"]), 3)
        self.assertEqual(len(ev["all"]), 1)

    def test_no_stop_run_does_not_emit_stopped(self):
        worker = self._make_worker(self.ids)
        ev = _WorkerEvents()
        _wire(worker, ev)

        def fake(db, track_id, config, progress_cb=None,
                force=False):
            with db.transaction() as conn:
                repo.set_track_status(conn, track_id, "analyzed", "fake done")

        with mock.patch("app.analysis.pipeline.analyze_track", fake):
            worker.run()

        self.assertEqual(ev["stopped"], [])        # stop path NOT taken
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual([ok for _, ok, _ in ev["finished"]], [True] * 3)
        for tid in self.ids:
            self.assertEqual(self._row(tid)["status"], "analyzed")


class TestMainWindowAnalysisPosition(unittest.TestCase):
    """The analysis status line carries "Analyzing pos/total — filename"."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.ids = _make_db_with_wavs(self._tmp, ["a.wav", "b.wav"])
        self.db_path = Path(self._tmp.name) / "lib.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _wait_for_status(self, win, expected: str, timeout: float = 5.0) -> None:
        """Bounded qWait loop until the status bar shows exactly *expected*."""
        from PySide6.QtTest import QTest

        deadline = time.time() + timeout
        while time.time() < deadline:
            if win.statusBar().currentMessage() == expected:
                return
            QTest.qWait(10)
        self.assertEqual(win.statusBar().currentMessage(), expected)

    def _wait_worker_done(self, worker) -> None:
        """Bounded qWait loop until the worker thread has stopped."""
        from PySide6.QtTest import QTest

        deadline = time.time() + 10.0
        while time.time() < deadline:
            QTest.qWait(20)
            if not worker.isRunning():
                return

    def test_status_line_shows_position_and_filename_during_run(self):
        from PySide6.QtTest import QTest

        from app.ui.main_window import MainWindow

        config = AppConfig()
        config.use_ollama = False
        config.analyze_wav = True    # these fixtures deliberately use .wav
        # Sequential mode: the exact "Analyzing 1/2 — a.wav" status line this
        # test waits for is only stable when track b cannot start meanwhile.
        config.analysis_parallelism = 1
        win = MainWindow(config, db_path=self.db_path)
        worker = None
        try:
            self.assertIsNone(win._analysis_position)   # unknown before a run
            self.assertIsNone(win._analysis_filename)
            in_track = threading.Event()
            release = threading.Event()

            def fake(db, track_id, cfg, progress_cb=None,
                force=False):
                notify = progress_cb or (lambda msg, cur=None, tot=None: None)
                notify("Decoding audio")     # coarse step only — no chunk tick
                in_track.set()
                # Hold the run until the UI side was asserted, so the status
                # line cannot be overwritten by the final messages.
                release.wait(5.0)

            during = "Analyzing 1/2 — a.wav — Decoding audio"
            with mock.patch("app.analysis.pipeline.analyze_track", fake):
                win.analyze_track_ids(self.ids)
                worker = win._analysis_worker
                self.assertIsNotNone(worker)
                self.assertTrue(in_track.wait(5.0))
                self._wait_for_status(win, during)

            # Let the fake finish and wait for the full run to complete.
            release.set()
            self._wait_for_status(win, "Analysis run complete.")
        finally:
            # Never leave the worker blocked in the fake, even on failure.
            release.set()
            if worker is not None:
                self._wait_worker_done(worker)
        self.assertIsNotNone(worker)
        self.assertFalse(worker.isRunning())   # clean finish, no hang
        win.close()

    def test_chunk_tick_status_carries_position_and_determinate_bar(self):
        from app.ui.main_window import MainWindow

        config = AppConfig()
        config.use_ollama = False
        config.analyze_wav = True    # these fixtures deliberately use .wav
        # Sequential mode: the exact "Analyzing 1/2 — a.wav" status line this
        # test waits for is only stable when track b cannot start meanwhile.
        config.analysis_parallelism = 1
        win = MainWindow(config, db_path=self.db_path)
        worker = None
        try:
            first_tick = threading.Event()
            release = threading.Event()

            def fake(db, track_id, cfg, progress_cb=None,
                force=False):
                notify = progress_cb or (lambda msg, cur=None, tot=None: None)
                notify("Decoding audio")
                notify("FakeChunk: chunk 1/4", 1, 4)   # tick: cur/tot set
                first_tick.set()
                release.wait(5.0)   # hold until the UI side was asserted

            during = "Analyzing 1/2 — a.wav — FakeChunk: chunk 1/4"
            with mock.patch("app.analysis.pipeline.analyze_track", fake):
                win.analyze_track_ids(self.ids)
                worker = win._analysis_worker
                self.assertIsNotNone(worker)
                self.assertTrue(first_tick.wait(5.0))
                # Poll the exact status line: the pre-analysis
                # "… Ns chunks …" message already contains the word "chunk",
                # so a substring check would be too loose.
                self._wait_for_status(win, during)
                # The bar stays determinate, driven by the tick (1/4).
                self.assertEqual(win._progress.minimum(), 0)
                self.assertEqual(win._progress.maximum(), 4)
                self.assertEqual(win._progress.value(), 1)

            # Let the run complete normally.
            release.set()
            self._wait_for_status(win, "Analysis run complete.")
        finally:
            # Never leave the worker blocked in the fake, even on failure.
            release.set()
            if worker is not None:
                self._wait_worker_done(worker)
        self.assertIsNotNone(worker)
        self.assertFalse(worker.isRunning())   # clean finish, no hang
        win.close()


if __name__ == "__main__":
    unittest.main()
