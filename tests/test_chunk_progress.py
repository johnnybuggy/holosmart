"""Per-chunk analysis progress: pipeline ticks, worker signals, main-window bar.

Three levels, mirroring the existing fixture styles:
- pipeline: real temp Database + generated 8 kHz mono sine wav + a fake plugin
  with a small ``batch_size`` (no torch, no GUI) — like tests/test_pipeline.py;
- worker: AnalysisWorker.run() driven synchronously with a faked
  analyze_track — like tests/test_ui.py::TestAnalysisStop;
- main window: offscreen MainWindow with a patched analyze_track and a bounded
  qWait loop — like tests/test_ui.py::TestMainWindow.
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


def _make_db_with_wav(tmp) -> tuple[Database, int]:
    """Real temp Database + a 2.0 s 8 kHz mono sine wav, indexed as one track.

    With chunk_seconds=0.5 / overlap 0 this yields exactly 4 chunks.
    """
    dir_path = Path(tmp.name)
    db = Database(dir_path / "lib.db")
    wav = dir_path / "song.wav"
    sr = 8000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sf.write(wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
    with db.transaction() as conn:
        folder_id = repo.add_folder(conn, str(dir_path))
        track_id = repo.upsert_track(
            conn, folder_id, str(wav),
            {"filename": "song.wav", "extension": ".wav"})
    return db, track_id


class FakeChunkPlugin:
    """Plugin stand-in with batch_size=2: records embed/describe batch sizes.

    Plain object (not a ModelPlugin subclass) — the pipeline only relies on
    the duck-typed surface: name/display_name/provides_text/batch_size/
    is_available/availability_error/embed/describe (+ apply_config).
    """

    name = "fakechunk"
    display_name = "FakeChunk"
    embedding_dim = 4
    provides_text = False
    preferred_sample_rate = 48000
    requirements = ()
    batch_size = 2
    tag_top_k = 5

    def __init__(self, provides_text: bool = False,
                 fail_on_embed_batch: int | None = None,
                 name: str = "fakechunk") -> None:
        self.name = name
        self.provides_text = provides_text
        self.fail_on_embed_batch = fail_on_embed_batch
        self.embed_batches: list[int] = []
        self.describe_batches: list[int] = []
        self.applied_configs: list = []

    def is_available(self) -> bool:
        return True

    def availability_error(self):
        return None

    def apply_config(self, config) -> None:
        self.applied_configs.append(config)

    def embed(self, chunks, sr):
        self.embed_batches.append(len(chunks))
        if (self.fail_on_embed_batch is not None
                and len(self.embed_batches) == self.fail_on_embed_batch):
            raise RuntimeError("boom on batch")
        return [np.array([0.1, 1.0, -1.0, 0.5], dtype=np.float32)
                for _ in chunks]

    def describe(self, chunks, sr, top_k=5):
        self.describe_batches.append(len(chunks))
        return [[("tagA", 0.9), ("tagB", 0.1)] for _ in chunks]


def _notify_log():
    """Record every notify(msg, cur, tot) call the pipeline makes."""
    calls: list[tuple[str, int | None, int | None]] = []

    def notify(msg, cur=None, tot=None):
        calls.append((msg, cur, tot))

    return calls, notify


class TestPipelineChunkProgress(unittest.TestCase):
    """_run_plugin batching reports real chunk progress via notify(cur, tot)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.track_id = _make_db_with_wav(self._tmp)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _config() -> AppConfig:
        cfg = AppConfig()
        cfg.models = ["fakechunk"]
        cfg.use_ollama = False
        cfg.chunk_seconds = 0.5   # 2.0 s wav -> 4 chunks
        cfg.overlap_percent = 0.0
        return cfg

    def test_embed_and_tag_batches_report_chunk_progress(self):
        import app.analysis.pipeline as pipeline

        plugin = FakeChunkPlugin(provides_text=True)
        calls, notify = _notify_log()
        with mock.patch.object(pipeline, "get_plugin", lambda name: plugin):
            pipeline.analyze_track(self.db, self.track_id, self._config(),
                                   progress_cb=notify)

        # Embed phase: strictly increasing 2,4 chunk ticks with tot == N.
        chunk_ticks = [(msg, cur, tot) for msg, cur, tot in calls
                       if "chunk" in msg]
        self.assertEqual([cur for _, cur, _ in chunk_ticks], [2, 4])
        self.assertTrue(all(tot == 4 for _, _, tot in chunk_ticks))
        for msg, _, _ in chunk_ticks:
            self.assertIn("FakeChunk", msg)
            self.assertIn("chunk", msg)
        self.assertEqual(plugin.embed_batches, [2, 2])  # really batched

        # Tags phase: same pattern with "tags" messages.
        tag_ticks = [(msg, cur, tot) for msg, cur, tot in calls
                     if "tags" in msg]
        self.assertEqual([cur for _, cur, _ in tag_ticks], [2, 4])
        self.assertTrue(all(tot == 4 for _, _, tot in tag_ticks))
        for msg, _, _ in tag_ticks:
            self.assertIn("FakeChunk", msg)
            self.assertIn("tags", msg)
        self.assertEqual(plugin.describe_batches, [2, 2])

        # Coarse steps and final completion are still announced unchanged.
        self.assertEqual(calls[0], ("Decoding audio", None, None))
        self.assertTrue(any(m == "Running FakeChunk" for m, _, _ in calls))
        self.assertIn(("Analysis complete", 4, 4), calls)

        # apply_config still applied exactly once before the plugin ran.
        self.assertEqual(len(plugin.applied_configs), 1)

        # End state: analyzed, every chunk embedded, all tags stored.
        with self.db.transaction() as conn:
            track = repo.get_track(conn, self.track_id)
            self.assertEqual(track["status"], "analyzed")
            chunks = repo.get_chunks(conn, self.track_id)
            self.assertEqual(len(chunks), 4)
            for c in chunks:
                rows = repo.get_chunk_embeddings(conn, c["id"])
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["model"], "fakechunk")
                self.assertEqual(rows[0]["dim"], 4)
            tags = repo.get_track_tags(conn, self.track_id)
            self.assertEqual(len(tags), 8)  # 4 chunks x 2 tags

    def test_incremental_run_keeps_other_models_results(self) -> None:
        """Re-analysis with a NEW model must not wipe existing vectors."""
        import app.analysis.pipeline as pipeline

        first = FakeChunkPlugin(name="fakechunk", provides_text=True)
        second = FakeChunkPlugin(name="secondmodel")
        with mock.patch.object(
                pipeline, "get_plugin",
                lambda name: {"fakechunk": first,
                              "secondmodel": second}[name]):
            pipeline.analyze_track(self.db, self.track_id, self._config())
            # snapshot: chunk ids + first model's stored vectors
            with self.db.transaction() as conn:
                chunks_before = repo.get_chunks(conn, self.track_id)
                vectors_before = {int(c["id"]):
                                  repo.get_chunk_embeddings(conn, c["id"])
                                  for c in chunks_before}
            # second run adds one more model
            cfg = self._config()
            cfg.models = ["fakechunk", "secondmodel"]
            pipeline.analyze_track(self.db, self.track_id, cfg)
        with self.db.transaction() as conn:
            chunks_after = repo.get_chunks(conn, self.track_id)
        self.assertEqual([int(c["id"]) for c in chunks_after],
                         [int(c["id"]) for c in chunks_before])
        # fakechunk's vectors survived untouched (same chunk ids and blobs)
        with self.db.transaction() as conn:
            for chunk in chunks_after:
                rows = repo.get_chunk_embeddings(conn, chunk["id"])
                models = {str(r["model"]) for r in rows}
                self.assertEqual(models, {"fakechunk", "secondmodel"})
                first_vec = next(r for r in rows
                                 if r["model"] == "fakechunk")
                self.assertEqual(
                    first_vec["vector"],
                    vectors_before[int(chunk["id"])][0]["vector"])
        track = None
        with self.db.transaction() as conn:
            track = repo.get_track(conn, self.track_id)
        self.assertEqual(track["status"], "analyzed")
        self.assertIn("already analyzed (kept)", track["status_message"])
        # the second plugin's display name is also "FakeChunk"; the note
        # just records that 4 embeddings were stored for it
        self.assertIn("4 chunk embeddings", track["status_message"])

    def test_incremental_all_covered_is_a_quiet_success(self) -> None:
        import app.analysis.pipeline as pipeline

        plugin = FakeChunkPlugin(name="fakechunk")
        with mock.patch.object(pipeline, "get_plugin",
                               lambda name: plugin):
            pipeline.analyze_track(self.db, self.track_id, self._config())
            plugin.embed_batches.clear()
            pipeline.analyze_track(self.db, self.track_id, self._config())
        # nothing re-embedded, status still analyzed (not an error)
        self.assertEqual(plugin.embed_batches, [])
        with self.db.transaction() as conn:
            track = repo.get_track(conn, self.track_id)
        self.assertEqual(track["status"], "analyzed")
        self.assertIn("already analyzed (kept)", track["status_message"])

    def test_changed_chunk_params_rechunks_and_drops_old_vectors(self) -> None:
        """Changing chunking is the legitimate destructive case."""
        import app.analysis.pipeline as pipeline

        plugin = FakeChunkPlugin(name="fakechunk")
        with mock.patch.object(pipeline, "get_plugin", lambda name: plugin):
            pipeline.analyze_track(self.db, self.track_id, self._config())
            with self.db.transaction() as conn:
                ids_before = [int(c["id"])
                              for c in repo.get_chunks(conn, self.track_id)]
            cfg = self._config()
            cfg.chunk_seconds = 1.0     # -> 2 chunks instead of 4
            pipeline.analyze_track(self.db, self.track_id, cfg)
        with self.db.transaction() as conn:
            chunks_after = repo.get_chunks(conn, self.track_id)
            ids_after = [int(c["id"]) for c in chunks_after]
        self.assertNotEqual(ids_after, ids_before)
        self.assertEqual(len(ids_after), 2)
        with self.db.transaction() as conn:
            for chunk in chunks_after:
                rows = repo.get_chunk_embeddings(conn, chunk["id"])
                self.assertEqual(len(rows), 1)   # re-embedded on new chunks

    def test_forced_run_reembeds_existing_model(self) -> None:
        import app.analysis.pipeline as pipeline

        plugin = FakeChunkPlugin(name="fakechunk")
        with mock.patch.object(pipeline, "get_plugin", lambda name: plugin):
            pipeline.analyze_track(self.db, self.track_id, self._config())
            embeds_after_first = len(plugin.embed_batches)
            with self.db.transaction() as conn:
                ids_before = [int(c["id"])
                              for c in repo.get_chunks(conn, self.track_id)]
            pipeline.analyze_track(self.db, self.track_id, self._config(),
                                   force=True)
        # chunks kept (plan unchanged) but the model re-ran and overwrote
        with self.db.transaction() as conn:
            ids_after = [int(c["id"])
                         for c in repo.get_chunks(conn, self.track_id)]
        self.assertEqual(ids_after, ids_before)
        self.assertGreater(len(plugin.embed_batches), embeds_after_first)

    def test_plugin_failure_midway_still_sets_error_status(self):
        import app.analysis.pipeline as pipeline

        plugin = FakeChunkPlugin(fail_on_embed_batch=2)  # 2nd embed batch dies
        calls, notify = _notify_log()
        with mock.patch.object(pipeline, "get_plugin", lambda name: plugin):
            with self.assertRaises(RuntimeError):
                pipeline.analyze_track(self.db, self.track_id,
                                       self._config(), progress_cb=notify)

        # The failure surfaced as the guarded per-plugin warning note.
        self.assertTrue(any("⚠" in m and "failed" in m for m, _, _ in calls))
        # First batch still made progress before the crash: exactly one tick.
        chunk_ticks = [cur for msg, cur, _ in calls if "chunk" in msg]
        self.assertEqual(chunk_ticks, [2])
        self.assertEqual(plugin.embed_batches, [2, 2])
        # Existing behavior preserved: the track ends in the error state.
        with self.db.transaction() as conn:
            track = repo.get_track(conn, self.track_id)
        self.assertEqual(track["status"], "error")
        self.assertIn("No analysis models available", track["status_message"])


class _WorkerEvents(dict):
    """Event recorder wired to every AnalysisWorker signal."""

    def __init__(self) -> None:
        super().__init__(chunk=[], progress=[], finished=[],
                         stopped=[], all=[], failed=[])


def _wire(worker, ev: _WorkerEvents) -> None:
    worker.track_chunk_progress.connect(
        lambda tid, cur, tot, label: ev["chunk"].append((tid, cur, tot, label)))
    worker.track_progress.connect(
        lambda tid, msg: ev["progress"].append((tid, msg)))
    worker.track_finished.connect(
        lambda tid, ok, msg: ev["finished"].append((tid, ok, msg)))
    worker.stopped.connect(lambda count: ev["stopped"].append(count))
    worker.all_finished.connect(lambda: ev["all"].append(True))
    worker.failed.connect(lambda msg: ev["failed"].append(msg))


class TestWorkerChunkProgress(unittest.TestCase):
    """AnalysisWorker: additive chunk-tick signal, stop semantics intact."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.track_id = _make_db_with_wav(self._tmp)
        self.db_path = Path(self._tmp.name) / "lib.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _make_worker(self, track_ids):
        from app.ui.workers import AnalysisWorker
        return AnalysisWorker(self.db_path, AppConfig(), list(track_ids))

    def _row(self, track_id):
        conn = self.db.connect()
        try:
            return repo.get_track(conn, track_id)
        finally:
            conn.close()

    def test_chunk_ticks_reach_both_signals(self):
        worker = self._make_worker([self.track_id])
        ev = _WorkerEvents()
        _wire(worker, ev)

        def fake(db, track_id, config, progress_cb=None,
                force=False):
            notify = progress_cb or (lambda msg, cur=None, tot=None: None)
            notify("Decoding audio", None, None)     # coarse: no cur/tot
            notify("CLAP: chunk 1/4", 1, 4)
            notify("CLAP: chunk 2/4", 2, 4)
            notify("CLAP: tags 4/4", 4, 4)

        # The worker throttles chunk ticks to ~10 Hz per track; a fake
        # clock spaced 1 s apart lets every tick of this test through.
        # the throttle seeds last=0.0, so the first tick's value must
        # already clear the 0.1 s threshold
        clock = iter([0.2, 1.0, 2.0, 3.0, 4.0])
        with mock.patch("app.analysis.pipeline.analyze_track", fake), \
                mock.patch("app.ui.workers.time.monotonic",
                           lambda: next(clock)):
            worker.run()  # synchronous, like TestAnalysisStop's fakes

        # Structured chunk ticks: (track_id, current, total, label).
        self.assertEqual(ev["chunk"], [
            (self.track_id, 1, 4, "CLAP: chunk 1/4"),
            (self.track_id, 2, 4, "CLAP: chunk 2/4"),
            (self.track_id, 4, 4, "CLAP: tags 4/4"),
        ])
        # Coarse signal still carries EVERY message, including chunk ones.
        self.assertEqual([msg for _, msg in ev["progress"]],
                         ["Decoding audio", "CLAP: chunk 1/4",
                          "CLAP: chunk 2/4", "CLAP: tags 4/4"])
        self.assertEqual(ev["failed"], [])
        self.assertEqual(ev["stopped"], [])            # NOT emitted
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual([ok for _, ok, _ in ev["finished"]], [True])

    def test_stop_flag_still_aborts_between_chunk_ticks(self):
        from app.ui import workers as workers_mod

        worker = self._make_worker([self.track_id])
        ev = _WorkerEvents()
        _wire(worker, ev)

        def fake(db, track_id, config, progress_cb=None,
                force=False):
            notify = progress_cb or (lambda msg, cur=None, tot=None: None)
            notify("CLAP: chunk 1/4", 1, 4)   # before the stop: goes through
            worker.request_stop()             # user clicks Stop mid-track
            notify("CLAP: chunk 2/4", 2, 4)   # progress_cb raises here

        with mock.patch("app.analysis.pipeline.analyze_track", fake):
            worker.run()

        # Only the pre-stop tick was emitted, on both signals.
        self.assertEqual(ev["chunk"],
                         [(self.track_id, 1, 4, "CLAP: chunk 1/4")])
        self.assertEqual([msg for _, msg in ev["progress"]],
                         ["CLAP: chunk 1/4"])
        self.assertEqual(ev["failed"], [])
        self.assertEqual(ev["stopped"], [0])
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual(len(ev["finished"]), 1)
        tid, ok, msg = ev["finished"][0]
        self.assertEqual(tid, self.track_id)
        self.assertFalse(ok)
        self.assertIn("Stopped by user", msg)
        row = self._row(self.track_id)
        self.assertEqual(row["status"], "new")         # retryable again
        self.assertIn("Stopped by user", row["status_message"])


class TestMainWindowChunkProgress(unittest.TestCase):
    """The status-bar progress bar becomes determinate on chunk ticks."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.track_id = _make_db_with_wav(self._tmp)
        self.db_path = Path(self._tmp.name) / "lib.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_progress_bar_follows_chunk_ticks(self):
        from PySide6.QtTest import QTest

        from app.ui.main_window import MainWindow

        config = AppConfig()
        config.use_ollama = False
        config.analyze_wav = True   # these fixtures deliberately analyze WAV
        win = MainWindow(config, db_path=self.db_path)
        worker = None
        try:
            first_tick = threading.Event()
            release = threading.Event()

            def fake(db, track_id, cfg, progress_cb=None,
                force=False):
                notify = progress_cb or (lambda msg, cur=None, tot=None: None)
                notify("Decoding audio", None, None)
                notify("FakeChunk: chunk 2/4", 2, 4)
                first_tick.set()
                # Hold the run until the UI side was asserted, so the tick
                # status/bar cannot be overwritten by the final messages.
                release.wait(5.0)
                notify("FakeChunk: chunk 4/4", 4, 4)

            with mock.patch("app.analysis.pipeline.analyze_track", fake):
                win.analyze_track_ids([self.track_id])
                worker = win._analysis_worker
                self.assertIsNotNone(worker)
                self.assertTrue(first_tick.wait(5.0))

                # Bounded qWait loop until the tick reached the UI. Poll the
                # determinate bar (not the status text): the pre-analysis
                # "… Ns chunks …" message already contains the word "chunk".
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    if (win._progress.maximum() == 4
                            and win._progress.value() == 2):
                        break
                    QTest.qWait(10)
                self.assertEqual(win._progress.maximum(), 4)
                self.assertEqual(win._progress.value(), 2)
                self.assertGreaterEqual(win._progress.value(), 1)
                # The chunk tick's status line carries the per-chunk label
                # ("Analyzing song.wav — FakeChunk: chunk 2/4", or the coarse
                # "Track 1: …" variant until the tick slot runs — both fine).
                self.assertIn("chunk", win.statusBar().currentMessage())
        finally:
            # Never leave the worker blocked in the fake, even on failure.
            release.set()
            if worker is not None:
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    QTest.qWait(20)
                    if not worker.isRunning():
                        break
        self.assertIsNotNone(worker)
        self.assertFalse(worker.isRunning())   # clean finish, no hang

    def test_track_start_resets_bar_to_indeterminate(self):
        from app.ui.main_window import MainWindow

        config = AppConfig()
        config.use_ollama = False
        config.analyze_wav = True   # these fixtures deliberately analyze WAV
        win = MainWindow(config, db_path=self.db_path)
        try:
            # Simulate a determinate leftover from a previous track/model…
            win._on_track_chunk_progress(self.track_id, 3, 4,
                                         "FakeChunk: chunk 3/4")
            self.assertEqual(win._progress.maximum(), 4)
            # …then the next track start must reset to indeterminate.
            win._on_track_analysis_started(self.track_id,
                                           str(Path(self._tmp.name) / "song.wav"))
            self.assertEqual(win._progress.minimum(), 0)
            self.assertEqual(win._progress.maximum(), 0)
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
