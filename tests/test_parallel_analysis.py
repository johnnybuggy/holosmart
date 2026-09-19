"""Parallel analysis, skip-already-analyzed, and the plugin load lock.

Four levels, mirroring the established fixture styles:
- config: default value + JSON save/load roundtrip — like
  tests/test_model_settings.py::ConfigRoundtripTests;
- settings dialog: offscreen widget poking a single spinbox — like
  tests/test_model_settings.py::SettingsDialogTests;
- worker: AnalysisWorker.run() driven synchronously with a faked
  analyze_track on a real temp Database — like tests/test_ui.py
  ::TestAnalysisStop and tests/test_chunk_progress.py
  ::TestWorkerChunkProgress (including the _AnalysisStopped abort style);
- model base: two threads racing ModelPlugin.ensure_loaded() — pure
  threading, no Qt, no torch.

Headless: no torch, no network; Qt only offscreen where a widget is needed.
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
from PySide6.QtCore import Qt

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.ui.workers import AnalysisWorker  # noqa: E402


def _make_db_with_tracks(tmp, count: int = 3) -> tuple[Database, list[int]]:
    """Real temp Database with *count* generated 8 kHz mono sine wav tracks.

    The analysis pipeline is faked in every worker test here, so the wav
    content only has to exist — the rows must, though, or the worker skips
    them (mirrors tests/test_analysis_position.py's fixture).
    """
    dir_path = Path(tmp.name)
    db = Database(dir_path / "lib.db")
    ids: list[int] = []
    with db.transaction() as conn:
        folder_id = repo.add_folder(conn, str(dir_path))
        for i in range(count):
            wav = dir_path / f"song{i}.wav"
            sr = 8000
            t = np.linspace(0, 0.5, sr // 2, endpoint=False)
            sf.write(wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
            ids.append(repo.upsert_track(
                conn, folder_id, str(wav),
                {"filename": wav.name, "extension": ".wav"}))
    return db, ids


def _config(degree: int | None = None) -> AppConfig:
    """AppConfig with an optional analysis_parallelism override."""
    cfg = AppConfig()
    cfg.use_ollama = False
    if degree is not None:
        cfg.analysis_parallelism = degree
    return cfg


class _Events(dict):
    """Event recorder wired to every AnalysisWorker signal."""

    def __init__(self) -> None:
        super().__init__(position=[], started=[], progress=[], finished=[],
                         stopped=[], skipped=[], all=[], failed=[])


def _wire(worker, ev: _Events) -> None:
    # Explicit DirectConnection: with parallelism > 1 the per-track signals
    # are emitted from pool threads, and these tests drive run() without an
    # event loop — an auto-connection would queue those emissions away and
    # they would never be recorded. Direct delivery from pool threads is
    # safe here (plain list appends; the pool is drained before run()
    # returns).
    direct = Qt.ConnectionType.DirectConnection
    worker.track_position.connect(
        lambda pos, tot, name: ev["position"].append((pos, tot, name)), direct)
    worker.track_started.connect(
        lambda tid, path: ev["started"].append(tid), direct)
    worker.track_progress.connect(
        lambda tid, msg: ev["progress"].append((tid, msg)), direct)
    worker.track_finished.connect(
        lambda tid, ok, msg: ev["finished"].append((tid, ok, msg)), direct)
    worker.stopped.connect(lambda count: ev["stopped"].append(count), direct)
    worker.skipped.connect(lambda count: ev["skipped"].append(count), direct)
    worker.all_finished.connect(lambda: ev["all"].append(True), direct)
    worker.failed.connect(lambda msg: ev["failed"].append(msg), direct)


def _run(worker, fake) -> None:
    """Drive run() synchronously with *fake* as the pipeline (TestAnalysisStop
    style)."""
    with mock.patch("app.analysis.pipeline.analyze_track", fake):
        worker.run()


def _concurrency_fake(tracker: "_Concurrency", delay: float = 0.15):
    """Fake analyze_track measuring real overlap: enter the tracker, sleep,
    leave, then emit exactly one coarse notify."""
    def fake(db, track_id, config, progress_cb=None,
                force=False):
        notify = progress_cb or (lambda msg, cur=None, tot=None: None)
        with tracker:
            time.sleep(delay)
        notify("step")
    return fake


class _Concurrency:
    """Thread-safe recorder of the observed maximum parallel overlap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def __enter__(self) -> "_Concurrency":
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        return self

    def __exit__(self, *exc) -> None:
        with self._lock:
            self._active -= 1


def _stop_fake(worker, state):
    """Fake pipeline shaped like tests/test_ui.py::TestAnalysisStop
    ._fake_analyze: notify, maybe abort, persist, maybe stop after N.

    ``state["stop_before_step"]``  — Stop clicked mid-track before the next
    pipeline step: the worker's own progress callback raises.
    ``state["stop_after_notify"]`` — Stop clicked between two steps: the fake
    itself raises the workers-module stop exception.
    ``state["stop_after"]``        — Stop clicked right after the Nth track
    finished: the stop surfaces before queued tracks start.
    """
    from app.ui import workers as workers_mod

    def fake(db, track_id, config, progress_cb=None,
                force=False):
        notify = progress_cb or (lambda msg, cur=None, tot=None: None)
        if state.get("stop_before_step"):
            worker.request_stop()
        notify("step")            # worker's progress_cb raises when stopped
        if state.get("stop_after_notify"):
            worker.request_stop()
        if worker._stop_requested:
            raise workers_mod._AnalysisStopped(track_id)
        with db.transaction() as conn:
            repo.set_track_status(conn, track_id, "analyzed", "fake done")
        state["completed"].append(track_id)
        if state.get("stop_after") == len(state["completed"]):
            worker.request_stop()
    return fake


# ------------------------------------------------------------------- config
class EffectiveParallelismTests(unittest.TestCase):
    """FFT-only runs scale to every core; every other run is sequential."""

    def _worker(self, cfg: AppConfig):
        from app.ui.workers import AnalysisWorker

        return AnalysisWorker(":memory:", cfg, [])

    def test_fft_only_run_uses_all_but_one_core(self) -> None:
        import os

        cfg = _config(degree=1)            # even a sequential setting...
        cfg.models = ["fft"]
        cores = max(1, (os.cpu_count() or 2) - 1)   # one core for the GUI
        self.assertEqual(
            self._worker(cfg)._effective_parallelism(cfg, 5),
            min(cores, 5))
        self.assertEqual(
            self._worker(cfg)._effective_parallelism(cfg, 1), 1)

    def test_mixed_models_run_sequential(self) -> None:
        """Any run involving a model other than FFT is strictly sequential —
        model inference serializes on the GPU/MPS anyway, and the legacy
        analysis_parallelism setting no longer widens these runs."""
        cfg = _config(degree=3)
        worker = self._worker(cfg)
        cfg.models = ["clap", "mert", "fft"]
        self.assertEqual(worker._effective_parallelism(cfg, 100), 1)
        cfg.models = ["mert330", "fft"]
        self.assertEqual(worker._effective_parallelism(cfg, 100), 1)
        cfg.models = ["clap"]
        self.assertEqual(worker._effective_parallelism(cfg, 100), 1)
        cfg.models = ["fft", "openl3"]
        self.assertEqual(worker._effective_parallelism(cfg, 100), 1)
        # ... and the pool never exceeds the workload either way
        self.assertEqual(self._worker(cfg)._effective_parallelism(cfg, 1), 1)

    def test_fft_only_run_ignores_the_setting(self) -> None:
        cfg = _config()
        cfg.models = ["fft"]
        object.__setattr__(cfg, "analysis_parallelism", "nonsense")
        import os
        cores = max(1, (os.cpu_count() or 2) - 1)
        self.assertEqual(
            self._worker(cfg)._effective_parallelism(cfg, 4),
            min(cores, 4))


class ConfigParallelismTests(unittest.TestCase):
    """Legacy analysis_parallelism field: still loads and persists (it no
    longer influences any run's parallelism)."""

    def test_default_analysis_parallelism_is_two(self) -> None:
        self.assertEqual(AppConfig().analysis_parallelism, 2)

    def test_config_roundtrip_persists_analysis_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            cfg = AppConfig()
            cfg.analysis_parallelism = 4
            with mock.patch("app.config.DATA_DIR", data_dir), \
                    mock.patch("app.config.CONFIG_PATH",
                               data_dir / "config.json"):
                cfg.save()
                loaded = AppConfig.load()
        self.assertEqual(loaded.analysis_parallelism, 4)


# ------------------------------------------------------------------ dialog
class SettingsDialogParallelismTests(unittest.TestCase):
    """The manual parallelism control is gone: parallelism is automatic
    (FFT-only runs use every core but one, other models run sequentially)."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def test_no_parallelism_spinbox(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppConfig())
        self.assertFalse(hasattr(dialog, "_parallelism"))

    def test_apply_leaves_legacy_field_untouched(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        config.analysis_parallelism = 5
        dialog = SettingsDialog(config)
        dialog.apply()
        self.assertEqual(config.analysis_parallelism, 5)   # legacy, ignored


# ------------------------------------------------------------------ worker
class TestWorkerParallelism(unittest.TestCase):
    """AnalysisWorker runs tracks concurrently per analysis_parallelism."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.ids = _make_db_with_tracks(self._tmp, 3)
        self.db_path = Path(self._tmp.name) / "lib.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _row(self, track_id):
        conn = self.db.connect()
        try:
            return repo.get_track(conn, track_id)
        finally:
            conn.close()

    def test_fft_only_run_parallelizes(self):
        # Pool mechanics need an FFT-only run: every other model set is
        # strictly sequential now (analyze_track is faked here regardless).
        # The FFT degree is cores-based, so pin the core count: 4 cores →
        # all but one reserved for the GUI → degree 3.
        tracker = _Concurrency()
        cfg = _config(3)
        cfg.models = ["fft"]
        worker = AnalysisWorker(self.db_path, cfg, self.ids)
        ev = _Events()
        _wire(worker, ev)

        _run(worker, _concurrency_fake(tracker))

        self.assertEqual(tracker.max_active, 3)      # real overlap observed
        self.assertEqual(ev["failed"], [])
        self.assertEqual(ev["skipped"], [0])         # always emitted, 0 here
        self.assertEqual(ev["stopped"], [])          # no stop path taken
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual(sorted(ev["started"]), sorted(self.ids))
        self.assertEqual(sorted(tid for tid, ok, _ in ev["finished"] if ok),
                         sorted(self.ids))

    def test_degree_one_stays_sequential(self):
        tracker = _Concurrency()
        worker = AnalysisWorker(self.db_path, _config(1), self.ids)
        ev = _Events()
        _wire(worker, ev)

        _run(worker, _concurrency_fake(tracker))

        self.assertEqual(tracker.max_active, 1)
        self.assertEqual(ev["failed"], [])
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual(len(ev["finished"]), 3)

    def test_config_without_models_field_runs_sequentially(self):
        # Old configs / fakes lacking attributes keep working — and any run
        # that is not FFT-only is strictly sequential now.
        class _OldConfig:
            models: list[str] = []
            chunk_seconds = 20.0
            overlap_percent = 50.0

        tracker = _Concurrency()
        worker = AnalysisWorker(self.db_path, _OldConfig(), self.ids)
        ev = _Events()
        _wire(worker, ev)

        _run(worker, _concurrency_fake(tracker))

        self.assertEqual(tracker.max_active, 1)
        self.assertEqual(ev["failed"], [])
        self.assertEqual(len(ev["all"]), 1)


class TestWorkerStopWithPool(unittest.TestCase):
    """Stop semantics survive the pool: abort in-flight, spare queued."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.ids = _make_db_with_tracks(self._tmp, 3)
        self.db_path = Path(self._tmp.name) / "lib.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _row(self, track_id):
        conn = self.db.connect()
        try:
            return repo.get_track(conn, track_id)
        finally:
            conn.close()

    def _fft_config(self, cores: int):
        """FFT-only config with a pinned core count (FFT degree = cores-1)."""
        cfg = _config(2)
        cfg.models = ["fft"]      # pool path only applies to FFT-only runs
        return cfg, mock.patch("app.ui.workers.os.cpu_count",
                               return_value=cores)

    def test_stop_before_start_starts_nothing(self):
        cfg, cores = self._fft_config(4)   # degree 3
        worker = AnalysisWorker(self.db_path, cfg, self.ids)
        self.addCleanup(cores.stop)
        worker.request_stop()                  # BEFORE run() — user was fast
        ev = _Events()
        _wire(worker, ev)

        with cores:
            _run(worker, _stop_fake(worker, {"completed": []}))

        self.assertEqual(ev["failed"], [])
        self.assertEqual(ev["position"], [])   # nothing started at all
        self.assertEqual(ev["started"], [])
        self.assertEqual(ev["finished"], [])
        self.assertEqual(ev["skipped"], [0])
        self.assertEqual(ev["stopped"], [0])   # completed count is 0
        self.assertEqual(len(ev["all"]), 1)
        for tid in self.ids:
            self.assertEqual(self._row(tid)["status"], "new")

    def test_stop_mid_run_aborts_inflight_and_spares_queued(self):
        # 3 tracks, degree 2 (3 pinned cores − 1 for the GUI): t1/t2 are in
        # flight and abort at their first notify; t3 is still queued and
        # bails out untouched.
        cfg, cores = self._fft_config(3)
        worker = AnalysisWorker(self.db_path, cfg, self.ids)
        ev = _Events()
        _wire(worker, ev)
        state = {"completed": [], "stop_before_step": True}

        with cores:
            _run(worker, _stop_fake(worker, state))

        self.assertEqual(ev["failed"], [])
        self.assertEqual(sorted(ev["started"]), self.ids[:2])
        self.assertEqual([ok for _, ok, _ in ev["finished"]], [False, False])
        self.assertTrue(all("Stopped by user" in msg
                            for _, _, msg in ev["finished"]))
        self.assertEqual(ev["stopped"], [0])
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual(state["completed"], [])        # nothing completed
        for tid in self.ids[:2]:                        # aborted → retryable
            row = self._row(tid)
            self.assertEqual(row["status"], "new")
            self.assertIn("Stopped by user", row["status_message"])
        queued = self._row(self.ids[2])                 # never touched
        self.assertEqual(queued["status"], "new")
        self.assertIsNone(queued["status_message"])

    def test_stop_after_first_completion_spares_the_rest(self):
        # degree 1 mirror of TestAnalysisStop's stop-between-tracks case:
        # the stop lands after track 1 completed, queued tracks stay
        # untouched (status "new", no message) for a later resume.
        worker = AnalysisWorker(self.db_path, _config(1), self.ids)
        ev = _Events()
        _wire(worker, ev)
        state = {"completed": [], "stop_after": 1}
        worker = worker  # degree 1 runs sequentially regardless of models

        _run(worker, _stop_fake(worker, state))

        self.assertEqual(ev["failed"], [])
        self.assertEqual(ev["started"], [self.ids[0]])
        self.assertEqual([ok for _, ok, _ in ev["finished"]], [True])
        self.assertEqual(ev["stopped"], [1])   # 1 track done before the stop
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual(self._row(self.ids[0])["status"], "analyzed")
        for tid in self.ids[1:]:
            row = self._row(tid)
            self.assertEqual(row["status"], "new")
            self.assertIsNone(row["status_message"])


class TestWorkerSkipAnalyzed(unittest.TestCase):
    """Batch runs skip status='analyzed' tracks; force re-analyzes them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db, self.ids = _make_db_with_tracks(self._tmp, 3)
        self.db_path = Path(self._tmp.name) / "lib.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _set_status(self, track_id: int, status: str) -> None:
        with self.db.transaction() as conn:
            repo.set_track_status(conn, track_id, status)
        if status == "analyzed":
            # "analyzed" now means FULLY analyzed: every enabled model has a
            # vector on every chunk (the skip policy checks coverage, not
            # just the status flag).
            config = _config()
            for model in config.models:
                self._add_full_model_vectors(track_id, model)

    def _add_full_model_vectors(self, track_id: int, model: str) -> None:
        with self.db.transaction() as conn:
            chunks = repo.get_chunks(conn, track_id)
            if not chunks:
                repo.replace_chunks(conn, track_id, [(0, 0.0, 0.5)])
                chunks = repo.get_chunks(conn, track_id)
            for chunk in chunks:
                vec = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, "
                    "vector, norm) VALUES (?, ?, ?, ?, ?)",
                    (int(chunk["id"]), model, 3,
                     np.asarray(vec, dtype="<f4").tobytes(),
                     float(np.linalg.norm(vec))))

    def _row(self, track_id):
        conn = self.db.connect()
        try:
            return repo.get_track(conn, track_id)
        finally:
            conn.close()

    @staticmethod
    def _recording_fake(record: list[int]):
        def fake(db, track_id, config, progress_cb=None,
                force=False):
            notify = progress_cb or (lambda msg, cur=None, tot=None: None)
            notify("step")
            with db.transaction() as conn:
                repo.set_track_status(conn, track_id, "analyzed", "fake done")
            record.append(track_id)
        return fake

    def test_analyzed_tracks_are_skipped(self):
        self._set_status(self.ids[0], "analyzed")
        self._set_status(self.ids[1], "analyzed")
        worker = AnalysisWorker(self.db_path, _config(2), self.ids)
        ev = _Events()
        _wire(worker, ev)
        record: list[int] = []

        _run(worker, self._recording_fake(record))

        self.assertEqual(ev["skipped"], [2])           # announced once, upfront
        self.assertEqual(ev["started"], [self.ids[2]])  # only the fresh track
        self.assertEqual([ok for _, ok, _ in ev["finished"]], [True])
        self.assertEqual(ev["failed"], [])
        self.assertEqual(ev["stopped"], [])
        self.assertEqual(len(ev["all"]), 1)
        self.assertEqual(record, [self.ids[2]])
        # Skipped rows keep their finished state untouched.
        self.assertEqual(self._row(self.ids[0])["status"], "analyzed")
        self.assertEqual(self._row(self.ids[1])["status"], "analyzed")

    def test_skip_analyzed_false_visits_analyzed_tracks(self) -> None:
        """Recursive folder analysis: analyzed tracks are visited too (the
        pipeline then keeps their results incrementally)."""
        self._set_status(self.ids[0], "analyzed")
        self._set_status(self.ids[1], "analyzed")
        worker = AnalysisWorker(self.db_path, _config(2), self.ids,
                                skip_analyzed=False)
        ev = _Events()
        _wire(worker, ev)
        record: list[int] = []

        _run(worker, self._recording_fake(record))

        self.assertEqual(ev["skipped"], [0])           # nothing skipped
        self.assertEqual(sorted(record), sorted(self.ids))
        self.assertEqual([ok for _, ok, _ in ev["finished"]],
                         [True] * 3)

    def test_all_analyzed_emits_skipped_and_finishes_quietly(self):
        for tid in self.ids:
            self._set_status(tid, "analyzed")
        worker = AnalysisWorker(self.db_path, _config(2), self.ids)
        ev = _Events()
        _wire(worker, ev)
        record: list[int] = []

        _run(worker, self._recording_fake(record))

        self.assertEqual(ev["skipped"], [3])
        self.assertEqual(ev["started"], [])            # no track_started at all
        self.assertEqual(ev["finished"], [])
        self.assertEqual(ev["stopped"], [])            # no stopped either
        self.assertEqual(ev["failed"], [])             # and no failed
        self.assertEqual(len(ev["all"]), 1)            # still a clean finish
        self.assertEqual(record, [])

    def test_force_reanalyze_reprocesses_analyzed(self):
        for tid in self.ids:
            self._set_status(tid, "analyzed")
        worker = AnalysisWorker(self.db_path, _config(2), self.ids,
                                force_reanalyze=True)
        ev = _Events()
        _wire(worker, ev)
        record: list[int] = []

        _run(worker, self._recording_fake(record))

        self.assertEqual(ev["skipped"], [0])           # nothing skipped
        self.assertEqual(sorted(ev["started"]), sorted(self.ids))
        self.assertEqual(sorted(record), sorted(self.ids))
        self.assertEqual(ev["failed"], [])
        self.assertEqual(len(ev["all"]), 1)
        for tid in self.ids:
            self.assertEqual(self._row(tid)["status"], "analyzed")


# --------------------------------------------------------------- base lock
class EnsureLoadedLockTests(unittest.TestCase):
    """ModelPlugin.ensure_loaded double-checks under a lock."""

    def test_concurrent_first_use_loads_once(self):
        from app.models.base import ModelPlugin

        class StubPlugin(ModelPlugin):
            name = "stub"
            display_name = "Stub"
            requirements = ()

            def __init__(self):
                super().__init__()
                self.load_calls = 0
                self._calls_lock = threading.Lock()

            def _load(self):
                with self._calls_lock:
                    self.load_calls += 1
                time.sleep(0.05)   # widen the race window

        plugin = StubPlugin()
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def user():
            try:
                barrier.wait(timeout=5.0)
                plugin.ensure_loaded()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=user) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)

        self.assertEqual(errors, [])
        self.assertEqual(plugin.load_calls, 1)   # no double-load race
        self.assertTrue(plugin.is_loaded)

    def test_load_failure_keeps_retryable_semantics(self):
        from app.models.base import ModelPlugin

        class FailingPlugin(ModelPlugin):
            name = "failing"
            display_name = "Failing"
            requirements = ()

            def __init__(self):
                super().__init__()
                self.attempts = 0

            def _load(self):
                self.attempts += 1
                raise ValueError("weights missing")

        plugin = FailingPlugin()
        with self.assertRaises(RuntimeError):
            plugin.ensure_loaded()
        self.assertFalse(plugin.is_loaded)       # stays False on failure
        self.assertEqual(plugin.attempts, 1)
        # The lock was released cleanly: a later call retries the load.
        with self.assertRaises(RuntimeError):
            plugin.ensure_loaded()
        self.assertEqual(plugin.attempts, 2)


if __name__ == "__main__":
    unittest.main()
