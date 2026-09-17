"""Scan-pipeline correctness under the ScanWorker performance optimisations.

ScanWorker.run() is called synchronously (no QThread event loop, no widgets),
with signals connected to a plain recorder object — the same pattern
tests/test_ui.py uses. Covers: discovery/progress accounting, the rescan
cache (unchanged files are not re-probed, touched ones are), parallel probing
across several pool threads, and per-file probe-failure isolation.
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

from app.db import repo
from app.db.database import Database
from app.ui.workers import ScanWorker


class _SignalRecorder:
    """Collect every ScanWorker signal emission as plain Python tuples."""

    def __init__(self) -> None:
        self.scan_started: list[int] = []
        self.folder_scanned: list[tuple[str, int]] = []
        self.track_upserted: list[tuple[int, str]] = []
        self.progress: list[tuple[int, int, str]] = []
        self.file_error: list[tuple[str, str]] = []
        self.permission_required: list[tuple[str, str]] = []
        self.finished_scan: list[tuple[int, int, int, int, int]] = []
        self.failed: list[str] = []

    def connect(self, worker: ScanWorker) -> None:
        worker.scan_started.connect(self.scan_started.append)
        worker.folder_scanned.connect(
            lambda path, found: self.folder_scanned.append((path, found)))
        worker.track_upserted.connect(
            lambda track_id, path: self.track_upserted.append(
                (track_id, path)))
        worker.progress.connect(
            lambda cur, tot, name: self.progress.append((cur, tot, name)))
        worker.file_error.connect(
            lambda path, msg: self.file_error.append((path, msg)))
        worker.permission_required.connect(
            lambda directory, msg: self.permission_required.append(
                (directory, msg)))
        worker.finished_scan.connect(
            lambda added, updated, removed, errors, denied:
            self.finished_scan.append(
                (added, updated, removed, errors, denied)))
        worker.failed.connect(self.failed.append)


class ScanTestBase(unittest.TestCase):
    """Temp music tree + temp database + a synchronous scan runner."""

    @classmethod
    def setUpClass(cls):
        cls.total = 30  # 5 files in the root + 5x5 in nested subdirectories

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "music"
        self.db_path = self.dir / "lib.db"
        self._make_tree()
        self.recorder = _SignalRecorder()

    def tearDown(self):
        self._tmp.cleanup()

    def _make_tree(self) -> None:
        """Create 30 tiny 16-bit PCM wavs: some in the root, the rest nested."""
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        self.root.mkdir(parents=True)
        for i in range(5):
            sf.write(self.root / f"root_{i}.wav", tone, sr, subtype="PCM_16")
        for d in range(5):
            sub = self.root / f"folder_{d}" / f"sub_{d}"
            sub.mkdir(parents=True)
            for i in range(5):
                sf.write(sub / f"track_{d}_{i}.wav", tone, sr,
                         subtype="PCM_16")

    def _scan(self) -> ScanWorker:
        worker = ScanWorker(self.db_path, [str(self.root)])
        self.recorder.connect(worker)
        worker.run()  # synchronous; no QThread event loop needed
        return worker

    def _tracks(self) -> list:
        with Database(self.db_path).transaction() as conn:
            return repo.list_tracks(conn)


class TestScanCorrectness(ScanTestBase):
    def test_scan_indexes_files_and_reports_progress(self):
        self._scan()

        self.assertEqual(self.recorder.failed, [])
        self.assertEqual(self.recorder.scan_started, [self.total])
        self.assertEqual(self.recorder.folder_scanned,
                         [(str(self.root), self.total)])
        self.assertEqual(len(self.recorder.track_upserted), self.total)
        self.assertEqual(self.recorder.permission_required, [])
        self.assertEqual(self.recorder.file_error, [])
        self.assertEqual(self.recorder.finished_scan,
                         [(self.total, 0, 0, 0, 0)])
        # exactly one progress emission per file, counting 1..total in order
        self.assertEqual([p[0] for p in self.recorder.progress],
                         list(range(1, self.total + 1)))
        self.assertTrue(all(p[1] == self.total for p in self.recorder.progress))
        self.assertEqual(len({p[2] for p in self.recorder.progress}),
                         self.total)
        tracks = self._tracks()
        self.assertEqual(len(tracks), self.total)
        self.assertTrue(all(t["codec"] == "pcm_s16le" for t in tracks))

    def test_rescan_counts_updates_not_adds(self):
        self._scan()
        self._scan()
        self.assertEqual(self.recorder.finished_scan[-1],
                         (0, self.total, 0, 0, 0))
        self.assertEqual(len(self._tracks()), self.total)


class TestRescanCache(ScanTestBase):
    """Unchanged files must be indexed from the stored row, not re-probed."""

    _FAKE_INFO = {"codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                  "duration_sec": 1.0}

    def test_unchanged_files_are_not_reprobed(self):
        self._scan()
        fake = mock.MagicMock(return_value=dict(self._FAKE_INFO))
        with mock.patch("app.ui.workers.probe_audio", fake):
            self._scan()

        fake.assert_not_called()
        self.assertEqual(self.recorder.finished_scan[-1],
                         (0, self.total, 0, 0, 0))
        # both scans emitted full progress; the rescan ran without any probe
        self.assertEqual(len(self.recorder.progress), 2 * self.total)
        second = self.recorder.progress[self.total:]
        self.assertEqual([p[0] for p in second],
                         list(range(1, self.total + 1)))
        # stored metadata survived the cache-hit upsert untouched
        tracks = self._tracks()
        self.assertEqual(len(tracks), self.total)
        self.assertTrue(all(t["codec"] == "pcm_s16le" for t in tracks))

    def test_touched_file_is_reprobed_exactly(self):
        self._scan()
        target = self.root / "root_0.wav"
        st = target.stat()
        os.utime(target, (st.st_atime, st.st_mtime + 100))

        fake = mock.MagicMock(return_value=dict(self._FAKE_INFO))
        with mock.patch("app.ui.workers.probe_audio", fake):
            self._scan()

        self.assertEqual(fake.call_count, 1)
        self.assertEqual(Path(fake.call_args[0][0]), target)
        self.assertEqual(self.recorder.finished_scan[-1],
                         (0, self.total, 0, 0, 0))
        self.assertEqual(len(self._tracks()), self.total)


class TestParallelProbing(ScanTestBase):
    def test_probes_run_on_multiple_threads(self):
        if (os.cpu_count() or 1) < 2:
            self.skipTest("single-core machine: probe pool has one worker")
        seen_threads: list[int] = []
        lock = threading.Lock()

        def slow_probe(path):
            with lock:
                seen_threads.append(threading.get_ident())
            time.sleep(0.02)  # hold this worker so siblings take other files
            return {"codec": "pcm_s16le", "sample_rate": 8000}

        with mock.patch("app.ui.workers.probe_audio", slow_probe):
            self._scan()  # 30 files in one folder, pool runs 8-wide here

        self.assertEqual(self.recorder.failed, [])
        self.assertEqual(self.recorder.finished_scan,
                         [(self.total, 0, 0, 0, 0)])
        self.assertGreaterEqual(len(set(seen_threads)), 2)


class TestProbeFailureIsolation(ScanTestBase):
    def test_one_bad_file_does_not_stall_the_scan(self):
        bad = self.root / "root_0.wav"

        def exploding_probe(path):
            if Path(path) == bad:
                raise RuntimeError("corrupt media")
            return {"codec": "pcm_s16le", "sample_rate": 8000,
                    "channels": 1, "duration_sec": 1.0}

        with mock.patch("app.ui.workers.probe_audio", exploding_probe):
            self._scan()

        self.assertEqual(self.recorder.failed, [])
        self.assertEqual(len(self.recorder.file_error), 1)
        self.assertEqual(Path(self.recorder.file_error[0][0]), bad)
        self.assertTrue(self.recorder.file_error[0][1].startswith(
            "Probe failed:"))
        self.assertEqual(self.recorder.finished_scan,
                         [(self.total, 0, 0, 1, 0)])
        # the bad file is still indexed (empty metadata), all progress emitted
        self.assertEqual(len(self._tracks()), self.total)
        self.assertEqual([p[0] for p in self.recorder.progress],
                         list(range(1, self.total + 1)))


if __name__ == "__main__":
    unittest.main()
