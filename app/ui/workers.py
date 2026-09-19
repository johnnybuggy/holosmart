"""Background workers for the GUI (QThread). Workers never touch widgets:
all communication happens through signals with plain Python payloads."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.audio.decode import probe_audio
from app.config import AppConfig
from app.db import repo
from app.db.database import Database
from app.models.registry import get_plugin

log = logging.getLogger(__name__)

#: Thread-local guard so a pool thread's priority is lowered exactly once.
_NICED = threading.local()


def _lower_thread_priority() -> None:
    """Drop the calling thread's scheduling priority below the GUI's.

    Analysis runs on a thread pool that saturates the CPU; nicing those
    threads (macOS/Linux lower priority per thread) keeps the Qt event
    loop snappy while analysis churns.  Increasing the nice value never
    needs privileges; Windows and exotic platforms are tolerated no-ops.
    """
    if getattr(_NICED, "done", False):
        return
    _NICED.done = True
    try:
        os.nice(5)
    except (OSError, AttributeError):  # pragma: no cover - platform variance
        pass

#: Parallel-probe pool size. Probing is subprocess/IO bound (an ffprobe launch
#: costs ~10-50 ms of mostly waiting), so a handful of concurrent workers
#: captures nearly all of the useful parallelism without flooding the machine
#: with processes.
_PROBE_WORKERS = min(8, os.cpu_count() or 4)

#: Probe futures kept in flight per folder while the main thread consumes
#: results in file order (bounds memory and process-spawn pressure).
_PROBE_LOOKAHEAD = _PROBE_WORKERS * 2

#: Upserts buffered before an intermediate commit. Batching amortises the
#: transaction cost over ~50 files; a crashed scan loses at most one batch of
#: upserts, and the per-folder missing-track prune commits together with the
#: folder's final batch, so tracks are never pruned on partially committed
#: state.
_COMMIT_EVERY = 50

#: Tolerance for comparing a stored REAL mtime against a fresh stat (float
#: seconds; comfortably covers SQLite round-trip and filesystem noise).
_MTIME_EPSILON = 1e-6

#: Track columns rebuilt from a stored row on a rescan-cache hit — everything
#: a fresh probe would return except the stat-derived filename/extension/
#: size_bytes/mtime, which the caller refreshes from the live stat.
_ROW_META_KEYS: tuple[str, ...] = (
    "container", "codec", "sample_rate", "channels", "bit_depth",
    "bitrate_kbps", "duration_sec", "title", "artist", "album", "genre",
    "year", "track_no",
)


def _probe_one(path: Path) -> tuple[dict[str, Any], str | None]:
    """Probe *path* on a pool thread. Never raises: a failed probe becomes
    ``({}, message)`` so one unreadable file cannot stall the scan."""
    try:
        return probe_audio(path), None
    except Exception as exc:  # corrupt/unreadable media
        return {}, f"Probe failed: {exc}"


def _cache_fresh(known: sqlite3.Row, stat: os.stat_result) -> bool:
    """True when the stored row still exactly describes *stat*'s file.

    Requires a previously successful probe (a codec was found) plus identical
    size and an mtime equal within :data:`_MTIME_EPSILON` — edited files and
    files whose earlier probe failed are always re-probed."""
    if not known["codec"] or known["size_bytes"] is None or known["mtime"] is None:
        return False
    return (known["size_bytes"] == stat.st_size
            and abs(known["mtime"] - stat.st_mtime) < _MTIME_EPSILON)


def _info_from_row(known: sqlite3.Row) -> dict[str, Any]:
    """Rebuild a probe-style info dict from a stored track row (cache hit)."""
    return {key: known[key] for key in _ROW_META_KEYS}


class _ScanUnit(NamedTuple):
    """One file's work item in the per-folder scan pipeline.

    ``future`` carries the parallel probe; it is ``None`` for units that were
    already resolved while queueing — a stat failure (``stat=None``, message
    in ``error``) or a rescan-cache hit (metadata prefilled in ``info``).
    """
    path: Path
    future: Future | None
    stat: os.stat_result | None
    known: sqlite3.Row | None
    info: dict[str, Any] | None
    error: str | None


def _prepare_unit(path: Path, conn: sqlite3.Connection,
                  pool: ThreadPoolExecutor) -> _ScanUnit:
    """Build the pipeline unit for *path*: stat it, consult the rescan cache,
    and either reuse the stored metadata or schedule a parallel probe."""
    try:
        stat = path.stat()
    except OSError as exc:  # cannot even stat the file
        return _ScanUnit(path=path, future=None, stat=None, known=None,
                         info=None, error=f"Unreadable file: {exc}")
    known = repo.get_track_by_path(conn, str(path))
    if known is not None and _cache_fresh(known, stat):
        return _ScanUnit(path=path, future=None, stat=stat, known=known,
                         info=_info_from_row(known), error=None)
    return _ScanUnit(path=path, stat=stat, known=known, info=None, error=None,
                     future=pool.submit(_probe_one, path))


class ScanWorker(QThread):
    """(Re-)scan the given folders: discover files, probe, upsert, prune missing.

    Two phases so the UI can show real progress: (1) walk the folder trees to
    discover music files — surfacing any protected/unreadable directories — then
    (2) probe + index each file with per-file progress updates. Phase 2 probes
    files on a small thread pool while consuming the results in walk order, and
    skips re-probing files whose stored row still matches the file's size and
    mtime (rescan cache). All database work stays on the worker thread over one
    batched-commit connection per scan.
    """

    scan_started = Signal(int)                       # total files discovered
    folder_scan_started = Signal(str)                # root path entering phase-2 indexing
    folder_scanned = Signal(str, int)                # path, tracks_found
    track_upserted = Signal(int, str)                # track_id, path
    progress = Signal(int, int, str)                 # current, total, filename
    file_error = Signal(str, str)                    # path, message
    permission_required = Signal(str, str)           # directory, message
    finished_scan = Signal(int, int, int, int, int)  # added, updated, removed, file_errors, denied_dirs
    failed = Signal(str)

    def __init__(self, db_path: Path | str, folder_paths: list[str], parent=None) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path)
        self._folders = list(folder_paths)

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            from app.fs_utils import walk_music_files

            db = Database(self._db_path)

            # ---- Phase 1: discover files + permission problems -------------
            per_folder: list[tuple[str, list[Path]]] = []
            denied_all: list[tuple[str, str]] = []
            total = 0
            for folder in self._folders:
                files, denied = walk_music_files(folder)
                per_folder.append((folder, files))
                denied_all.extend(denied)
                total += len(files)
            for directory, message in denied_all:
                self.permission_required.emit(directory, message)
            self.scan_started.emit(total)

            # ---- Phase 2: probe + index with per-file progress -------------
            # One connection serves the whole scan: opening a connection
            # re-applies the schema script, which used to cost more than the
            # upsert itself once per file. Commits are batched (_COMMIT_EVERY
            # upserts, plus once per folder together with the missing-track
            # prune), so a crashed scan loses at most one small batch and
            # never prunes the tracks of a half-committed folder.
            added = updated = removed = 0
            file_errors = 0
            processed = 0
            conn = db.connect()
            try:
                with ThreadPoolExecutor(
                        max_workers=_PROBE_WORKERS,
                        thread_name_prefix="probe") as pool:
                    for folder, files in per_folder:
                        row = repo.get_folder_by_path(conn, folder)
                        if row is None:
                            folder_id = repo.add_folder(conn, folder)
                            conn.commit()
                        else:
                            folder_id = int(row["id"])

                        # Announce that this folder's phase-2 indexing is
                        # beginning, right before its file loop starts (the
                        # UI highlights the folder's tree root while it runs).
                        self.folder_scan_started.emit(folder)

                        seen: set[str] = set()
                        since_commit = 0
                        # In-order pipeline over this folder's files: units
                        # are queued in walk order and consumed from the left,
                        # so progress is emitted 1..total in file order no
                        # matter when the parallel probes finish. The deque
                        # doubles as the in-flight bound — at most
                        # _PROBE_LOOKAHEAD probes run ahead of the consumer.
                        queue: deque[_ScanUnit] = deque()
                        next_idx = 0
                        while next_idx < len(files) or queue:
                            while (len(queue) < _PROBE_LOOKAHEAD
                                   and next_idx < len(files)):
                                queue.append(_prepare_unit(
                                    files[next_idx], conn, pool))
                                next_idx += 1

                            unit = queue.popleft()
                            path = unit.path
                            if unit.stat is None:  # could not even stat it
                                file_errors += 1
                                self.file_error.emit(str(path),
                                                     unit.error or "")
                                processed += 1
                                self.progress.emit(processed, total,
                                                   path.name)
                                continue

                            if unit.future is not None:
                                info, probe_error = unit.future.result()
                            else:  # rescan-cache hit: metadata from the row
                                info, probe_error = unit.info or {}, None
                            if probe_error is not None:
                                file_errors += 1
                                self.file_error.emit(str(path), probe_error)
                            info.setdefault("filename", path.name)
                            info.setdefault("extension", path.suffix.lower())
                            info["size_bytes"] = unit.stat.st_size
                            info["mtime"] = unit.stat.st_mtime

                            track_id = repo.upsert_track(
                                conn, folder_id, str(path), info)
                            seen.add(str(path))
                            if unit.known is None:
                                added += 1
                            else:
                                updated += 1
                            since_commit += 1
                            if since_commit >= _COMMIT_EVERY:
                                conn.commit()
                                since_commit = 0
                            self.track_upserted.emit(int(track_id), str(path))
                            processed += 1
                            self.progress.emit(processed, total, path.name)

                        removed += repo.delete_tracks_missing(conn, folder_id,
                                                              seen)
                        conn.commit()
                        self.folder_scanned.emit(folder, len(seen))
            except BaseException:
                try:
                    conn.rollback()  # never leave a half-committed folder
                except Exception:  # pragma: no cover - rollback is best-effort
                    log.debug("Rollback after scan failure failed",
                              exc_info=True)
                raise
            finally:
                conn.close()

            self.finished_scan.emit(added, updated, removed, file_errors,
                                    len(denied_all))
        except Exception as exc:
            log.exception("Scan failed")
            self.failed.emit(f"Scan failed: {exc}")


class _AnalysisStopped(BaseException):
    """Internal abort signal raised by AnalysisWorker's progress callback.

    Deliberately a BaseException (not Exception): the analysis pipeline's
    per-step ``except Exception`` handlers must not swallow it, while
    ``Database.transaction`` still rolls back cleanly because it catches
    BaseException.  It never escapes ``AnalysisWorker.run()``.
    """


#: Files longer than this many seconds are skipped by the analysis pre-pass
#: when ``config.analysis_skip_long_files`` is set (Settings → Performance).
#: The 20-minute threshold is fixed per the feature spec.
LONG_TRACK_SEC = 20.0 * 60.0


class AnalysisWorker(QThread):
    """Run the analysis pipeline over a list of track ids, in parallel.

    Skip policy: tracks whose row no longer exists are ignored, and — unless
    ``force_reanalyze`` is set — tracks that are FULLY analyzed are skipped,
    so batch runs (Analyze All, folder context-menu Analyze) never redo
    finished work. "Fully analyzed" means every enabled, available model has
    a vector on every chunk: a track analyzed with FFT only is NOT skipped
    when MERT-330M gets enabled — it is revisited and the incremental
    pipeline fills in the new model without touching the stored FFT
    results. The count is announced exactly once on
    :attr:`skipped` before processing starts (0 is emitted too); when every
    requested track is skipped the worker emits ``skipped`` + ``all_finished``
    and stops there. ``force_reanalyze=True`` analyzes every requested track
    regardless of status — the escape hatch used when a single file is
    explicitly selected for re-analysis (e.g. after changing model settings).

    Parallelism: ONLY FFT-only runs use a thread pool (sized to every CPU
    core but one — see :meth:`_effective_parallelism`).  Every run that
    involves any other model (CLAP/MERT/MERT-330M/OpenL3) is strictly
    sequential: torch model inference serializes on the GPU/MPS anyway,
    and interleaved multi-model runs only add contention, memory spikes
    and interleaved progress output.  The legacy
    ``config.analysis_parallelism`` setting no longer widens any run.
    Pool tasks touch the database only through their own short-lived
    connections and report through Qt signals, which are thread-safe;
    position/started/finished emissions from pool threads may therefore
    interleave, which the UI handlers tolerate. Model plugins are
    process-wide singletons whose first-use weight loading is serialized
    in ``ModelPlugin.ensure_loaded`` (inference itself is not serialized).

    Stop semantics: the flag is shared by all tasks. Every progress callback
    checks it FIRST and raises :class:`_AnalysisStopped`; on stop no new work
    starts, in-flight tracks abort at their next notify (each is reset to
    ``new`` so it stays retryable), queued tasks leave their tracks
    untouched, and ``stopped`` carries the number of successfully completed
    tracks once the pool has drained.
    """

    track_started = Signal(int, str)           # track_id, path
    track_progress = Signal(int, str)          # track_id, message
    track_chunk_progress = Signal(int, int, int, str)  # track_id, current, total, model/phase label
    # 1-based position within the requested batch, requested total, filename.
    # Position counts tracks actually started — a missing (skipped) row does
    # not advance it. With parallelism > 1 positions are assigned at start
    # time and may be announced out of order.
    track_position = Signal(int, int, str)
    track_finished = Signal(int, bool, str)    # track_id, ok, message
    all_finished = Signal()
    stopped = Signal(int)                      # tracks completed before a stop
    failed = Signal(str)
    # Emitted exactly once per run before processing starts: how many
    # requested tracks were skipped as already analyzed (0 included).
    skipped = Signal(int)
    # Emitted exactly once per run before processing starts: how many
    # requested tracks were skipped for being longer than 20 minutes
    # (config.analysis_skip_long_files, 0 included).
    skipped_long = Signal(int)

    def __init__(self, db_path: Path | str, config: AppConfig,
                 track_ids: list[int], parent=None,
                 force_reanalyze: bool = False,
                 skip_analyzed: bool = True) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path)
        self._config = config
        self._track_ids = list(track_ids)
        # False (default): tracks already analyzed are skipped. True: every
        # requested track is analyzed again (single-file escape hatch).
        self._force_reanalyze = bool(force_reanalyze)
        # False: already-analyzed tracks are visited too, but INCREMENTALLY —
        # analyze_track keeps their chunks and only fills in models that do
        # not cover every chunk yet (recursive folder analysis).
        self._skip_analyzed = bool(skip_analyzed)
        # Plain bool flag: CPython's GIL makes set/read safe for this purpose.
        self._stop_requested = False

    def request_stop(self) -> None:
        """Ask the worker to stop at the next pipeline step / track boundary.

        Safe to call from the UI thread at any time; idempotent.  The stop
        takes effect either inside a running track (at the next progress
        notification — the track is reset to ``new`` so it can be
        re-analyzed) or, for tracks not yet started, before they begin.
        """
        self._stop_requested = True

    # ------------------------------------------------------------- pre-pass --
    def _partition_requested(self) -> tuple[list[int], int, int]:
        """Split the requested ids into startable and skipped tracks.

        Returns ``(startable_ids, skipped_analyzed, skipped_long)``: rows
        that no longer exist are dropped entirely; tracks whose status is
        ``analyzed`` count as skipped unless ``force_reanalyze`` is set or
        ``skip_analyzed`` is False (recursive folder analysis visits them
        incrementally); so do tracks longer than
        :data:`LONG_TRACK_SEC` when the config asks for them to be skipped
        (``duration_sec`` unknown/None never skips). One connection serves
        the whole pre-pass (opening a connection re-applies the schema
        script, so per-id connections would dominate the cost).
        """
        startable: list[int] = []
        skipped = 0
        skipped_long = 0
        skip_long = (bool(getattr(self._config, "analysis_skip_long_files",
                                  False))
                     and not self._force_reanalyze)
        conn = Database(self._db_path).connect()
        available: dict[str, bool] = {}
        try:
            for track_id in self._track_ids:
                row = repo.get_track(conn, track_id)
                if row is None:
                    continue
                if (not self._force_reanalyze and self._skip_analyzed
                        and row["status"] == "analyzed"
                        and self._fully_analyzed(conn, int(row["id"]),
                                                 available)):
                    skipped += 1
                    continue
                if (skip_long and row["duration_sec"] is not None
                        and float(row["duration_sec"]) > LONG_TRACK_SEC):
                    skipped_long += 1
                    continue
                startable.append(track_id)
        finally:
            conn.close()
        return startable, skipped, skipped_long

    def _fully_analyzed(self, conn: sqlite3.Connection, track_id: int,
                        available: dict[str, bool]) -> bool:
        """True when every enabled, available model covers every chunk.

        The skip-policy completeness check.  ``available`` caches per-model
        plugin availability across the batch (a plugin check may touch the
        filesystem; model weights do not appear mid-run).
        """
        n_chunks, coverage = repo.track_model_coverage(conn, track_id)
        if n_chunks == 0:
            return False
        for name in self._config.models:
            is_available = available.get(name)
            if is_available is None:
                try:
                    plugin = get_plugin(name)
                    is_available = bool(plugin.is_available())
                except KeyError:
                    is_available = False
                available[name] = is_available
            if not is_available:
                continue   # cannot gain coverage this run — not required
            if coverage.get(name, 0) < n_chunks:
                return False
        return True

    @staticmethod
    def _effective_parallelism(config, n_startable: int) -> int:
        """Pool size for one analysis run, clamped to the workload.

        FFT-only runs are numpy-bound and scale: they may use all but one
        available CPU core (one is reserved for the GUI so the interface
        stays responsive while every remaining core is busy).  Every run
        that involves any model other than FFT runs SEQUENTIALLY — model
        inference (CLAP/MERT/MERT-330M/OpenL3) serializes on the GPU/MPS
        anyway, and parallel multi-model runs only add contention, memory
        spikes and interleaved progress output.  The legacy
        ``analysis_parallelism`` setting no longer widens any run.
        """
        models = set(getattr(config, "models", []) or [])
        if models == {"fft"}:
            cores = max(1, (os.cpu_count() or 2) - 1)  # one core for the GUI
            return max(1, min(cores, n_startable))
        return 1

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        from app.analysis.pipeline import analyze_track  # lazy: heavy siblings

        stop_msg = "Stopped by user — resume later"
        db = Database(self._db_path)
        total = len(self._track_ids)
        completed = 0
        # 1-based count of tracks actually started: a missing row consumes no
        # position, so the UI's "Analyzing 3/12" always refers to a track
        # that is really being analyzed. Positions are assigned under the
        # lock at start time, so with parallelism > 1 they may be announced
        # out of order — the UI handlers tolerate that.
        position = 0
        hit_stop = False
        # GIL makes plain reads/writes of the stop flag safe; the lock only
        # serializes the counter read-modify-writes across pool threads.
        lock = threading.Lock()

        def task(track_id: int) -> bool:
            """One pool task: the per-track body. True when it succeeded."""
            nonlocal completed, position, hit_stop
            if self._stop_requested:
                # Stop landed before this queued task could start: leave the
                # track untouched (no fetch, no signals, no status write).
                return False
            _lower_thread_priority()
            conn = db.connect()
            try:
                track = repo.get_track(conn, track_id)
            finally:
                conn.close()
            if track is None:
                return False
            path = track["path"]
            with lock:
                position += 1
                pos = position
            # Announced BEFORE track_started so the UI's position state
            # is ready when the start handler builds its status line.
            self.track_position.emit(pos, total, Path(path).name)
            self.track_started.emit(track_id, path)

            # Per-track tick throttling: the pipeline notifies once per
            # embed batch; with a full-core parallel run that floods the
            # UI event loop (one queued cross-thread signal per notify per
            # track).  Cap the delivered ticks at ~10 Hz per track — the
            # progress bar/status do not benefit from more.
            last_tick: dict[int, float] = {}

            def progress_cb(msg, cur=None, tot=None, _tid=track_id):
                # Flag checked FIRST so a stop surfaces at the very next
                # pipeline notify() step: _AnalysisStopped is a
                # BaseException, so analyze_track's `except Exception`
                # handlers cannot catch it while Database.transaction
                # still rolls back cleanly.  Per-chunk-batch notifications
                # make these checkpoints much more frequent (one per
                # model batch instead of one per coarse pipeline step).
                if self._stop_requested:
                    raise _AnalysisStopped(_tid)
                if cur is None:
                    # Coarse phase change (decoding, model start, tags…):
                    # always delivered.
                    self.track_progress.emit(_tid, msg)
                    return
                now = time.monotonic()
                due = now - last_tick.get(_tid, 0.0) >= 0.1
                if due:
                    last_tick[_tid] = now
                    self.track_chunk_progress.emit(
                        _tid, int(cur), int(tot or 0), msg)
                    self.track_progress.emit(_tid, msg)

            try:
                analyze_track(db, track_id, self._config,
                              progress_cb=progress_cb,
                              force=self._force_reanalyze)
            except _AnalysisStopped:
                # Interrupted track becomes retryable. Partially-written
                # chunks/embeddings from the aborted run are harmless —
                # re-analysis wipes them via repo.replace_chunks.
                conn = db.connect()
                try:
                    repo.set_track_status(conn, track_id, "new", stop_msg)
                    conn.commit()
                finally:
                    conn.close()
                self.track_finished.emit(track_id, False, stop_msg)
                hit_stop = True
                return False
            except RuntimeError as exc:
                self.track_finished.emit(track_id, False, str(exc))
                return False
            except Exception as exc:  # unexpected — still report per track
                log.exception("Analysis crashed for track %s", track_id)
                self.track_finished.emit(track_id, False,
                                         f"Unexpected error: {exc}")
                return False
            conn = db.connect()
            try:
                row = repo.get_track(conn, track_id)
                message = row["status_message"] if row else ""
            finally:
                conn.close()
            self.track_finished.emit(track_id, True, message or "Analyzed")
            with lock:
                completed += 1
            return True

        try:
            startable, skipped_count, skipped_long = self._partition_requested()
            # One immediate snapshot for the UI: how many requested tracks
            # are already analyzed and will not be redone (0 emitted too).
            self.skipped.emit(skipped_count)
            self.skipped_long.emit(skipped_long)
            if not startable:
                # Everything already analyzed (or nothing left to do): no
                # failed, no stopped — the UI only needs all_finished.
                return
            # FFT-only runs parallelize up to the CPU core count; mixed
            # runs use the configured degree (see _effective_parallelism).
            degree = self._effective_parallelism(self._config,
                                                 len(startable))

            if degree == 1:
                # Sequential fast path: run the bodies right here on the
                # worker thread — no pool overhead, and emissions keep the
                # historical threading (direct for same-thread observers).
                for track_id in startable:
                    if self._stop_requested:
                        break
                    task(track_id)
            else:
                with ThreadPoolExecutor(
                        max_workers=degree,
                        thread_name_prefix="analyze") as pool:
                    # Submit while nobody asked to stop. In-flight tasks keep
                    # running (the `with` exit drains the pool) and tasks that
                    # were already queued bail out on the flag check in task()
                    # before touching their track.
                    for track_id in startable:
                        if self._stop_requested:
                            break
                        pool.submit(task, track_id)
            if self._stop_requested or hit_stop:
                self.stopped.emit(completed)
        except Exception as exc:
            log.exception("Analysis worker failed")
            self.failed.emit(f"Analysis worker failed: {exc}")
        finally:
            self._post_run_normalization(db)
            self.all_finished.emit()

    def _post_run_normalization(self, db) -> None:
        """FFT per-component normalization after EVERY finished analysis.

        Database-wise and unconditional: no matter which models the run
        used — Analyze All, Analyze Selected, folder / single-file /
        context-menu runs, forced runs, stopped runs — the run ends with

        1. re-standardizing ALL stored FFT chunk vectors per component
           (whole dataset, not just the tracks of this run) and rebuilding
           the FFT track centroids, see :mod:`app.analysis.normalization`;
           a cheap existence check skips the work for libraries without
           FFT vectors; an already-standardized dataset rewrites nothing
           (idempotent);
        2. nothing else — the HDBSCAN/OPTICS noise flags are NOT
           touched by analysis runs any more; they are re-judged only
           when the user clicks the Chunks tab's "Find outliers" button
           (incrementally, for tracks not yet clustered — see
           :mod:`app.similarity.noise_filter`).

        Best-effort: a failure here is logged, never reported as an
        analysis failure.
        """
        conn = db.connect()
        try:
            n_fft = int(conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model = 'fft'"
            ).fetchone()[0])
        except Exception:
            log.exception("FFT normalization pre-check failed")
            return
        finally:
            conn.close()
        if n_fft >= 2:
            try:
                from app.analysis.normalization import normalize_model

                n_vectors, n_centroids = normalize_model(db, "fft")
                if n_vectors:
                    log.info("Post-run FFT normalization: %d vectors, "
                             "%d centroids", n_vectors, n_centroids)
            except Exception:
                log.exception("FFT per-component normalization failed")

class OllamaDetectWorker(QThread):
    """Startup probe: is Ollama alive and which embedding models are installed?"""

    detected = Signal(bool, list)   # running, embedding_model_names
    failed = Signal(str)

    def __init__(self, host: str, parent=None) -> None:
        super().__init__(parent)
        self._host = host

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            from app.similarity.ollama import detect_ollama
            running, models = detect_ollama(self._host)
            self.detected.emit(running, models)
        except Exception as exc:
            self.failed.emit(str(exc))


class SimilarSearchWorker(QThread):
    """Similarity search off the UI thread."""

    results_ready = Signal(list)    # list[SimilarResult]
    failed = Signal(str)

    def __init__(self, db_path: Path | str, seed_track_id,
                 dataset: str, algorithm: str, limit: int,
                 ollama_host: str, parent=None,
                 discard_noise: tuple[str, ...] = ()) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path)
        # One reference track, or several: multi-reference searches combine
        # the per-reference match percentages via their geometric mean.
        if isinstance(seed_track_id, (list, tuple)):
            self._seeds = [int(t) for t in seed_track_id]
        else:
            self._seeds = [int(seed_track_id)]
        self._dataset = dataset
        self._algorithm = algorithm
        self._limit = limit
        self._ollama_host = ollama_host
        self._discard_noise = tuple(discard_noise or ())

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            from app.similarity.ollama import OllamaClient
            from app.similarity.search import similar_tracks_multi

            db = Database(self._db_path)
            conn = db.connect()
            try:
                client = OllamaClient(self._ollama_host)
                client = client if client.is_running() else None
                results = similar_tracks_multi(conn, self._seeds,
                                               dataset=self._dataset,
                                               algorithm=self._algorithm,
                                               limit=self._limit,
                                               ollama=client,
                                               discard_noise=self._discard_noise)
            finally:
                conn.close()
            self.results_ready.emit(results)
        except RuntimeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            log.exception("Similarity search failed")
            self.failed.emit(f"Similarity search failed: {exc}")


class ReductionWorker(QThread):
    """Fit a dimensionality reduction over one model's chunk vectors.

    Runs off the UI thread (PCA/UMAP/t-SNE over a whole library take
    seconds to minutes) and reports every phase through signals so the
    dialog can show a progress bar:

    * ``stage`` — human-readable phase text ("Loading vectors…");
    * ``progress`` — 0..100 (the slow fit phase reports 10 and the dialog
      switches the bar to busy mode + elapsed time);
    * ``finished_ok`` — ``(reduction_id, name, n_vectors)`` after the
      reduced vectors AND the per-track centroids (stored under the model
      name ``red:<id>``) are committed;
    * ``failed`` — friendly error text.
    """

    stage = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(int, str, int)
    failed = Signal(str)

    def __init__(self, db_path: Path | str, source_model: str, method: str,
                 n_components: int, name: str, *, n_neighbors: int = 15,
                 min_dist: float = 0.1, perplexity: float = 30.0,
                 replace_reduction_id: int | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path)
        self._source_model = source_model
        self._method = method
        self._n_components = int(n_components)
        self._name = name
        self._n_neighbors = int(n_neighbors)
        self._min_dist = float(min_dist)
        self._perplexity = float(perplexity)
        # Refresh mode: re-fit OVER an existing reduction (keeps its id and
        # ``red:<id>`` references; the old vectors are replaced).
        self._replace_reduction_id = (
            None if replace_reduction_id is None
            else int(replace_reduction_id))
        # numba/llvmlite (UMAP's JIT engine) recurse deeply during
        # compilation and dispatch; on Qt's default ~512 KB secondary-
        # thread stack they die with a SIGBUS mid-fit.  A generous stack
        # makes large fits (84k chunks x 40 dims, ~1 min) survive.
        self.setStackSize(256 * 1024 * 1024)

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            from app.analysis import dim_reduction
            from app.similarity.search import compute_centroid

            db = Database(self._db_path)
            self.stage.emit(f"Loading '{self._source_model}' chunk vectors…")
            self.progress.emit(0)
            with db.transaction() as conn:
                rows = repo.get_chunk_embedding_rows(
                    conn, models=[self._source_model])
            if len(rows) < 4:
                self.failed.emit(
                    f"'{self._source_model}' has only {len(rows)} chunk "
                    "vectors — analyze more tracks first (at least a few "
                    "chunks are needed).")
                return
            matrix = np.stack([np.asarray(row["vec"], dtype=np.float64)
                               for row in rows])
            keys = [(int(row["chunk_id"]), int(row["track_id"]))
                    for row in rows]
            self.stage.emit(
                f"Fitting {self._method.upper()} on {len(matrix):,} × "
                f"{matrix.shape[1]} → {self._n_components} dims (long "
                "running — watch the elapsed time)…")
            self.progress.emit(10)
            coords, info = dim_reduction.fit_reduce(
                matrix, self._method, self._n_components,
                n_neighbors=self._n_neighbors, min_dist=self._min_dist,
                perplexity=self._perplexity)

            self.stage.emit("Storing reduced vectors…")
            with db.transaction() as conn:
                if self._replace_reduction_id is not None:
                    reduction_id = self._replace_reduction_id
                    repo.update_reduction(
                        conn, reduction_id, self._name, self._method,
                        {"n_neighbors": self._n_neighbors,
                         "min_dist": self._min_dist,
                         "perplexity": self._perplexity},
                        self._n_components)
                else:
                    try:
                        reduction_id = repo.create_reduction(
                            conn, self._name, self._source_model,
                            self._method,
                            {"n_neighbors": self._n_neighbors,
                             "min_dist": self._min_dist,
                             "perplexity": self._perplexity},
                            self._n_components)
                    except sqlite3.IntegrityError as exc:
                        raise RuntimeError(
                            f"A dataset named '{self._name}' already "
                            "exists — pick another name.") from exc
                repo.replace_reduced_embeddings(
                    conn, reduction_id,
                    [(chunk_id, vec) for (chunk_id, _tid), vec
                     in zip(keys, coords)])
                self.progress.emit(90)
                # Per-track centroids under "red:<id>" make the dataset
                # searchable by the centroid algorithm with no extra code.
                # A refresh must drop the previous ones first: re-analyzed
                # tracks changed chunk ids and removed tracks would keep a
                # dead centroid forever.
                conn.execute(
                    "DELETE FROM track_embeddings WHERE model = ?",
                    (f"red:{reduction_id}",))
                by_track: dict[int, list[np.ndarray]] = {}
                for (chunk_id, track_id), vec in zip(keys, coords):
                    by_track.setdefault(track_id, []).append(np.asarray(vec))
                for track_id, vecs in by_track.items():
                    centroid = compute_centroid(vecs)
                    if centroid.size:
                        repo.set_track_embedding(conn, track_id,
                                                 f"red:{reduction_id}",
                                                 centroid)
                repo.set_reduction_result(
                    conn, reduction_id, len(keys),
                    info.get("explained_variance"))
            self.progress.emit(100)
            self.finished_ok.emit(reduction_id, self._name, len(keys))
        except RuntimeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            log.exception("Dimensionality reduction failed")
            self.failed.emit(f"Dimensionality reduction failed: {exc}")


class NoiseFilterWorker(QThread):
    """Cluster one dataset's chunk vectors and store its noise flags.

    One-time, cached background run (HDBSCAN/OPTICS via scikit-learn —
    minutes for a large library).  See
    :mod:`app.similarity.noise_filter` for the semantics.

    Signals:

    * ``stage`` — human-readable phase text;
    * ``progress`` — 0..100, real per-song progress of the run;
    * ``finished_ok`` — ``(dataset, method, n_vectors, n_noise)``;
    * ``failed`` — friendly error text.
    """

    stage = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(str, str, int, int)
    failed = Signal(str)

    def __init__(self, db_path: Path | str, dataset: str, method: str,
                 parent=None) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path)
        self._dataset = str(dataset)
        self._method = str(method)
        # HDBSCAN/OPTICS recurse inside Cython; give the thread more room
        # than Qt's ~512 KB default (cheap insurance against hard crashes).
        self.setStackSize(64 * 1024 * 1024)

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            from app.similarity.noise_filter import fit_noise_filter

            def on_fit_progress(message: str, fraction: float) -> None:
                self.stage.emit(message)
                self.progress.emit(
                    max(0, min(100, int(round(fraction * 100.0)))))

            result = fit_noise_filter(
                self._db_path, self._dataset, self._method,
                progress_cb=on_fit_progress)
            self.finished_ok.emit(result["dataset"], result["method"],
                                  result["n_vectors"], result["n_noise"])
        except RuntimeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            log.exception("Noise filter run failed")
            self.failed.emit(f"Noise filter failed: {exc}")


class NoiseSweepWorker(QThread):
    """Button-driven incremental noise pass over EVERY covered dataset.

    Triggered by the Chunks tab's "Find outliers" button (analysis runs do
    NOT touch noise flags any more).  For every model dataset that has
    chunk vectors, only the PENDING songs are re-clustered — tracks not
    yet clustered, and re-analyzed tracks whose chunk ids changed — with
    BOTH methods, and the results are merged into the stored filters.

    Signals:

    * ``stage`` — human-readable phase text;
    * ``progress`` — 0..100;
    * ``finished_ok`` — list of per-(dataset, method) result dicts;
    * ``failed`` — friendly error text.
    """

    stage = Signal(str)
    progress = Signal(int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: Path | str, parent=None) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path)
        # HDBSCAN/OPTICS recurse inside Cython; give the thread more room
        # than Qt's ~512 KB default (cheap insurance against hard crashes).
        self.setStackSize(64 * 1024 * 1024)

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            from app.similarity.noise_filter import fit_noise_filter_pending

            def on_progress(message: str, fraction: float) -> None:
                self.stage.emit(message)
                self.progress.emit(
                    max(0, min(100, int(round(fraction * 100.0)))))

            with Database(self._db_path).connect() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT model FROM embeddings "
                    "ORDER BY model").fetchall()
            datasets = [str(r["model"]) for r in rows]
            results = fit_noise_filter_pending(
                self._db_path, datasets,
                progress_cb=on_progress)
            self.finished_ok.emit(results)
        except RuntimeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            log.exception("Noise sweep failed")
            self.failed.emit(f"Noise sweep failed: {exc}")


class LearningWorker(QThread):
    """Learn per-component similarity weights from the stored song pairs.

    Runs the pair-based optimizer (:mod:`app.learning.weights`) off the UI
    thread and persists the resulting weight vector for one model.  The
    Learning tab stays responsive while the optimizer works through the
    pairs; results arrive on :attr:`learned` (model, pairs used, weights)
    or :attr:`failed`.
    """

    learned = Signal(str, int, object)   # model, pairs_used, np.ndarray
    failed = Signal(str)

    def __init__(self, db_path: Path | str, model: str, parent=None) -> None:
        super().__init__(parent)
        self._db_path = Path(db_path)
        self._model = model

    def run(self) -> None:
        # Below-normal scheduling priority: the GUI thread wins CPU
        # time when an analysis/scan run saturates the machine.
        self.setPriority(QThread.Priority.LowPriority)
        try:
            from app.learning.weights import learn_and_store

            db = Database(self._db_path)
            summary = learn_and_store(db, self._model)
            self.learned.emit(self._model, int(summary["pairs_used"]),
                              summary["weights"])
        except RuntimeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            log.exception("Learning weights failed")
            self.failed.emit(f"Learning weights failed: {exc}")
