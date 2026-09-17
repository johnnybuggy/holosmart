"""Main window: toolbar, folder tree, detail pane, worker orchestration."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QSplitter, QVBoxLayout, QWidget,
)

from app.config import DB_PATH, AppConfig
from app.db import repo
from app.db.database import Database
from app.fs_utils import ACCESS_HELP_TEXT, check_read_access, open_privacy_settings
from app.playlist.generator import default_m3u_path, export_m3u
from app.ui.detail_pane import DetailPane
from app.ui.folder_tree import LibraryPane
from app.ui.settings_dialog import rank_embedding_models
from app.ui.system_player import open_in_system_player
from app.ui.workers import (
    AnalysisWorker, NoiseFilterWorker, OllamaDetectWorker, ScanWorker,
    SimilarSearchWorker,
)

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """HoloSmart Music Explorer main window."""

    def __init__(self, config: AppConfig, db_path: Path | str | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._db_path = Path(db_path) if db_path else Path(DB_PATH)
        self._db = Database(self._db_path)
        self._scan_worker: ScanWorker | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._analysis_track_path: str | None = None
        # Position of the track being analyzed within its requested batch
        # (for the "Analyzing 3/12 — song.mp3" status line); None until the
        # first track_position signal of the current run.
        self._analysis_position: int | None = None
        self._analysis_total: int | None = None
        self._analysis_filename: str | None = None
        # Wall-clock start (time.monotonic) of the current analysis run for
        # the "Speed: N min/h" readout; None while no run is live — plus the
        # audio minutes accumulated so far in that run.
        self._speed_started_at: float | None = None
        self._speed_audio_minutes: float = 0.0
        # Root folder paths whose phase-2 scan indexing is running right now.
        self._scanning_paths: list[str] = []
        # Tracks completed before a user stop (None = no stop requested);
        # also written directly by analyze_track_ids at run start.
        self._analysis_stopped_at: int | None = None
        self._similar_worker: SimilarSearchWorker | None = None
        self._last_similar_search: tuple | None = None
        self._noise_filter_worker: NoiseFilterWorker | None = None
        self._ollama_worker: OllamaDetectWorker | None = None
        self._similar_seed: int | None = None
        # Tracks currently under analysis (track id -> path): feeds the
        # tree's yellow "analyzing" highlight for files and their folders.
        self._analyzing: dict[int, str] = {}
        self._scan_denied: list[tuple[str, str]] = []   # (directory, error)
        self._perm_dialog_shown = False
        self._pending_add: str | None = None
        # Non-modal Visualisation scatter-plot window; created on demand.
        self._viz_dialog: "VisualisationDialog | None" = None

        self.setWindowTitle("HoloSmart Music Explorer")
        # Wide enough that the tree pane shows the filename column AND the
        # whole status area (Status + one column per model plugin) without
        # horizontal scrolling; _autosize_name_column keeps its end of the
        # bargain by capping the filename column to the leftover width.
        self.resize(1500, 940)
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._tree.refresh()
        self._start_ollama_detection()
        self._suggest_desktop_if_empty()

    # ------------------------------------------------------------------ UI ---
    def _build_central(self) -> None:
        from app.config import analysis_excluded_extensions

        splitter = QSplitter(Qt.Orientation.Horizontal)
        pane = LibraryPane(
            self._db,
            excluded_extensions=analysis_excluded_extensions(self._config))
        self._tree = pane.tree
        self._tree.track_selected.connect(self._on_track_selected)
        splitter.addWidget(pane)

        self._details = DetailPane(self._db, self._config)
        self._details.search_requested.connect(self._on_search_similar)
        self._details.folder_references_requested.connect(
            self._on_add_folder_references)
        self._details.noise_filter_toggled.connect(self._on_noise_filter_toggled)
        self._details.playlist_requested.connect(self._on_create_playlist)
        self._details.playlist_play_requested.connect(
            self._on_create_playlist_and_play)
        self._details.play_track_requested.connect(self._on_play_track)
        self._tree.track_play_requested.connect(self._on_play_track)
        self._details.playlist_changed.connect(lambda: None)
        splitter.addWidget(self._details)
        splitter.setSizes([620, 880])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Main")
        bar.setMovable(False)

        self._act_add = QAction("Add Folder…", self)
        self._act_add.triggered.connect(self.add_folder)
        bar.addAction(self._act_add)

        self._act_remove = QAction("Remove Folder", self)
        self._act_remove.triggered.connect(self.remove_selected_folder)
        bar.addAction(self._act_remove)

        bar.addSeparator()
        self._act_rescan = QAction("Rescan", self)
        self._act_rescan.setShortcut(QKeySequence("F5"))
        self._act_rescan.triggered.connect(self.rescan_all)
        bar.addAction(self._act_rescan)

        bar.addSeparator()
        self._act_analyze = QAction("Analyze Selected", self)
        self._act_analyze.triggered.connect(self.analyze_selected)
        bar.addAction(self._act_analyze)

        self._act_analyze_all = QAction("Analyze All", self)
        self._act_analyze_all.setToolTip(
            "Analyze every track that is not fully analyzed yet — already "
            "analyzed tracks are skipped, so after stopping or interrupting "
            "a run this continues exactly where it left off.")
        self._act_analyze_all.triggered.connect(self.analyze_all)
        bar.addAction(self._act_analyze_all)

        self._act_clear = QAction("Clear Analysis", self)
        self._act_clear.setToolTip(
            "Clear analysis results for the selected file or folder "
            "(chunks, embeddings, tags, descriptions)")
        self._act_clear.triggered.connect(self.clear_analysis_results)
        bar.addAction(self._act_clear)

        self._act_stop = QAction("Stop Analysis", self)
        self._act_stop.setEnabled(False)
        self._act_stop.triggered.connect(self.stop_analysis)
        bar.addAction(self._act_stop)

        bar.addSeparator()
        self._act_viz = QAction("Visualisation", self)
        self._act_viz.setToolTip(
            "Scatter plots of the analysis datasets (CLAP / MERT / FFT / "
            "OpenL3 components, optionally PCA / t-SNE / UMAP reduced)")
        self._act_viz.triggered.connect(self.open_visualisation)
        bar.addAction(self._act_viz)

        self._act_settings = QAction("Settings…", self)
        self._act_settings.triggered.connect(self.open_settings)
        bar.addAction(self._act_settings)

    def _build_statusbar(self) -> None:
        self._speed_label = QLabel("")
        self._speed_label.setToolTip(
            "Audio processing speed: minutes of audio analyzed per hour of "
            "wall time")
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setMaximumWidth(220)
        self.statusBar().addPermanentWidget(self._speed_label)
        self.statusBar().addPermanentWidget(self._progress)
        self._status(f"Library: {self._db_path}")

    def _status(self, message: str) -> None:
        self.statusBar().showMessage(message, 0)

    def _suggest_desktop_if_empty(self) -> None:
        conn = self._db.connect()
        try:
            empty = not repo.list_folders(conn)
        finally:
            conn.close()
        if empty:
            desktop = Path.home() / "Desktop"
            self._status("Library is empty — use 'Add Folder…' "
                         f"(e.g. {desktop}) to index your music.")

    # ------------------------------------------------------------- folders ---
    def add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add music folder",
                                                str(Path.home() / "Desktop"))
        if not path:
            return
        self._try_add_folder(path)

    def _try_add_folder(self, path: str) -> bool:
        """Add + scan ``path`` after a read-access pre-check (triggers the macOS
        permission prompt on first access to protected folders)."""
        ok, error = check_read_access(path)
        if not ok:
            self._pending_add = path
            self._show_permission_dialog([(path, error)], retry_add=True)
            return False
        self._pending_add = None
        with self._db.transaction() as conn:
            repo.add_folder(conn, path)
        self._scan_folders([path])
        return True

    def remove_selected_folder(self) -> None:
        folder_id = self._tree.selected_folder_id()
        if folder_id is None:
            self._status("Select a folder in the tree first.")
            return
        conn = self._db.connect()
        try:
            row = conn.execute("SELECT path FROM folders WHERE id = ?",
                               (folder_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return
        answer = QMessageBox.question(
            self, "Remove folder",
            f"Remove '{row['path']}' and all its indexed tracks from the library?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        with self._db.transaction() as conn:
            repo.remove_folder(conn, folder_id)
        self._tree.refresh()
        self._details.show_track(None)
        self._status(f"Removed folder: {row['path']}")

    def rescan_all(self) -> None:
        conn = self._db.connect()
        try:
            folders = [row["path"] for row in repo.list_folders(conn)]
        finally:
            conn.close()
        if not folders:
            self._status("No folders to scan — add one first.")
            return
        self._scan_folders(folders)

    # -------------------------------------------------------------- workers ---
    def _set_busy(self, busy: bool) -> None:
        self._progress.setRange(0, 0 if busy else 1)
        for action in (self._act_add, self._act_remove, self._act_rescan,
                       self._act_analyze, self._act_analyze_all,
                       self._act_clear):
            action.setEnabled(not busy)
        # Stop only applies to a running analysis (a scan cannot be stopped):
        # enabled while busy AND an analysis worker exists and is still running.
        analysis_running = (self._analysis_worker is not None
                            and self._analysis_worker.isRunning())
        self._act_stop.setEnabled(busy and analysis_running)

    def _scan_folders(self, folders: list[str]) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._status("Scan already running…")
            return
        self._scan_denied = []
        self._perm_dialog_shown = False
        self._scanning_paths = []   # yellow-highlight accumulation, fresh run
        self._set_busy(True)
        self._progress.setRange(0, 0)   # indeterminate until discovery finishes
        self._status(f"Scanning {len(folders)} folder(s) — discovering files…")
        self._scan_worker = ScanWorker(self._db_path, folders)
        self._scan_worker.scan_started.connect(self._on_scan_started)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.folder_scan_started.connect(
            self._on_folder_scan_started)
        self._scan_worker.permission_required.connect(self._on_permission_denied)
        self._scan_worker.file_error.connect(self._on_file_error)
        self._scan_worker.track_upserted.connect(
            lambda tid, path: None)  # per-file visibility comes via progress
        self._scan_worker.finished_scan.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _on_scan_started(self, total: int) -> None:
        self._progress.setRange(0, max(1, int(total)))
        self._progress.setValue(0)
        self._status(f"Discovered {total} music file(s) — indexing…")

    def _on_scan_progress(self, current: int, total: int, filename: str) -> None:
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(current)
        self._status(f"Scanning {current}/{total} — {filename}")

    def _on_file_error(self, path: str, message: str) -> None:
        self._status(f"Skipped: {path} ({message})")

    def _on_permission_denied(self, directory: str, message: str) -> None:
        self._scan_denied.append((directory, message))
        if not self._perm_dialog_shown:
            self._perm_dialog_shown = True
            self._show_permission_dialog([(directory, message)])

    def _show_permission_dialog(self, denied: list[tuple[str, str]],
                                retry_add: bool = False) -> None:
        """Explain an OS read-access denial and offer actionable fixes."""
        detail = "\n".join(f"• {path}" + (f" — {error}" if error else "")
                           for path, error in denied[:8])
        if len(denied) > 8:
            detail += f"\n… and {len(denied) - 8} more"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Folder access required")
        box.setText("The operating system denied read access to a protected folder.")
        box.setInformativeText(ACCESS_HELP_TEXT +
                               ("\n\nAffected:\n" + detail if detail else ""))
        open_btn = box.addButton("Open Privacy & Security",
                                 QMessageBox.ButtonRole.ActionRole)
        reveal_btn = box.addButton("Show in Finder",
                                   QMessageBox.ButtonRole.ActionRole)
        retry_btn = box.addButton("Retry", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        clicked = box.clickedButton()
        if clicked is open_btn:
            open_privacy_settings()
        elif clicked is reveal_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(denied[0][0]))
        elif clicked is retry_btn:
            if retry_add and self._pending_add:
                pending = self._pending_add
                QTimer.singleShot(0, lambda: self._try_add_folder(pending))
            else:
                QTimer.singleShot(0, self.rescan_all)

    def _on_scan_finished(self, added: int, updated: int, removed: int,
                          file_errors: int = 0, denied_dirs: int = 0) -> None:
        self._set_busy(False)
        sel = self._tree.selected_track_id()
        self._tree.refresh()
        if sel is not None:
            self._tree.select_track(sel)
        self._tree.set_scanning_paths([])
        self._scanning_paths = []
        summary = (f"Scan complete — {added} added, {updated} updated, "
                   f"{removed} removed.")
        if file_errors:
            summary += f" {file_errors} file(s) had read/probe errors."
        self._status(summary)
        if denied_dirs and not self._perm_dialog_shown:
            self._show_permission_dialog(self._scan_denied)

    def _on_scan_failed(self, message: str) -> None:
        self._set_busy(False)
        sel = self._tree.selected_track_id()
        self._tree.refresh()
        if sel is not None:
            self._tree.select_track(sel)
        self._tree.set_scanning_paths([])
        self._scanning_paths = []
        # This handler serves scan AND analysis worker failures: only end the
        # speed readout when a (finished) analysis worker produced the failure.
        if (self._analysis_worker is not None
                and not self._analysis_worker.isRunning()):
            self._speed_label.setText("")
            self._speed_started_at = None
        self._status(message)
        QMessageBox.warning(self, "Scan failed", message)

    def _on_folder_scan_started(self, path: str) -> None:
        """A folder's phase-2 indexing began: mark its tree root yellow
        alongside the folders already being indexed in this scan run."""
        self._scanning_paths = [*self._scanning_paths, path]
        self._tree.set_scanning_paths(self._scanning_paths)

    def analyze_selected(self) -> None:
        ids = self._tree.selected_track_ids_in_folder()
        if not ids:
            self._status("Select a track or folder to analyze.")
            return
        # Escape hatch: explicitly analyzing a SINGLE selected file re-analyzes
        # it even if it is already analyzed (e.g. after changing model
        # settings); folder or multi-track selections keep the skip policy.
        force = len(ids) == 1 and self._tree.selected_track_id() is not None
        self.analyze_track_ids(ids, force=force)

    def analyze_all(self) -> None:
        conn = self._db.connect()
        try:
            ids = [int(r["id"]) for r in repo.list_tracks(conn)]
        finally:
            conn.close()
        if not ids:
            self._status("Library is empty — nothing to analyze.")
            return
        self.analyze_track_ids(ids)

    def _split_wav_tracks(self, track_ids: list[int]) -> tuple[list[int], int]:
        """Drop WAV tracks from a batch when ``analyze_wav`` is off (default).

        Returns ``(kept_ids, wav_skipped_count)``.  WAV files stay in the
        library and remain playable; they are only excluded from analysis
        runs because the raw PCM decode + embed work is disproportionately
        expensive and WAV is rarely the primary format.  One DB read serves
        the whole batch.
        """
        if bool(getattr(self._config, "analyze_wav", False)) or not track_ids:
            return list(track_ids), 0
        conn = self._db.connect()
        try:
            placeholders = ", ".join("?" for _ in track_ids)
            rows = conn.execute(
                f"SELECT id, extension FROM tracks WHERE id IN ({placeholders})",
                [int(t) for t in track_ids]).fetchall()
        finally:
            conn.close()
        kept: list[int] = []
        wav = 0
        for row in rows:
            if (row["extension"] or "").lower() == ".wav":
                wav += 1
            else:
                kept.append(int(row["id"]))
        return kept, wav

    def analyze_track_ids(self, track_ids: list[int], force: bool = False,
                          skip_analyzed: bool = True) -> None:
        """Start analyzing *track_ids* on a background worker.

        ``force=False`` (default) skips tracks that are fully analyzed —
        every enabled model already has a vector on every chunk — so batch
        runs (Analyze All) never redo finished work. A track analyzed with
        FFT only is NOT skipped after enabling MERT-330M: it is revisited
        and only the missing model is computed (incrementally).
        ``force=True`` analyzes every requested track again (single-file
        escape hatch after changing model settings).  ``skip_analyzed=False``
        visits already-analyzed tracks too, but incrementally: their chunks
        are kept and only models that do not cover every chunk yet run —
        this is how the recursive folder analysis fills a newly enabled
        model into a whole subtree without losing anything.

        WAV files are excluded from batch runs unless ``analyze_wav`` is
        enabled in the config (default off); an explicitly forced
        single-file analysis ignores the exclusion.
        """
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            self._status("Analysis already running…")
            return
        self._analysis_stopped_at: int | None = None
        self._analysis_position = None
        self._analysis_total = None
        self._analysis_filename = None
        if not force:
            track_ids, wav_skipped = self._split_wav_tracks(track_ids)
        else:
            wav_skipped = 0
        if not track_ids:
            self._status("Nothing to analyze (WAV files are excluded by "
                         "default — enable analyze_wav in the config)."
                         if wav_skipped else
                         "Nothing to analyze.")
            return
        models = ", ".join(self._config.models) or "no models"
        message = (f"Analyzing {len(track_ids)} track(s) with: {models} "
                   f"({self._config.chunk_seconds:.0f}s chunks, "
                   f"{self._config.overlap_percent:.0f}% overlap)")
        if not force:
            if skip_analyzed:
                message += " (fully-analyzed skipped)"
            if wav_skipped:
                message += f", {wav_skipped} WAV file(s) excluded"
        self._status(message)
        self._analysis_worker = AnalysisWorker(self._db_path, self._config,
                                               track_ids,
                                               force_reanalyze=force,
                                               skip_analyzed=skip_analyzed)
        # Connected BEFORE track_started: the worker emits track_position
        # first, so the start handler already knows position/total/filename
        # when it composes its status line.
        self._analysis_worker.track_position.connect(
            self._on_track_analysis_position)
        self._analysis_worker.track_started.connect(self._on_track_analysis_started)
        self._analysis_worker.track_progress.connect(
            lambda tid, msg: self._status(self._analysis_status(msg)))
        # Connected AFTER track_progress so, on a chunk tick (both signals
        # fire), the richer per-chunk status line below wins.
        self._analysis_worker.track_chunk_progress.connect(
            self._on_track_chunk_progress)
        self._analysis_worker.track_finished.connect(self._on_track_analysis_finished)
        self._analysis_worker.stopped.connect(self._on_analysis_stopped)
        self._analysis_worker.skipped.connect(self._on_analysis_skipped)
        self._analysis_worker.skipped_long.connect(self._on_analysis_skipped_long)
        self._analysis_worker.all_finished.connect(self._on_all_analyzed)
        self._analysis_worker.failed.connect(self._on_scan_failed)
        self._analysis_worker.start()
        # Busy-state set after start() so _set_busy sees the running worker
        # and thereby enables the Stop action (it stays disabled for scans).
        self._set_busy(True)

    def clear_analysis_results(self) -> None:
        """Delete all analysis results of the selected file or folder.

        After confirmation, every selected track loses its chunks (with
        embeddings and tags), its ``track_embeddings`` aggregates and its
        description; the status resets to ``new`` so the normal Analyze
        paths pick the tracks up again.  Refuses to run while an analysis
        is in flight — the worker may still be writing results for the very
        same tracks.
        """
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            self._status("Stop the running analysis before clearing results.")
            return
        ids = self._tree.selected_track_ids_in_folder()
        if not ids:
            self._status("Select a track or folder to clear.")
            return
        answer = QMessageBox.question(
            self, "Clear analysis results",
            f"Clear analysis results for {len(ids)} track(s)?\n\n"
            "Chunks, embeddings, tags and descriptions will be deleted. "
            "The tracks can be re-analyzed afterwards.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        with self._db.transaction() as conn:
            for track_id in ids:
                repo.clear_track_analysis(conn, track_id)
        sel = self._tree.selected_track_id()
        self._tree.refresh()
        if sel is not None:
            self._tree.select_track(sel)
        current = self._details.current_track_id()
        if current is not None:
            self._details.show_track(current)
        self._status(f"Cleared analysis results for {len(ids)} track(s).")

    def _on_analysis_skipped(self, count: int) -> None:
        """Status note for tracks the worker skipped as already analyzed.

        The worker emits ``skipped`` once per run before processing starts
        (0 included); a zero count stays quiet.
        """
        count = int(count)
        if count > 0:
            self._status(f"Skipping {count} already-analyzed track(s) — "
                         "analyze a single selected file to force re-analysis.")

    def _on_analysis_skipped_long(self, count: int) -> None:
        """Status note for tracks skipped as longer than 20 minutes.

        The worker emits ``skipped_long`` once per run before processing
        starts (0 included); a zero count stays quiet.
        """
        count = int(count)
        if count > 0:
            self._status(f"Skipped {count} file(s) longer than 20 minutes "
                         "(Settings → Performance).")

    def stop_analysis(self) -> None:
        """Ask the running analysis to stop; it can be resumed later by
        re-running Analyze Selected / Analyze All."""
        if (self._analysis_worker is None
                or not self._analysis_worker.isRunning()):
            self._status("No analysis is currently running.")
            return
        self._analysis_worker.request_stop()
        self._status("Stopping analysis…")

    def _on_analysis_stopped(self, completed: int) -> None:
        self._analysis_stopped_at = completed

    def _analysis_status(self, suffix: str = "") -> str:
        """Status line for the track being analyzed, e.g.
        "Analyzing 3/12 — song.mp3" plus " — {suffix}" when a coarse step
        ("Decoding audio") or a per-chunk-batch label ("CLAP: chunk 7/24")
        is given.

        Before the first position signal of a run (position unknown) the
        line degrades gracefully — never showing "None/None".
        """
        name = self._analysis_filename
        if name is None and self._analysis_track_path:
            name = Path(self._analysis_track_path).name
        if self._analysis_position is None or self._analysis_total is None:
            head = f"Analyzing {name}" if name else "Analyzing"
        else:
            head = (f"Analyzing {self._analysis_position}/"
                    f"{self._analysis_total} — {name}")
        return f"{head} — {suffix}" if suffix else head

    def _on_track_analysis_position(self, position: int, total: int,
                                    filename: str) -> None:
        """Remember the current track's 1-based position in the requested
        batch (total = number of requested track ids)."""
        self._analysis_position = int(position)
        self._analysis_total = int(total)
        self._analysis_filename = filename

    def _on_track_analysis_started(self, track_id: int, path: str) -> None:
        self._analysis_track_path = path
        self._status(self._analysis_status())
        worker = self._analysis_worker
        if worker is not None and not worker.isRunning():
            # Stale queued signal — the run already ended (e.g. stopped by the
            # user).  Do not clobber the worker's final per-track status
            # ("new" + resume hint after a stop) with a late "analyzing".
            return
        # Next model's chunk total is unknown until its first chunk tick:
        # show an indeterminate bar again for the new track.
        self._progress.setRange(0, 0)
        with self._db.transaction() as conn:
            repo.set_track_status(conn, track_id, "analyzing")
        # In-place status update instead of refresh(): a full rebuild here
        # would reset the user's selection/scroll on every track start.
        self._tree.update_track_status(track_id, "analyzing", None)
        # Yellow highlight for this file and its folder chain (parallel runs
        # keep every in-flight track's chain lit until its own finish).
        self._analyzing[int(track_id)] = str(path)
        self._tree.set_analyzing_paths(list(self._analyzing.values()))
        # Lazy speed-clock init: the first track start of a run begins the
        # wall-time measurement (keeps analyze_track_ids untouched).
        if self._speed_started_at is None:
            self._speed_started_at = time.monotonic()
            self._speed_audio_minutes = 0.0

    def _on_track_chunk_progress(self, track_id: int, current: int,
                                 total: int, label: str) -> None:
        """Per-chunk-batch tick from the analysis pipeline: the status-bar
        progress bar becomes determinate for the current model phase
        (e.g. "CLAP: chunk 6/32") and the status line carries the track's
        position/name ("Analyzing 3/12 — song.mp3 — CLAP: chunk 6/32").
        Scans keep managing the bar themselves — this handler only runs
        for analysis worker signals."""
        self._progress.setRange(0, max(1, int(total)))
        self._progress.setValue(int(current))
        self._status(self._analysis_status(label))

    def _on_track_analysis_finished(self, track_id: int, ok: bool, message: str) -> None:
        # A successful analysis can still carry per-model failure notes
        # (e.g. "MERT failed: ...") — surface them instead of a plain ✓.
        text = (message or "").strip() or ("Analyzed" if ok else "Failed")
        if ok and "failed:" in text:
            self._status("⚠ " + text[:200])
        else:
            self._status(("✓ " if ok else "✗ ") + text[:200])
        if self._details.current_track_id() == track_id:
            self._details.show_track(track_id)
        # In-place glyph/percentage update — no rebuild, selection preserved.
        self._tree.update_track_status(
            track_id, "analyzed" if ok else "error", message)
        # This file is done: un-light it and its folder chain (other tracks
        # of a parallel run stay highlighted until they finish).
        self._analyzing.pop(int(track_id), None)
        self._tree.set_analyzing_paths(list(self._analyzing.values()))
        if ok:
            self._update_speed_label(track_id)

    def _update_speed_label(self, track_id: int) -> None:
        """Fold a finished track's audio duration into the run's
        "Speed: N min/h" status-bar readout (minutes of audio analyzed per
        hour of wall time).

        Only meaningful while a run is live (``_speed_started_at`` set).
        Multiple near-simultaneous finishes simply accumulate — signals all
        arrive on the UI thread, so a plain float attribute suffices — and
        elapsed wall time keeps growing across the whole run.  The first
        second is skipped so the rate starts out stable (previous text kept).
        """
        if self._speed_started_at is None:
            return
        conn = self._db.connect()
        try:
            row = repo.get_track(conn, track_id)
        finally:
            conn.close()
        if row is not None and row["duration_sec"]:
            self._speed_audio_minutes += float(row["duration_sec"]) / 60.0
        elapsed = time.monotonic() - self._speed_started_at
        if elapsed < 1.0:
            return   # too early for a meaningful rate — keep previous text
        speed = self._speed_audio_minutes / (elapsed / 3600.0)
        value = f"{speed:.1f}" if speed < 10 else f"{speed:.0f}"
        self._speed_label.setText(f"Speed: {value} min/h")

    def _on_all_analyzed(self) -> None:
        self._set_busy(False)
        # Defensive: every started track already popped itself on finish; a
        # queued-but-never-started track was never added. Clear + repaint so
        # no stray yellow can survive the run.
        self._analyzing.clear()
        self._tree.set_analyzing_paths([])
        sel = self._tree.selected_track_id()
        self._tree.refresh()
        self._details.refresh_datasets()
        if sel is not None:
            self._tree.select_track(sel)
        # Run is over: the final refresh already shows every status, and the
        # speed readout belongs to that run only.
        self._speed_label.setText("")
        self._speed_started_at = None
        current = self._details.current_track_id()
        if current is not None:
            self._details.show_track(current)
        if self._analysis_stopped_at is not None:
            self._status(f"Analysis stopped — {self._analysis_stopped_at} "
                         "track(s) done; use Analyze Selected / Analyze All "
                         "to continue later.")
            self._analysis_stopped_at = None
        else:
            self._status("Analysis run complete.")

    # --------------------------------------------------------------- similar ---
    def _on_search_similar(self, seed_track_ids, dataset: str,
                           algorithm: str, limit: int,
                           discard_noise: tuple = ()) -> None:
        if self._similar_worker is not None and self._similar_worker.isRunning():
            self._status("Search already running…")
            return
        seeds = ([int(t) for t in seed_track_ids]
                 if isinstance(seed_track_ids, (list, tuple))
                 else [int(seed_track_ids)])
        self._similar_seed = seeds[0] if seeds else None
        self._last_similar_search = (seeds, dataset, algorithm,
                                     limit, tuple(discard_noise or ()))
        refs_note = (f", {len(seeds)} references (geometric mean)"
                     if len(seeds) > 1 else "")
        noise_note = (f", noise filter: {'+'.join(discard_noise)}"
                      if discard_noise else "")
        self._status(f"Searching similar tracks (dataset={dataset}, "
                     f"algorithm={algorithm}{refs_note}{noise_note})…")
        self._similar_worker = SimilarSearchWorker(
            self._db_path, seeds, dataset, algorithm, limit,
            self._config.ollama_host, discard_noise=tuple(discard_noise or ()))
        self._similar_worker.results_ready.connect(self._on_similar_results)
        self._similar_worker.failed.connect(self._on_similar_failed)
        self._similar_worker.start()

    #: Guard against multi-minute freezes: a folder reference set makes the
    #: search run one full candidate pass PER reference, so very large
    #: subtrees are refused with an explanation instead of hanging the UI.
    MAX_FOLDER_REFERENCES = 400

    def _on_add_folder_references(self) -> None:
        """Add every track of the selected tree folder as Similar references."""
        ids = self._tree.folder_track_ids_for_references()
        if not ids:
            self._status("Select a folder (or a file inside one) in the "
                         "tree to add it as references.")
            return
        if len(ids) > self.MAX_FOLDER_REFERENCES:
            self._status(
                f"Folder holds {len(ids)} tracks — the reference list is "
                f"capped at {self.MAX_FOLDER_REFERENCES} (each reference "
                "runs a full candidate pass). Add a smaller folder.")
            return
        added = self._details.add_reference_tracks(ids)
        self._status(f"Added {added} folder track(s) as Similar references "
                     f"({len(ids)} in folder).")

    def _rerun_similar_search(self) -> None:
        """Repeat the last Similar search with the current filter state."""
        if self._similar_worker is not None and self._similar_worker.isRunning():
            return   # a finishing search will already reflect fresh state
        params = getattr(self, "_last_similar_search", None)
        if params is None:
            return
        self._on_search_similar(*params)

    def _on_noise_filter_toggled(self, dataset: str, method: str,
                                 on: bool) -> None:
        """A Chunks-tab noise checkbox changed state.

        Checked but no cached run for this dataset -> launch the one-time
        clustering (results re-search automatically).  Otherwise just
        re-run the last search so the results reflect the new filter.
        """
        # (the pane already re-renders its Chunks table on toggle)
        if not on:
            self._rerun_similar_search()
            return
        with self._db.transaction() as conn:
            stored = repo.get_noise_filter(conn, dataset, method)
        if stored is not None:
            self._rerun_similar_search()
            return
        self._start_noise_filter_run(dataset, method)

    def _start_noise_filter_run(self, dataset: str, method: str) -> None:
        """Run the (long) clustering with a busy progress dialog."""
        from PySide6.QtWidgets import QProgressDialog

        n_hint = ""
        try:
            with self._db.transaction() as conn:
                counts = repo.chunk_vector_counts(conn)
            if dataset in counts:
                n_hint = f"{counts[dataset]:,} chunks — "
        except Exception:
            pass
        progress = QProgressDialog(
            f"Clustering {n_hint}one-time run for the {method.upper()} "
            "noise filter…\nSearches resume automatically when it "
            "finishes.", "", 0, 100, self)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumWidth(420)
        progress.setMinimumDuration(0)
        progress.show()

        worker = NoiseFilterWorker(self._db_path, dataset, method,
                                   parent=self)
        self._noise_filter_worker = worker   # keep alive

        def on_stage(text: str) -> None:
            progress.setLabelText(text)

        def on_ok(ds: str, m: str, n_vectors: int, n_noise: int) -> None:
            progress.close()
            share = n_noise / max(1, n_vectors) * 100.0
            self._status(f"{m.upper()} noise filter ready for {ds}: "
                         f"{n_noise:,} of {n_vectors:,} chunks "
                         f"({share:.1f} %) flagged as noise.")
            self._details.refresh_datasets()
            self._details.refresh_chunks()   # highlight the new outliers
            self._rerun_similar_search()

        def on_failed(message: str) -> None:
            progress.close()
            self._details.set_noise_check(method, False)
            self._status(message)

        worker.stage.connect(on_stage)
        worker.progress.connect(progress.setValue)
        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_failed)
        worker.start()

    def _on_similar_results(self, results: list) -> None:
        self._details.show_similar_results(results)
        if results:
            seed = "seed + " if len(results) > 1 else ""
            self._status(f"{len(results)} rows ({seed}best matches first, "
                         f"dataset={results[-1].method}).")

    def _on_similar_failed(self, message: str) -> None:
        self._details.show_similar_error(message)
        self._status(message)

    def open_similar_for(self, track_id: int) -> None:
        """Context-menu entry point: focus the Similar tab and search."""
        self._details.setCurrentIndex(2)
        self._details.show_track(track_id)
        self._details._on_search()

    def _on_create_playlist(self, pairs: list, method: str) -> None:
        if not pairs:
            return
        name, ok = QInputDialog.getText(self, "New playlist",
                                        "Playlist name:", text="Similar mix")
        if not ok or not name.strip():
            return
        playlist_id = self._create_playlist_from_pairs(pairs, method,
                                                       name.strip())
        self._details.refresh_playlists(select_id=playlist_id)
        self._status(f"Playlist '{name.strip()}' created with "
                     f"{len(pairs)} track(s).")

    def _create_playlist_from_pairs(self, pairs: list, method: str,
                                    name: str) -> int:
        """Persist a playlist from ``(track_id, score)`` pairs; returns the id."""
        conn = self._db.connect()
        try:
            playlist_id = repo.create_playlist(
                conn, name,
                seed_track_id=self._similar_seed,
                method=method if method != "auto" and pairs else method)
            conn.commit()
            repo.add_playlist_items(conn, playlist_id, pairs)
            conn.commit()
        finally:
            conn.close()
        return int(playlist_id)

    def _on_create_playlist_and_play(self, pairs: list, method: str) -> None:
        """One-click similar mix: create playlist, export .m3u, play it now.

        No dialogs — the playlist is auto-named with a timestamp, written to
        ``data/playlists/<name>.m3u`` and handed to the OS default music
        player, so "listen immediately" is a single click.
        """
        if not pairs:
            return
        from datetime import datetime

        name = f"Similar mix {datetime.now():%Y-%m-%d %H.%M}"
        playlist_id = self._create_playlist_from_pairs(pairs, method, name)
        self._details.refresh_playlists(select_id=playlist_id)
        conn = self._db.connect()
        try:
            path = export_m3u(conn, playlist_id, default_m3u_path(name))
        finally:
            conn.close()
        if open_in_system_player(path):
            self._status(f"Playing playlist '{name}' ({len(pairs)} track(s)) "
                         f"in the system player — {path}")
        else:
            self._status(f"Playlist exported to {path} — no default player "
                         "is registered for .m3u files.")

    def _on_play_track(self, track_id: int) -> None:
        """Open a track's audio file in the system default music player."""
        conn = self._db.connect()
        try:
            row = repo.get_track(conn, int(track_id))
        finally:
            conn.close()
        if row is None:
            return
        if open_in_system_player(row["path"]):
            self._status(f"Playing {row['filename']} in the system player.")
        else:
            self._status(f"Could not play {row['filename']} — no default "
                         "player is registered for this file type.")

    def export_playlist(self, playlist_id: int) -> None:
        out, _ = QFileDialog.getSaveFileName(self, "Export playlist",
                                             "playlist.m3u", "M3U playlist (*.m3u)")
        if not out:
            return
        conn = self._db.connect()
        try:
            path = export_m3u(conn, playlist_id, out)
        finally:
            conn.close()
        self._status(f"Exported playlist to {path}")

    # -------------------------------------------------------------- settings ---
    def open_visualisation(self) -> None:
        """Open the (non-modal) Visualisation scatter-plot window."""
        from app.ui.visualisation import VisualisationDialog

        if self._viz_dialog is None:
            self._viz_dialog = VisualisationDialog(self._db, self._config,
                                                   self)
        self._viz_dialog.refresh_and_plot()
        self._viz_dialog.show()
        self._viz_dialog.raise_()
        self._viz_dialog.activateWindow()

    def open_settings(self) -> None:
        from app.ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self._config, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        dialog.apply()
        self._config.save()
        self._status("Settings saved.")
        self._start_ollama_detection()

    # ---------------------------------------------------------------- ollama ---
    def _start_ollama_detection(self) -> None:
        if not self._config.use_ollama:
            self._status("Ollama disabled in settings.")
            return
        self._ollama_worker = OllamaDetectWorker(self._config.ollama_host)
        self._ollama_worker.detected.connect(self._on_ollama_detected)
        self._ollama_worker.failed.connect(
            lambda msg: self._status(f"Ollama detection failed: {msg}"))
        self._ollama_worker.start()

    def _on_ollama_detected(self, running: bool, models: list) -> None:
        if not running:
            self._status("Ollama not reachable — text-embedding similarity "
                         "unavailable (audio-model similarity still works).")
            return
        models = rank_embedding_models(models)
        if models:
            with self._db.transaction() as conn:
                repo.save_ollama_models(conn, models)
            if not self._config.ollama_embedding_model:
                self._config.ollama_embedding_model = models[0]
            self._status(f"Ollama running — {len(models)} embedding model(s) "
                         f"detected (default: {models[0]}).")
        else:
            self._status("Ollama running but no embedding models installed — "
                         "try `ollama pull nomic-embed-text`.")

    # -------------------------------------------------------------- selection ---
    def _on_track_selected(self, track_id: int) -> None:
        self._details.show_track(track_id)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        for worker in (self._scan_worker, self._analysis_worker,
                       self._similar_worker, self._ollama_worker):
            if worker is not None and worker.isRunning():
                worker.wait(3000)
        super().closeEvent(event)
