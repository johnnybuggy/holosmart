"""Right pane: track details — Overview, Chunks (grouped per track), Similar, Playlists."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget, QCheckBox,
)

from app.audio.chunking import format_duration, format_size
from app.db import repo
from app.db.database import Database
from app.models.registry import plugin_info

import logging

log = logging.getLogger(__name__)

_META_ROWS = [
    ("filename", "Filename"), ("path", "Path"), ("size_bytes", "Size"),
    ("duration_sec", "Duration"), ("container", "Container"), ("codec", "Codec"),
    ("sample_rate", "Sample rate"), ("channels", "Channels"),
    ("bit_depth", "Bit depth"), ("bitrate_kbps", "Bitrate"),
    ("title", "Title"), ("artist", "Artist"), ("album", "Album"),
    ("genre", "Genre"), ("year", "Year"), ("track_no", "Track #"),
    ("status", "Status"), ("status_message", "Analysis note"),
]


def _human(key: str, value) -> str:
    if value is None:
        return "?"
    if key == "size_bytes":
        return format_size(int(value))
    if key == "duration_sec":
        return f"{format_duration(float(value))} ({float(value):.1f} s)"
    if key == "sample_rate":
        return f"{int(value):,} Hz"
    if key == "bitrate_kbps":
        return f"{float(value):.0f} kbps"
    return str(value)


def _truncate(text: str, limit: int = 120) -> str:
    """One-line, length-capped text for quiet inline display."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


#: The noise-filter methods, their tree label and their dedicated highlight
#: color (used both for the checkbox legend and the chunk-row backgrounds).
_NOISE_METHOD_COLORS: tuple[tuple[str, str, str], ...] = (
    ("hdbscan", "HDBSCAN", "#ffd54f"),      # amber
    ("optics", "OPTICS", "#90caf9"),        # light blue
)
#: Chunks flagged by BOTH detectors get their own blend color.
_NOISE_BOTH_COLOR = "#f48fb1"               # pink

#: Outlier highlight styles applied to a flagged chunk's whole row.
_NOISE_ROW_STYLE = {
    "hdbscan": "#ffd54f",
    "optics": "#90caf9",
    "both": "#f48fb1",
}


class DetailPane(QTabWidget):
    """Everything known about the selected track + similarity + playlists."""

    search_requested = Signal(object, str, str, int, object)
    # seed, dataset, algorithm, limit, discard-noise method names
    noise_filter_toggled = Signal(str, str, bool)
    #: "Add folder" pressed in the Similar tab — the main window resolves
    #: the selected tree folder and replies via :meth:`add_reference_tracks`.
    folder_references_requested = Signal()   # dataset, method, on
    playlist_requested = Signal(list, str)     # [(track_id, score)...], method
    playlist_changed = Signal()                # a playlist was created/deleted
    #: A chunk's index cell was clicked in the Chunks tab: play exactly
    #: that chunk in the built-in player (path, start_sec, end_sec).
    chunk_play_requested = Signal(str, float, float)
    #: "Find outliers" pressed in the Chunks tab: cluster the chunks of
    #: tracks not yet clustered (newly/re-analyzed) for every covered
    #: model, both methods.  Nothing runs automatically any more.
    noise_sweep_requested = Signal()
    # Double-clicked a similar-results row (or a tree track): play the file
    # in the system player.
    play_track_requested = Signal(int)
    # "Create .m3u & play" clicked: build the playlist from the current
    # results, export it and hand it to the system player immediately.
    playlist_play_requested = Signal(list, str)

    def __init__(self, db: Database, config, parent=None) -> None:
        super().__init__(parent)
        self._db = db
        self._config = config
        self._current_track_id: int | None = None
        # Track id whose reference list the user deliberately cleared: the
        # auto-fill stays suppressed for it until the selection changes.
        self._cleared_track_id: int | None = None
        self._similar_cache: list[tuple[int, float]] = []

        self.addTab(self._build_overview_tab(), "Overview")
        self.addTab(self._build_chunks_tab(), "Chunks")
        self.addTab(self._build_similar_tab(), "Similar")
        self.addTab(self._build_learning_tab(), "Learning")
        self.addTab(self._build_playlists_tab(), "Playlists")
        self.refresh_playlists()

    # ------------------------------------------------------------ overview --
    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self._warning = QLabel("")
        # Quiet presentation: muted small text instead of a bold red block,
        # height-capped so diagnostics can never dominate the layout — the
        # full text lives in the banner's tooltip (see _set_warning_banner).
        self._warning.setStyleSheet("color: #8a6d3b; font-size: 11px;")
        self._warning.setWordWrap(True)
        self._warning.setMaximumHeight(48)  # ~3 wrapped lines at most
        self._warning.setVisible(False)
        layout.addWidget(self._warning)
        self._meta_table = QTableWidget(len(_META_ROWS), 2)
        self._meta_table.setHorizontalHeaderLabels(["Field", "Value"])
        self._meta_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._meta_table.verticalHeader().setVisible(False)
        self._meta_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, (key, label) in enumerate(_META_ROWS):
            self._meta_table.setItem(row, 0, QTableWidgetItem(label))
            self._meta_table.setItem(row, 1, QTableWidgetItem("—"))
        layout.addWidget(self._meta_table)
        group = QGroupBox("Model description (aggregated from chunk tags)")
        group_layout = QVBoxLayout(group)
        self._description = QTextEdit()
        self._description.setReadOnly(True)
        self._description.setMaximumHeight(90)
        self._description.setPlaceholderText("Analyze the track to get model tags.")
        group_layout.addWidget(self._description)
        layout.addWidget(group)
        return page

    # -------------------------------------------------------------- chunks --
    def _build_chunks_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        hint_row = QHBoxLayout()
        self._chunks_hint = QLabel("Select a track in the tree to see its chunks.")
        # Word-wrap + keeping the text short (see _refresh_chunks) stop this
        # label from stretching the results pane: a long single-line sizeHint
        # was what forced the whole window too wide.
        self._chunks_hint.setWordWrap(True)
        hint_row.addWidget(self._chunks_hint, 1)

        # Outlier highlighting in this table is UNCONDITIONAL: whichever
        # noise runs exist (per model + method), this table tints the
        # flagged chunks — no checkbox gates that.  Clustering is
        # BUTTON-driven (below) and incremental — only tracks not yet
        # clustered are judged.  The HDBSCAN/OPTICS checkboxes live on
        # the Similar tab and decide whether noise chunks are DISCARDED
        # from searches.
        self._noise_sweep_button = QPushButton("Find outliers")
        self._noise_sweep_button.setToolTip(
            "Cluster the chunks of newly analyzed tracks (and "
            "re-analyzed tracks whose chunks changed) with HDBSCAN + "
            "OPTICS for every model.  Tracks already clustered are "
            "skipped — nothing runs automatically after analysis.")
        self._noise_sweep_button.clicked.connect(
            self.noise_sweep_requested.emit)
        hint_row.addWidget(self._noise_sweep_button, 0)
        hint_row.addStretch(1)
        layout.addLayout(hint_row)
        self._noise_checks: dict[str, QCheckBox] = {}
        self._chunks_table = QTableWidget()
        self._chunks_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._chunks_table.setAlternatingRowColors(True)
        self._chunks_table.verticalHeader().setVisible(False)
        self._chunks_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        # Double-click a model cell -> full-vector dialog. The column->model
        # map and row->chunk list are (re)built by _refresh_chunks.
        self._chunks_model_columns: dict[int, str] = {}
        self._chunks_chunk_ids: list[int] = []
        self._chunks_table.cellClicked.connect(
            self._on_chunk_cell_clicked)
        self._chunks_table.cellDoubleClicked.connect(
            self._on_chunk_cell_double_clicked)
        layout.addWidget(self._chunks_table)
        return page

    def _refresh_chunks(self, conn, track_id: int, status_notes: list[str] | None = None,
                        config_models: list[str] | None = None) -> None:
        chunks = repo.get_chunks(conn, track_id)
        track = repo.get_track(conn, track_id)
        display = {info["name"]: info["display_name"] for info in plugin_info()}
        # the hint stays short: per-model diagnostics must never be
        # concatenated here (they used to stretch the results pane)
        hint = (f"{track['filename']}: {len(chunks)} chunks"
                + (f" (track analyzed: {track['status']})" if track else ""))
        self._chunks_hint.setText(hint)
        # discover which models have embeddings on these chunks
        models: list[str] = []
        for chunk in chunks:
            for row in repo.get_chunk_embeddings(conn, chunk["id"]):
                if row["model"] not in models:
                    models.append(row["model"])
        # configured models that produced nothing on an analyzed track get
        # their own column of compact "⚠" cells; the full reason stays
        # reachable on the column header's tooltip (hover, not shout)
        missing: dict[str, str] = {}
        if track["status"] == "analyzed" and config_models:
            for name in config_models:
                if name in models:
                    continue
                disp = display.get(name, name)
                missing[name] = next(
                    (n for n in (status_notes or []) if n.startswith(disp)), "")
        extra = list(missing)
        columns = ["Chunk", "Start", "End", "Tags"] + [
            display.get(m, m) for m in models] + [
            display.get(m, m) for m in extra]
        self._chunks_table.setColumnCount(len(columns))
        self._chunks_table.setHorizontalHeaderLabels(columns)
        # Double-click support: remember which column shows which model and
        # which chunk each row is (see _on_chunk_cell_double_clicked).
        self._chunks_model_columns = {4 + i: m for i, m in enumerate(models)}
        self._chunks_chunk_ids = [int(c["id"]) for c in chunks]
        # Chunk playback: the file and the per-row [start, end] seconds.
        self._chunks_track_path = str(track["path"]) if track else None
        self._chunks_ranges = [(float(chunk["start_sec"]),
                                float(chunk["end_sec"]))
                               for chunk in chunks]
        for offset, name in enumerate(extra):
            tip = f"⚠ {display.get(name, name)}: no results"
            if missing[name]:
                tip += f" — {missing[name]}"
            self._chunks_table.horizontalHeaderItem(
                4 + len(models) + offset).setToolTip(tip)
        # Outlier highlighting is UNCONDITIONAL (every analysis run
        # re-clusters every covered model — HDBSCAN + OPTICS — so flags
        # are always current): flagged chunks get their row tinted with
        # the method's color (both -> blend color).  The Similar-tab
        # checkboxes only decide whether noise is DISCARDED from
        # searches.
        flagged: dict[str, set[int]] = {}
        try:
            from app.similarity.noise_filter import noise_ids_for

            with self._db.transaction() as nconn:
                for method, _label, _color in _NOISE_METHOD_COLORS:
                    # "auto" pools EVERY stored run of the method: the
                    # grid shows outliers regardless of which dataset the
                    # Similar tab has selected.
                    flagged[method] = noise_ids_for(nconn, "auto", (method,))
        except ImportError:
            flagged = {}
        row_color: dict[int, str] = {}
        for method, ids in flagged.items():
            for chunk_id in ids:
                prior = row_color.get(chunk_id)
                row_color[chunk_id] = (
                    _NOISE_ROW_STYLE["both"] if prior else
                    _NOISE_ROW_STYLE[method])
        labels = {method: label for method, label, _color
                  in _NOISE_METHOD_COLORS if method in flagged}

        self._chunks_table.setRowCount(len(chunks))
        for r, chunk in enumerate(chunks):
            tags = repo.get_chunk_tags(conn, chunk["id"])
            by_model: dict[str, list[str]] = {}
            import math

            for tag in tags:
                score = float(tag["score"])
                # Weights live in [0, 1]; the interesting dynamic range is
                # in the tail — show log10 so 0.001 doesn't round to 0.00.
                shown = ("— (score 0)" if score <= 0.0
                         else f"{math.log10(score):.2f}")
                by_model.setdefault(tag["model"], []).append(
                    f"{tag['text']} {shown}")
            tag_text = "; ".join(f"{m}: {', '.join(items)}"
                                 for m, items in by_model.items()) or "—"
            values = [str(chunk["idx"]), f"{chunk['start_sec']:.2f} s",
                      f"{chunk['end_sec']:.2f} s", tag_text]
            for model in models:
                rows = [e for e in repo.get_chunk_embeddings(conn, chunk["id"])
                        if e["model"] == model]
                if rows:
                    vec = rows[0]["vec"]
                    head = ", ".join(f"{float(v):+.3f}" for v in vec[:4])
                    values.append(f"dim={rows[0]['dim']} [{head}, …]")
                else:
                    values.append("—")
            # quiet "no results" marker for configured-but-absent models
            for _ in extra:
                values.append("⚠")
            color = row_color.get(int(chunk["id"]))
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c < 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color is not None:
                    item.setBackground(QColor(color))
                if c in self._chunks_model_columns:
                    item.setToolTip(
                        "Double-click for the full vector values and "
                        "dimension names.")
                self._chunks_table.setItem(r, c, item)
            play_tip = (f"Click to play this chunk "
                        f"({chunk['start_sec']:.2f} – "
                        f"{chunk['end_sec']:.2f} s) in the built-in player.")
            self._chunks_table.item(r, 0).setToolTip(play_tip)
            if color is not None:
                methods_hit = [labels[m] for m in flagged
                               if int(chunk["id"]) in flagged[m]]
                self._chunks_table.item(r, 0).setToolTip(
                    "Noise outlier: " + " + ".join(methods_hit)
                    + "\n" + play_tip)
        self._chunks_table.resizeColumnsToContents()
        self._chunks_table.setColumnWidth(3, 360)

    # -------------------------------------------------------------- similar --
    def _build_similar_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Dataset:"))
        self._dataset_combo = QComboBox()
        self._dataset_combo.setToolTip(
            "Which analysis dataset to compare in: a model's chunk/centroid "
            "vectors, a stored dimensionality reduction, or Ollama track "
            "descriptions.")
        self._dataset_combo.currentIndexChanged.connect(
            lambda _index: self._sync_algorithm_and_noise())
        controls.addWidget(self._dataset_combo)
        self._reduce_button = QPushButton("Reduce…")
        self._reduce_button.setToolTip(
            "Create a new searchable dataset by dimensionality-reducing a "
            "model's chunk vectors (PCA / UMAP / t-SNE).")
        self._reduce_button.clicked.connect(self._on_open_reduction_dialog)
        controls.addWidget(self._reduce_button)
        self._refresh_reduce_button = QPushButton("Refresh")
        self._refresh_reduce_button.setToolTip(
            "Re-run this reduction over the CURRENT chunk vectors in place "
            "(same method and parameters; the dataset keeps its id). Use "
            "it when the projection became outdated by later analysis.")
        self._refresh_reduce_button.clicked.connect(
            self._on_refresh_reduction_dialog)
        controls.addWidget(self._refresh_reduce_button)
        controls.addSpacing(10)
        controls.addWidget(QLabel("Algorithm:"))
        self._algorithm_combo = QComboBox()
        self._algorithm_combo.addItem(
            "Centroid", "centroid")
        self._algorithm_combo.addItem(
            "PSVI (Pareto surface volume intersection)", "pareto")
        self._algorithm_combo.addItem(
            "EMD (Earth Mover's Distance)", "emd")
        self._algorithm_combo.addItem("Chamfer (chunk distance)", "chamfer")
        self._algorithm_combo.setToolTip(
            "How two tracks are compared: mean-centroid cosine, best-match "
            "comparison of the seed's Pareto-surface chunks (PSVI), optimal "
            "transport between chunk sets (EMD) or symmetric nearest-chunk "
            "distance (Chamfer).")
        controls.addWidget(self._algorithm_combo)
        controls.addWidget(QLabel("Max results:"))
        self._limit_spin = QSpinBox()
        self._limit_spin.setRange(5, 100)
        self._limit_spin.setValue(int(self._config.playlist_length))
        controls.addWidget(self._limit_spin)
        self._search_button = QPushButton("Search")
        self._search_button.clicked.connect(self._on_search)
        controls.addWidget(self._search_button)
        controls.addSpacing(10)
        # Noise-filter toggles (moved here from the Chunks tab): they
        # decide whether chunks flagged as noise outliers are DISCARDED
        # from searches.  The Chunks tab always TINTS those chunks in
        # this detector's color, checkbox or not.
        controls.addWidget(QLabel("Discard noise:"))
        self._noise_checks: dict[str, QCheckBox] = {}
        for method, label, _color in _NOISE_METHOD_COLORS:
            checkbox = QCheckBox(label)
            checkbox.setToolTip(
                f"Discard chunks that the one-time {label} clustering "
                "flags as noise outliers before comparing (junk chunks — "
                "silence, fades, transitions — skew chunk-level results). "
                "The Chunks tab highlights those chunks regardless of "
                "this checkbox.")
            checkbox.toggled.connect(
                lambda on, m=method: self._on_noise_toggled(m, on))
            self._noise_checks[method] = checkbox
            controls.addWidget(checkbox)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Multi-reference support: the list is auto-filled with the file
        # selected in the tree; "Add selected" appends further references.
        # Results combine the per-reference match percentages via their
        # geometric mean (see similar_tracks_multi).
        refs_box = QGroupBox("Reference tracks")
        refs_layout = QHBoxLayout(refs_box)
        self._seed_list = QListWidget()
        self._seed_list.setToolTip(
            "Tracks the search compares against. Auto-filled with the file "
            "selected in the tree; add more with 'Add selected'. Results "
            "rank by the geometric mean of the per-reference match "
            "percentages, so a track must be similar to ALL references.")
        self._seed_list.setMaximumHeight(64)
        self._seed_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        refs_layout.addWidget(self._seed_list, 1)
        # The four buttons live in a compact 2x2 grid: a tall button column
        # would dictate the box height and eat the results table's space.
        seed_buttons = QGridLayout()
        self._seed_add_button = QPushButton("Add selected")
        self._seed_add_button.setToolTip(
            "Append the file currently selected in the tree as another "
            "reference for the similarity search.")
        self._seed_add_button.clicked.connect(self._on_add_seed)
        seed_buttons.addWidget(self._seed_add_button, 0, 0)
        self._seed_remove_button = QPushButton("Remove")
        self._seed_remove_button.setToolTip(
            "Remove the highlighted reference from the list.")
        self._seed_remove_button.clicked.connect(self._on_remove_seed)
        seed_buttons.addWidget(self._seed_remove_button, 0, 1)
        self._seed_clear_button = QPushButton("Clear")
        self._seed_clear_button.setToolTip(
            "Empty the reference list. The currently selected file is NOT "
            "re-added automatically; picking another file in the tree "
            "resumes the auto-fill.")
        self._seed_clear_button.clicked.connect(self._on_clear_seeds)
        seed_buttons.addWidget(self._seed_clear_button, 1, 0)
        self._seed_folder_button = QPushButton("Add folder…")
        self._seed_folder_button.setToolTip(
            "Add every track of the folder selected in the tree (or of the "
            "selected file's folder) as references. References without "
            "vectors in the search dataset are skipped during the search.")
        self._seed_folder_button.clicked.connect(
            self.folder_references_requested.emit)
        seed_buttons.addWidget(self._seed_folder_button, 1, 1)
        seed_buttons.setContentsMargins(0, 0, 0, 0)
        refs_layout.addLayout(seed_buttons)
        # Maximum vertical policy: the box shrinks to its content and never
        # grows into the results table's space, whatever the window size.
        refs_box.setSizePolicy(QSizePolicy.Policy.Maximum,
                               QSizePolicy.Policy.Maximum)
        layout.addWidget(refs_box)
        self._refs_box = refs_box
        self._auto_seed_id: int | None = None

        self._populate_dataset_combo()

        self._similar_status = QLabel("")
        self._similar_status.setWordWrap(True)
        self._similar_status.setVisible(False)
        layout.addWidget(self._similar_status)

        self._similar_table = QTableWidget(0, 4)
        self._similar_table.setHorizontalHeaderLabels(
            ["Filename", "Artist", "Title", "Score"])
        self._similar_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._similar_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._similar_table.setAlternatingRowColors(True)
        self._similar_table.verticalHeader().setVisible(False)
        self._similar_table.setToolTip(
            "Double-click a row to play the file in the system player.")
        self._similar_table.cellDoubleClicked.connect(
            self._on_similar_double_clicked)
        layout.addWidget(self._similar_table)

        actions = QHBoxLayout()
        self._playlist_button = QPushButton("Create playlist from results")
        self._playlist_button.setEnabled(False)
        self._playlist_button.clicked.connect(self._on_create_playlist)
        actions.addWidget(self._playlist_button)
        # One-click export + immediate playback: the playlist is created,
        # written as .m3u into data/playlists/ and handed to the OS default
        # music player without any further dialogs.
        self._playlist_play_button = QPushButton("Create .m3u & play")
        self._playlist_play_button.setEnabled(False)
        self._playlist_play_button.setToolTip(
            "Create a playlist from these results, export it as .m3u to "
            "data/playlists/ and open it in the system music player.")
        self._playlist_play_button.clicked.connect(
            self._on_create_playlist_and_play)
        actions.addWidget(self._playlist_play_button)
        self._copy_button = QPushButton("Copy files…")
        self._copy_button.setEnabled(False)
        self._copy_button.setToolTip(
            "Copy the files of these search results into a folder you "
            "pick — identical copies already there are skipped, colliding "
            "differing files get 'name (2)' style names.")
        self._copy_button.clicked.connect(self._on_copy_results)
        actions.addWidget(self._copy_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return page

    def refresh_datasets(self) -> None:
        """Public hook: re-read the database into the Dataset picker."""
        self._populate_dataset_combo()
        self._sync_noise_checkboxes()

    # ---- noise-filter state -------------------------------------------------
    def _current_dataset(self) -> str:
        return self._dataset_combo.currentData() or "auto"

    def _noise_methods(self) -> tuple[str, ...]:
        """Enabled + checked noise-filter methods, in a stable order."""
        return tuple(method for method in ("hdbscan", "optics")
                     if self._noise_checks[method].isEnabled()
                     and self._noise_checks[method].isChecked())

    def _on_chunk_cell_clicked(self, row: int, column: int) -> None:
        """Click the chunk's number (column 0) → play exactly that chunk."""
        if column != 0 or not self._chunks_track_path:
            return
        if 0 <= row < len(self._chunks_ranges):
            start, end = self._chunks_ranges[row]
            self.chunk_play_requested.emit(self._chunks_track_path,
                                           start, end)

    def _on_chunk_cell_double_clicked(self, row: int, column: int) -> None:
        """Double-click: model cell -> the chunk's full vector; any other
        cell -> every tag the chunk has."""
        if not 0 <= row < len(self._chunks_chunk_ids):
            return
        chunk_id = self._chunks_chunk_ids[row]
        model = self._chunks_model_columns.get(column)
        if model is not None:
            self._show_chunk_vector_dialog(chunk_id, model)
        else:
            self._show_chunk_tags_dialog(chunk_id)

    def _show_chunk_vector_dialog(self, chunk_id: int, model: str) -> None:
        """Modal read-out of every dimension of one chunk embedding."""
        import numpy as np
        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QHeaderView, QTableWidget,
            QVBoxLayout,
        )

        from app.models import fft_model

        conn = self._db.connect()
        try:
            rows = repo.get_chunk_embeddings(conn, chunk_id)
        finally:
            conn.close()
        entry = next((r for r in rows if str(r["model"]) == model), None)
        if entry is None:
            return
        vec = np.asarray(entry["vec"], dtype=np.float64).reshape(-1)
        dim = int(vec.size)
        names = (fft_model.feature_names() if model == "fft"
                 else tuple(f"dim {i}" for i in range(dim)))

        dialog = QDialog(self)
        dialog.setWindowTitle(
            f"{self._plugin_display_name(model)} — chunk #{chunk_id} "
            f"({dim} dims)")
        dialog.setMinimumSize(420, 480)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(dim, 2)
        table.setHorizontalHeaderLabels(["Dimension", "Value"])
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for i in range(dim):
            name_item = QTableWidgetItem(names[i])
            name_item.setToolTip(names[i])
            table.setItem(i, 0, name_item)
            table.setItem(i, 1, QTableWidgetItem(f"{float(vec[i]):.6g}"))
        layout.addWidget(table)
        note = QLabel(
            "Values as stored for this chunk (L2-normalized per chunk).")
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.clicked.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _show_chunk_tags_dialog(self, chunk_id: int) -> None:
        """Modal read-out of ALL tags stored for one chunk (any model)."""
        import math

        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QHeaderView, QTableWidget,
            QVBoxLayout,
        )

        conn = self._db.connect()
        try:
            tags = repo.get_chunk_tags(conn, chunk_id)
        finally:
            conn.close()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Chunk #{chunk_id} — all tags "
                              f"({len(tags)})")
        dialog.setMinimumSize(460, 420)
        layout = QVBoxLayout(dialog)
        if tags:
            table = QTableWidget(len(tags), 3)
            table.setHorizontalHeaderLabels(
                ["Model", "Tag", "Score (log10)"])
            table.horizontalHeader().setSectionResizeMode(
                1, QHeaderView.ResizeMode.Stretch)
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for r, tag in enumerate(tags):
                score = float(tag["score"])
                # Weights live in [0, 1]; the interesting dynamic range is
                # in the tail — show log10(p) so 0.001 doesn't round to 0.00
                log10 = math.log10(score) if score > 0.0 else -99.0
                model_item = QTableWidgetItem(
                    self._plugin_display_name(str(tag["model"])))
                tag_item = QTableWidgetItem(str(tag["text"]))
                score_item = QTableWidgetItem(
                    "— (score 0)" if score <= 0.0 else f"{log10:.2f}")
                score_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                for c, item in enumerate((model_item, tag_item, score_item)):
                    table.setItem(r, c, item)
            layout.addWidget(table)
            note = QLabel(
                "Scores as log10 of the stored weight (closer to 0 = "
                "more confident; -99 = score 0). Double-click a model's "
                "column in the Chunks table for that model's vectors.")
            note.setWordWrap(True)
            note.setStyleSheet("color: gray; font-size: 11px;")
            layout.addWidget(note)
        else:
            empty = QLabel("This chunk has no tags.")
            empty.setStyleSheet("color: gray;")
            layout.addWidget(empty)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.clicked.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _plugin_display_name(self, model: str) -> str:
        for info in plugin_info():
            if info["name"] == model:
                return str(info["display_name"])
        return model

    def _resolve_noise_run_dataset(self) -> str:
        """The dataset a freshly-checked filter should cluster first.

        Preference: the Similar tab's selected dataset when it is a model
        or reduction; else the first model that has chunk vectors for the
        currently displayed track; else any model with chunk vectors.
        """
        current = (self._current_dataset()
                   if getattr(self, "_dataset_combo", None) else "")
        if current and not current.startswith(("auto", "ollama:")):
            return current
        if self._current_track_id is not None:
            conn = self._db.connect()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT e.model AS model FROM embeddings e "
                    "JOIN chunks c ON c.id = e.chunk_id "
                    "WHERE c.track_id = ? ORDER BY e.model",
                    (self._current_track_id,)).fetchall()
            finally:
                conn.close()
            models = [str(r["model"]) for r in rows]
            if models:
                return models[0]
        conn = self._db.connect()
        try:
            row = conn.execute(
                "SELECT model FROM embeddings GROUP BY model "
                "ORDER BY model LIMIT 1").fetchone()
        finally:
            conn.close()
        return str(row["model"]) if row is not None else "fft"

    def _on_noise_toggled(self, method: str, on: bool) -> None:
        self.noise_filter_toggled.emit(self._resolve_noise_run_dataset(),
                                       method, on)
        # the checkbox doubles as the highlight switch for the Chunks table
        self.refresh_chunks()

    def _sync_noise_checkboxes(self) -> None:
        """Update checkbox enablement + tooltips from the stored filters.

        The checkboxes are METHOD toggles, not dataset-bound: checked =
        this detector's outliers are active (Chunks-tab highlighting and
        Similar-search filtering).  They are never auto-unchecked — a
        dataset without a cached run simply contributes no noise ids
        (searches fall back to the stored runs of other datasets where
        available; the pooled tooltip reports what exists).
        """
        if not getattr(self, "_noise_checks", None):
            return   # the tabs are still being built
        from app.similarity.noise_filter import (
            availability, filters_for_dataset)

        missing_lib = availability()
        with self._db.transaction() as conn:
            pooled = filters_for_dataset(conn, "auto")
        for method, checkbox in self._noise_checks.items():
            if missing_lib.get(method):
                checkbox.setEnabled(False)
                checkbox.setToolTip(missing_lib[method])
                continue
            checkbox.setEnabled(True)
            info = pooled.get(method)
            if info is None:
                checkbox.setToolTip(
                    f"No cached {method.upper()} noise data anywhere yet — "
                    "checking it starts a one-time per-song clustering run "
                    "in the background (fast, cached for every later "
                    "search and for the Chunks-tab highlighting).")
                continue
            share = info["n_noise"] / max(1, info["n_vectors"]) * 100.0
            checkbox.setToolTip(
                f"{method.upper()} noise data ready: "
                f"{info['n_noise']:,} of {info['n_vectors']:,} chunks "
                f"({share:.1f} %) flagged as noise "
                f"({info['dataset']}, fitted {info['created_at']}).")
        # Checked-state changes must be visible in the Chunks table at once.
        self.refresh_chunks()

    def refresh_chunks(self) -> None:
        """Re-render the current track's Chunks table (highlight updates)."""
        if self._current_track_id is None:
            return
        with self._db.transaction() as conn:
            self._refresh_chunks(conn, self._current_track_id)

    def set_noise_check(self, method: str, checked: bool) -> None:
        """Public hook for the main window (failed runs uncheck the box)."""
        checkbox = self._noise_checks.get(method)
        if checkbox is not None and checkbox.isChecked() != bool(checked):
            checkbox.setChecked(bool(checked))

    def _populate_dataset_combo(self, select: str | None = None) -> None:
        """Fill the Dataset picker from what the database can compare.

        Entries: every registered model that has chunk vectors (disabled
        with a hint when fewer than two tracks carry it), then the stored
        dimensionality reductions, then Ollama description datasets.
        """
        combo = self._dataset_combo
        # preserve the user's selection across repopulation (e.g. after an
        # analysis run adds datasets): keep the current dataset when it
        # still exists.
        previous = combo.currentData() if combo.currentIndex() >= 0 else None
        combo.blockSignals(True)
        combo.clear()
        with self._db.transaction() as conn:
            chunk_models = set(repo.get_chunk_vector_models(conn))
            track_rows = conn.execute(
                "SELECT model, COUNT(DISTINCT track_id) AS n "
                "FROM track_embeddings GROUP BY model").fetchall()
            track_counts = {str(r["model"]): int(r["n"]) for r in track_rows}
            reductions = repo.list_reductions(conn)
            coverage = {int(r["id"]): repo.reduction_coverage(
                conn, int(r["id"]), str(r["source_model"]))
                for r in reductions}
        for info in plugin_info():
            combo.addItem(info["display_name"], info["name"])
            item = combo.model().item(combo.count() - 1)
            name = info["name"]
            if name not in chunk_models:
                item.setEnabled(False)
                item.setToolTip("No chunk vectors yet — analyze tracks with "
                                "this model first.")
            else:
                # Chunk-level algorithms (PSVI/EMD/Chamfer) need just the
                # chunk vectors; centroid search additionally wants >= 2
                # analyzed tracks (the search explains when it can't).
                item.setEnabled(True)
                n = track_counts.get(name, 0)
                item.setToolTip(
                    f"{n} track(s) ready for centroid search"
                    + ("" if n >= 2 else " — analyze at least 2 for it"))
        for red in reductions:
            red_id = int(red["id"])
            covered, total = coverage.get(red_id, (0, 0))
            suffix = "  ⚠ outdated" if covered < total else ""
            combo.addItem(f"{red['name']}{suffix}", f"red:{red_id}")
            if covered < total:
                item = combo.model().item(combo.count() - 1)
                item.setToolTip(
                    f"Outdated: the stored projection covers {covered:,} of "
                    f"{total:,} current chunk vectors — new/changed chunks "
                    "are missing. Press Refresh to re-run the same "
                    "reduction over the current data.")
        for model in sorted(track_counts):
            if not model.startswith("ollama:"):
                continue
            combo.addItem(f"Description ({model.split(':', 1)[1]})", model)
        if select is None and previous is not None:
            select = previous
        if select is not None:
            index = combo.findData(select)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._sync_algorithm_for_dataset()
        self._sync_reduction_staleness()
        self._sync_noise_checkboxes()

    def _sync_algorithm_and_noise(self) -> None:
        self._sync_algorithm_for_dataset()
        self._sync_reduction_staleness()
        self._sync_noise_checkboxes()

    def _sync_algorithm_for_dataset(self) -> None:
        """Ollama description datasets only compare by centroid."""
        dataset = self._dataset_combo.currentData() or ""
        ollama_only = dataset.startswith("ollama:")
        self._algorithm_combo.setEnabled(not ollama_only)
        if ollama_only:
            self._algorithm_combo.setToolTip(
                "Text-description datasets compare by centroid cosine.")
            self._algorithm_combo.setCurrentIndex(0)
        else:
            self._algorithm_combo.setToolTip(
                "How two tracks are compared: mean-centroid cosine, best-match "
                "comparison of the seed's Pareto-surface chunks (PSVI), optimal "
                "transport between chunk sets (EMD) or symmetric nearest-chunk "
                "distance (Chamfer).")

    def _current_search_label(self) -> str:
        """``algorithm:dataset`` label for playlist metadata."""
        dataset = self._dataset_combo.currentData() or "auto"
        algorithm = self._algorithm_combo.currentData() or "centroid"
        if dataset.startswith("ollama:"):
            return dataset
        label = f"{algorithm}:{dataset}"
        if self._noise_methods():
            label += "+noise"
        return label

    def _sync_reduction_staleness(self) -> None:
        """Enable the Refresh button for reduction datasets, with coverage."""
        button = getattr(self, "_refresh_reduce_button", None)
        if button is None:
            return
        dataset = self._current_dataset()
        if not dataset.startswith("red:"):
            button.setEnabled(False)
            button.setToolTip(
                "Select a reduction dataset to refresh it after new "
                "analysis made it outdated.")
            return
        try:
            red_id = int(dataset[4:])
        except ValueError:
            button.setEnabled(False)
            return
        with self._db.transaction() as conn:
            row = repo.get_reduction(conn, red_id)
            if row is None:
                button.setEnabled(False)
                return
            covered, total = repo.reduction_coverage(
                conn, red_id, str(row["source_model"]))
        stale = covered < total
        button.setEnabled(True)
        button.setText("Refresh ⚠" if stale else "Refresh")
        button.setToolTip(
            (f"⚠ Outdated: covers {covered:,} of {total:,} chunk vectors. "
             if stale else
             f"Up to date ({covered:,} of {total:,} chunk vectors). ")
            + "Click to re-run the same reduction over the CURRENT chunk "
            "vectors — the dataset is replaced in place and keeps its id.")

    def _on_refresh_reduction_dialog(self) -> None:
        """Re-run the selected reduction in place over the current vectors."""
        from app.ui.reduction_dialog import ReductionDialog

        dataset = self._current_dataset()
        if not dataset.startswith("red:"):
            return
        red_id = int(dataset[4:])
        with self._db.transaction() as conn:
            row = repo.get_reduction(conn, red_id)
        if row is None:
            return
        dialog = ReductionDialog(self._db, self._config, self, rerun_of=row)
        dialog.reduction_created.connect(
            lambda data_id: self._populate_dataset_combo(select=data_id))
        dialog.exec()

    def _on_open_reduction_dialog(self) -> None:
        """Open the dimensionality-reduction dialog; refresh on success."""
        from app.ui.reduction_dialog import ReductionDialog

        dialog = ReductionDialog(self._db, self._config, self)
        dialog.reduction_created.connect(
            lambda data_id: self._populate_dataset_combo(select=data_id))
        dialog.exec()

    # ------------------------------------------------ multi-reference seeds --
    def _seed_ids(self) -> list[int]:
        """Track ids of the reference list, in list order."""
        return [int(self._seed_list.item(r).data(Qt.ItemDataRole.UserRole))
                for r in range(self._seed_list.count())]

    def _append_seed(self, track_id: int, auto: bool = False) -> bool:
        """Add *track_id* to the reference list (no duplicates).

        Returns True when the list changed.  ``auto`` marks the entry that
        mirrors the tree selection.
        """
        if track_id is None or track_id in self._seed_ids():
            return False
        conn = self._db.connect()
        try:
            row = repo.get_track(conn, track_id)
        finally:
            conn.close()
        if row is None:
            return False
        item = QListWidgetItem(str(row["filename"]))
        item.setData(Qt.ItemDataRole.UserRole, int(track_id))
        item.setToolTip(str(row["path"]))
        if auto:
            self._seed_list.insertItem(0, item)
        else:
            self._seed_list.addItem(item)
        return True

    def _sync_seed_list_with_selection(self) -> None:
        """Auto-fill the reference list from the tree selection.

        An empty list gets the selected file; a list holding only the
        previous auto entry follows the selection.  Manually added
        references are never touched.  A "Clear" press suppresses the
        auto-fill for the currently selected file (the list stays empty);
        selecting a different file resumes it.
        """
        tid = self._current_track_id
        if tid is None:
            return
        if tid == self._cleared_track_id and self._seed_list.count() == 0:
            return                     # cleared by the user: stay empty
        self._cleared_track_id = None
        only_auto = (self._seed_list.count() == 1
                     and self._auto_seed_id is not None
                     and self._seed_ids() == [self._auto_seed_id])
        if self._seed_list.count() == 0 or only_auto:
            self._seed_list.clear()
            self._auto_seed_id = int(tid)
            self._append_seed(int(tid), auto=True)

    def add_reference_tracks(self, track_ids: list[int]) -> int:
        """Append *track_ids* as references (used by "Add folder…").

        Skips duplicates and unknown tracks; returns how many were added.
        """
        self._cleared_track_id = None
        added = 0
        for track_id in track_ids:
            if self._append_seed(int(track_id), auto=False):
                added += 1
        return added

    def _on_add_seed(self) -> None:
        if self._current_track_id is None:
            return
        self._cleared_track_id = None
        self._append_seed(int(self._current_track_id), auto=False)
        # An explicit "Add selected" pins the list: the auto entry stops
        # following the selection (duplicates change nothing but still pin).
        self._auto_seed_id = None

    def _on_remove_seed(self) -> None:
        row = self._seed_list.currentRow()
        if row < 0:
            return
        self._cleared_track_id = None
        removed = int(self._seed_list.item(row).data(Qt.ItemDataRole.UserRole))
        self._seed_list.takeItem(row)
        if removed == self._auto_seed_id:
            self._auto_seed_id = None

    def _on_clear_seeds(self) -> None:
        # Clear must leave the list EMPTY: the auto entry that normally
        # mirrors the tree selection is suppressed until the user picks a
        # different file (or edits the list manually) — otherwise the
        # freshly cleared list would instantly refill with the very file
        # still selected in the tree and Clear would look like a no-op.
        self._seed_list.clear()
        self._auto_seed_id = None
        self._cleared_track_id = self._current_track_id

    def _on_search(self) -> None:
        seeds = self._seed_ids()
        if not seeds and self._current_track_id is not None:
            seeds = [int(self._current_track_id)]
        if not seeds:
            return
        dataset = self._dataset_combo.currentData() or "auto"
        algorithm = self._algorithm_combo.currentData() or "centroid"
        self.search_requested.emit(seeds, dataset, algorithm,
                                   self._limit_spin.value(),
                                   self._noise_methods())

    def show_similar_results(self, results: list) -> None:
        self._similar_cache = [(r.track_id, r.score) for r in results]
        self._similar_table.setRowCount(len(results))
        for row, res in enumerate(results):
            score_item = QTableWidgetItem(f"{res.score * 100:.1f} %")
            # First column (header stays "Filename") shows the FULL path:
            # filenames collide across folders, the path disambiguates. The
            # same text rides on the item's tooltip for hover readability
            # when the stretch column squeezes long paths.
            path_item = QTableWidgetItem(res.path)
            path_item.setToolTip(res.path)
            self._similar_table.setItem(row, 0, path_item)
            for col, text in enumerate(
                    [res.artist or "—", res.title or "—"], start=1):
                self._similar_table.setItem(row, col, QTableWidgetItem(text))
            self._similar_table.setItem(row, 3, score_item)
        self._playlist_button.setEnabled(bool(results))
        self._playlist_play_button.setEnabled(bool(results))
        self._copy_button.setEnabled(bool(results))
        self._set_similar_status("")

    def show_similar_error(self, message: str) -> None:
        self._similar_table.setRowCount(0)
        self._similar_cache = []
        self._playlist_button.setEnabled(False)
        self._playlist_play_button.setEnabled(False)
        self._copy_button.setEnabled(False)
        self._set_similar_status(f"⚠ {message}", error=True)

    def _result_paths(self) -> list[str]:
        """The result rows' file paths, deduplicated, in display order."""
        paths: list[str] = []
        seen: set[str] = set()
        for row in range(self._similar_table.rowCount()):
            item = self._similar_table.item(row, 0)
            if item is None:
                continue
            path = item.text()
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        return paths

    def _on_copy_results(self) -> None:
        """Copy every result file into a user-chosen folder."""
        paths = self._result_paths()
        if not paths:
            return
        dest_text = QFileDialog.getExistingDirectory(
            self, "Copy results into folder", "",
            QFileDialog.Option.ShowDirsOnly)
        if not dest_text:
            return
        import shutil
        from pathlib import Path

        dest = Path(dest_text)
        copied = same = renamed = failed = 0
        for path_text in paths:
            src_file = Path(path_text)
            try:
                if not src_file.is_file():
                    failed += 1
                    continue
                target = dest / src_file.name
                if target.exists():
                    if target.stat().st_size == src_file.stat().st_size:
                        same += 1        # byte-identical copy already there
                        continue
                    stem, suffix = src_file.stem, src_file.suffix
                    counter = 2
                    while True:
                        candidate = dest / f"{stem} ({counter}){suffix}"
                        if not candidate.exists():
                            break
                        counter += 1
                    target = candidate
                    renamed += 1
                shutil.copy2(src_file, target)
                copied += 1
            except OSError as exc:
                log.warning("Copy failed for %s: %s", src_file, exc)
                failed += 1
        parts = [f"{copied} file(s) copied"]
        if same:
            parts.append(f"{same} already present (identical, skipped)")
        if renamed:
            parts.append(f"{renamed} renamed (name collision)")
        if failed:
            parts.append(f"{failed} failed (missing/unreadable)")
        summary = "; ".join(parts) + f".\nFolder: {dest}"
        self._set_similar_status("✓ " + summary.replace("\n", " "))
        QMessageBox.information(self, "Copy results", summary)

    def _set_similar_status(self, text: str, error: bool = False) -> None:
        self._similar_status.setVisible(bool(text))
        self._similar_status.setText(text)
        self._similar_status.setStyleSheet(
            "color: #b00020;" if error else "color: gray;")

    def show_available_methods(self, models: list[str]) -> None:
        """Note under the search controls listing what this track can be
        searched by (models that have track-level embeddings)."""
        if models:
            self._set_similar_status(
                "Embeddings available for this track: " + ", ".join(models))
        else:
            self._set_similar_status(
                "⚠ No embeddings yet — analyze this track first.", error=True)

    def _on_create_playlist(self) -> None:
        if self._similar_cache:
            self.playlist_requested.emit(list(self._similar_cache),
                                         self._current_search_label())

    def _on_create_playlist_and_play(self) -> None:
        if self._similar_cache:
            self.playlist_play_requested.emit(list(self._similar_cache),
                                              self._current_search_label())

    def _on_similar_double_clicked(self, row: int, _column: int) -> None:
        """Play the double-clicked similar-track file in the system player."""
        if 0 <= row < len(self._similar_cache):
            self.play_track_requested.emit(int(self._similar_cache[row][0]))

    # ------------------------------------------------------------ playlists --
    # -------------------------------------------------------------- learning --
    def _build_learning_tab(self) -> QWidget:
        """Learning tab: song pairs + learned per-component weights.

        The user marks pairs of tracks that sound similar; the optimizer
        (app.learning.weights) learns which vector components carry that
        perceived similarity (largest learned weights) and which carry
        none (weights at zero).  The weights persist and scale every
        vector comparison in the Similar search.
        """
        from app.ui.workers import LearningWorker

        page = QWidget()
        layout = QVBoxLayout(page)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._learn_model_combo = QComboBox()
        self._learn_model_combo.setToolTip(
            "Which analysis model's vector components to learn weights for "
            "(primarily FFT — the components of its band statistics).")
        model_row.addWidget(self._learn_model_combo)
        refresh_btn = QPushButton("Refresh models")
        refresh_btn.clicked.connect(self._refresh_learn_models)
        model_row.addWidget(refresh_btn)
        model_row.addStretch(1)
        layout.addLayout(model_row)

        pair_box = QGroupBox("Similar song pairs (your ground truth)")
        pair_layout = QVBoxLayout(pair_box)
        pending_row = QHBoxLayout()
        self._learn_a_label = QLabel("Track A: —")
        self._learn_b_label = QLabel("Track B: —")
        pending_row.addWidget(self._learn_a_label)
        pending_row.addWidget(self._learn_b_label)
        pending_row.addStretch(1)
        pair_layout.addLayout(pending_row)
        self._learn_set_a_button = QPushButton("Set A from selection")
        self._learn_set_a_button.setToolTip(
            "Use the file currently selected in the tree as the pair's "
            "first track.")
        self._learn_set_a_button.clicked.connect(self._on_learn_set_a)
        pending_row.addWidget(self._learn_set_a_button)
        self._learn_set_b_button = QPushButton("Set B from selection")
        self._learn_set_b_button.setToolTip(
            "Use the file currently selected in the tree as the pair's "
            "second track.")
        self._learn_set_b_button.clicked.connect(self._on_learn_set_b)
        pending_row.addWidget(self._learn_set_b_button)
        self._pairs_list = QListWidget()
        self._pairs_list.setToolTip(
            "Track pairs that sound similar to you. The optimizer looks "
            "for components on which these pairs consistently agree.")
        pair_layout.addWidget(self._pairs_list)
        pair_buttons = QHBoxLayout()
        self._learn_add_pair_button = QPushButton("Add pair")
        self._learn_add_pair_button.clicked.connect(self._on_learn_add_pair)
        pair_buttons.addWidget(self._learn_add_pair_button)
        self._learn_remove_pair_button = QPushButton("Remove selected pair")
        self._learn_remove_pair_button.clicked.connect(
            self._on_learn_remove_pair)
        pair_buttons.addWidget(self._learn_remove_pair_button)
        pair_buttons.addStretch(1)
        pair_layout.addLayout(pair_buttons)
        layout.addWidget(pair_box)

        weight_box = QGroupBox(
            "Learned component weights (applied to every similarity search)")
        weight_layout = QVBoxLayout(weight_box)
        self._learn_summary = QLabel("No learned weights yet.")
        self._learn_summary.setWordWrap(True)
        weight_layout.addWidget(self._learn_summary)
        self._weights_table = QTableWidget(0, 2)
        self._weights_table.setHorizontalHeaderLabels(
            ["Component", "Weight"])
        self._weights_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._weights_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._weights_table.setAlternatingRowColors(True)
        self._weights_table.verticalHeader().setVisible(False)
        weight_layout.addWidget(self._weights_table)
        learn_buttons = QHBoxLayout()
        self._learn_button = QPushButton("Learn weights")
        self._learn_button.setToolTip(
            "Optimize per-component weights from the pairs above "
            "(ridge-regularized diagonal metric, closed-form solution). "
            "The weights persist and are applied to all similarity "
            "searches of this model.")
        self._learn_button.clicked.connect(self._on_learn_weights)
        learn_buttons.addWidget(self._learn_button)
        self._learn_clear_button = QPushButton("Clear learned weights")
        self._learn_clear_button.clicked.connect(self._on_learn_clear)
        learn_buttons.addWidget(self._learn_clear_button)
        learn_buttons.addStretch(1)
        weight_layout.addLayout(learn_buttons)
        layout.addWidget(weight_box)

        self._learn_a: int | None = None
        self._learn_b: int | None = None
        self._learn_worker: LearningWorker | None = None
        self._refresh_learn_models()
        self._refresh_pairs()
        self._load_persisted_weights()
        return page

    def _learn_model(self) -> str:
        return str(self._learn_model_combo.currentData() or "fft")

    def _refresh_learn_models(self) -> None:
        """Fill the model combo from the datasets that have chunk vectors."""
        conn = self._db.connect()
        try:
            models = sorted(
                str(r["model"]) for r in conn.execute(
                    "SELECT DISTINCT model FROM embeddings")
                if not str(r["model"]).startswith(("red:", "ollama:")))
        finally:
            conn.close()
        self._learn_model_combo.blockSignals(True)
        self._learn_model_combo.clear()
        for model in models:
            self._learn_model_combo.addItem(model, model)
        if "fft" in models:
            self._learn_model_combo.setCurrentIndex(
                self._learn_model_combo.findData("fft"))
        self._learn_model_combo.blockSignals(False)

    def _refresh_pairs(self) -> None:
        from app.learning.weights import list_pairs

        conn = self._db.connect()
        try:
            pairs = list_pairs(conn)
        finally:
            conn.close()
        self._pairs_list.clear()
        for p in pairs:
            item = QListWidgetItem(
                f"{p['name_a']}  ↔  {p['name_b']}")
            item.setData(Qt.ItemDataRole.UserRole, int(p["pair_id"]))
            item.setToolTip(f"{p['name_a']} ↔ {p['name_b']}")
            self._pairs_list.addItem(item)

    def _on_learn_set_a(self) -> None:
        if self._current_track_id is None:
            return
        self._learn_a = int(self._current_track_id)
        self._learn_a_label.setText(
            f"Track A: {self._learn_track_label(self._learn_a)}")

    def _on_learn_set_b(self) -> None:
        if self._current_track_id is None:
            return
        self._learn_b = int(self._current_track_id)
        self._learn_b_label.setText(
            f"Track B: {self._learn_track_label(self._learn_b)}")

    def _learn_track_label(self, track_id: int) -> str:
        conn = self._db.connect()
        try:
            row = repo.get_track(conn, track_id)
        finally:
            conn.close()
        return str(row["filename"]) if row else f"#{track_id}"

    def _on_learn_add_pair(self) -> None:
        from app.learning.weights import add_pair

        if self._learn_a is None or self._learn_b is None:
            self._learn_summary.setText(
                "Set track A and track B from the tree selection first.")
            return
        conn = self._db.connect()
        try:
            pair_id = add_pair(conn, self._learn_a, self._learn_b)
            conn.commit()
        finally:
            conn.close()
        if pair_id is None:
            self._learn_summary.setText(
                "Pair already exists (or both slots hold the same track).")
            return
        self._learn_a = None
        self._learn_b = None
        self._learn_a_label.setText("Track A: —")
        self._learn_b_label.setText("Track B: —")
        self._refresh_pairs()

    def _on_learn_remove_pair(self) -> None:
        from app.learning.weights import remove_pair

        item = self._pairs_list.currentItem()
        if item is None:
            return
        pair_id = int(item.data(Qt.ItemDataRole.UserRole))
        conn = self._db.connect()
        try:
            remove_pair(conn, pair_id)
            conn.commit()
        finally:
            conn.close()
        self._refresh_pairs()

    def _on_learn_weights(self) -> None:
        from app.ui.workers import LearningWorker

        if self._learn_worker is not None and self._learn_worker.isRunning():
            self._learn_summary.setText("Learning already running…")
            return
        model = self._learn_model()
        self._learn_summary.setText(f"Learning weights for '{model}'…")
        self._learn_button.setEnabled(False)
        self._learn_worker = LearningWorker(self._db.db_path, model)
        self._learn_worker.learned.connect(self._on_learned)
        self._learn_worker.failed.connect(self._on_learn_failed)
        self._learn_worker.start()

    def _weight_component_name(self, index: int) -> str:
        if self._learn_model() == "fft":
            from app.models.fft_model import feature_names

            names = feature_names()
            if index < len(names):
                return str(names[index])
        return f"dim {index}"

    def _show_weights(self, model: str, weights) -> None:
        import numpy as np

        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        order = np.argsort(w)[::-1]   # most similarity-carrying first
        self._weights_table.setRowCount(w.size)
        for row, idx in enumerate(order):
            name_item = QTableWidgetItem(self._weight_component_name(int(idx)))
            self._weights_table.setItem(row, 0, name_item)
            self._weights_table.setItem(
                row, 1, QTableWidgetItem(f"{w[idx]:.4f}"))
        top = [self._weight_component_name(int(i)) for i in order[:3]]
        bottom = [self._weight_component_name(int(i)) for i in order[-3:]]
        self._learn_summary.setText(
            f"Model '{model}': MOST similarity-carrying components: "
            f"{', '.join(top)} — LEAST: {', '.join(bottom)}. The weights "
            "scale every similarity search of this model.")

    def _on_learned(self, model: str, pairs_used: int, weights) -> None:
        self._learn_button.setEnabled(True)
        self._show_weights(model, weights)
        self._learn_summary.setText(
            f"Learned from {pairs_used} pair(s). "
            + self._learn_summary.text())

    def _on_learn_failed(self, message: str) -> None:
        self._learn_button.setEnabled(True)
        self._learn_summary.setText(message)

    def _on_learn_clear(self) -> None:
        from app.learning.weights import clear_weights

        model = self._learn_model()
        conn = self._db.connect()
        try:
            clear_weights(conn, model)
            conn.commit()
        finally:
            conn.close()
        self._weights_table.setRowCount(0)
        self._learn_summary.setText(f"Cleared learned weights of '{model}'.")

    def _load_persisted_weights(self) -> None:
        """Show weights learned in a previous session, if any."""
        from app.learning.weights import load_weight_vector

        model = self._learn_model()
        conn = self._db.connect()
        try:
            weights = load_weight_vector(conn, model)
        finally:
            conn.close()
        if weights is not None:
            self._show_weights(model, weights)

    def _build_playlists_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        left = QVBoxLayout()
        left.addWidget(QLabel("Playlists"))
        self._playlist_list = QListWidget()
        self._playlist_list.currentRowChanged.connect(lambda _: self._refresh_playlist_items())
        left.addWidget(self._playlist_list)
        buttons = QHBoxLayout()
        self._export_button = QPushButton("Export .m3u…")
        self._delete_button = QPushButton("Delete")
        self._export_button.clicked.connect(self._on_export)
        self._delete_button.clicked.connect(self._on_delete)
        buttons.addWidget(self._export_button)
        buttons.addWidget(self._delete_button)
        left.addLayout(buttons)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(280)
        layout.addWidget(left_widget)

        right = QVBoxLayout()
        self._playlist_items = QTableWidget(0, 4)
        self._playlist_items.setHorizontalHeaderLabels(
            ["#", "Filename", "Artist", "Similarity"])
        self._playlist_items.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._playlist_items.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._playlist_items.verticalHeader().setVisible(False)
        right.addWidget(self._playlist_items)
        right_widget = QWidget()
        right_widget.setLayout(right)
        layout.addWidget(right_widget)
        return page

    def refresh_playlists(self, select_id: int | None = None) -> None:
        self._playlist_list.blockSignals(True)
        self._playlist_list.clear()
        conn = self._db.connect()
        try:
            playlists = repo.list_playlists(conn)
        finally:
            conn.close()
        target_row = 0
        for i, pl in enumerate(playlists):
            label = f"{pl['name']} ({pl['created_at'][:16]})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, int(pl["id"]))
            self._playlist_list.addItem(item)
            if select_id is not None and int(pl["id"]) == int(select_id):
                target_row = i
        self._playlist_list.blockSignals(False)
        if self._playlist_list.count():
            self._playlist_list.setCurrentRow(target_row)
        self._refresh_playlist_items()

    def _current_playlist_id(self) -> int | None:
        item = self._playlist_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _refresh_playlist_items(self) -> None:
        playlist_id = self._current_playlist_id()
        self._playlist_items.setRowCount(0)
        if playlist_id is None:
            return
        conn = self._db.connect()
        try:
            items = repo.get_playlist_items(conn, playlist_id)
        finally:
            conn.close()
        self._playlist_items.setRowCount(len(items))
        for row, it in enumerate(items):
            similarity = it["similarity"]
            values = [str(it["position"]), it["filename"], it["artist"] or "—",
                      f"{similarity * 100:.1f} %" if similarity is not None else "—"]
            for col, text in enumerate(values):
                self._playlist_items.setItem(row, col, QTableWidgetItem(text))

    def _on_export(self) -> None:
        playlist_id = self._current_playlist_id()
        if playlist_id is None:
            return
        window = self.window()
        if hasattr(window, "export_playlist"):
            window.export_playlist(playlist_id)

    def _on_delete(self) -> None:
        playlist_id = self._current_playlist_id()
        if playlist_id is None:
            return
        with self._db.transaction() as conn:
            repo.delete_playlist(conn, playlist_id)
        self.refresh_playlists()
        self.playlist_changed.emit()

    # --------------------------------------------------------------- track --
    def show_track(self, track_id: int | None) -> None:
        self._current_track_id = track_id
        # keep the multi-reference list in sync (auto entry follows the
        # selection; manually added references stay)
        self._sync_seed_list_with_selection()
        conn = self._db.connect()
        try:
            if track_id is None:
                self._meta_table.setRowCount(len(_META_ROWS))
                for row in range(self._meta_table.rowCount()):
                    item = self._meta_table.item(row, 1)
                    item.setText("—")
                    item.setToolTip("")  # drop the previous track's note
                self._description.setPlainText("")
                self._chunks_table.setRowCount(0)
                self._chunks_hint.setText("Select a track in the tree.")
                self._set_warning_banner([])
                self._set_similar_status("")
                return
            track = repo.get_track(conn, track_id)
            if track is None:
                return
            notes = [n.strip() for n in (track["status_message"] or "").split(";")
                     if n.strip()]
            for row, (key, _label) in enumerate(_META_ROWS):
                value = track[key]
                item = self._meta_table.item(row, 1)
                if key == "status_message" and value:
                    # quiet presentation: long analysis notes are truncated
                    # in the cell, the full text stays reachable on hover
                    item.setToolTip(str(value))
                    item.setText(_truncate(str(value), 100))
                else:
                    item.setToolTip("")
                    item.setText(_human(key, value))
            self._description.setPlainText(track["description"] or "")

            # --- warning banner: hard failures + per-model problems ---------
            warnings = []
            if track["status"] == "error":
                warnings.append(track["status_message"] or "Analysis failed")
            else:
                warnings = [n for n in notes
                            if "failed:" in n or "unavailable" in n]
            self._set_warning_banner(warnings)

            # --- which similarity methods exist for this track? -------------
            rows = conn.execute(
                "SELECT model FROM track_embeddings WHERE track_id=? ORDER BY model",
                (track_id,)).fetchall()
            display = {info["name"]: info["display_name"]
                       for info in plugin_info()}
            self.show_available_methods(
                [display.get(r["model"], r["model"]) for r in rows])

            self._refresh_chunks(conn, track_id, notes, self._config.models)
        finally:
            conn.close()

    def _set_warning_banner(self, warnings: list[str]) -> None:
        """Quiet banner: one short line visible, full diagnostics on hover.

        Several problems collapse to a summary line so the banner can never
        grow into a tall loud block; a hard error keeps its (truncated) first
        line visible because the user must still see what failed.
        """
        self._warning.setVisible(bool(warnings))
        if not warnings:
            self._warning.setText("")
            self._warning.setToolTip("")
            return
        if len(warnings) > 1:
            visible = f"⚠ {len(warnings)} model warnings — hover for details"
        else:
            visible = f"⚠ {_truncate(warnings[0])}"
        self._warning.setText(visible)
        self._warning.setToolTip("\n".join(f"⚠ {w}" for w in warnings))

    def current_track_id(self) -> int | None:
        return self._current_track_id
