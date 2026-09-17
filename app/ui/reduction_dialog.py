"""The "Dimensionality reduction" dialog: build searchable reduced datasets.

Reduces one model's chunk vectors (CLAP/MERT/OpenL3/FFT) to *n* components
with PCA (built-in), UMAP or t-SNE and stores the result as a new analysis
dataset the Similar tab can search like a model (dataset ``red:<id>``).
Runs the fit on a background worker with a progress bar + stage/elapsed
readout, since a whole-library UMAP takes minutes.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout,
)

from app.analysis import dim_reduction
from app.db import repo
from app.models.registry import plugin_info
from app.ui.workers import ReductionWorker

#: Components allowed per method (t-SNE is practically a 2-D technique).
_COMPONENT_RANGES = {"pca": (2, 128), "umap": (2, 32), "tsne": (2, 2)}


class ReductionDialog(QDialog):
    """Pick source dataset + method + parameters and run the reduction."""

    #: Emitted after a successful run with the dataset id ("red:3" — for a
    #: refresh this is the SAME id as the refreshed reduction).
    reduction_created = Signal(str)

    def __init__(self, db, config, parent=None,
                 rerun_of=None) -> None:
        """*rerun_of*: a ``reductions`` row to refresh in place.

        Refresh mode prefills every parameter from the stored row, locks
        source + name, and re-fits OVER the existing reduction id: the
        dataset keeps its ``red:<id>`` references and becomes current
        again.
        """
        super().__init__(parent)
        self._rerun_id = int(rerun_of["id"]) if rerun_of is not None else None
        self.setWindowTitle(
            "Refresh dimensionality reduction" if self._rerun_id is not None
            else "Dimensionality reduction")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._db = db
        self._worker = None
        self._name_dirty = False
        self._fit_started_at: float | None = None

        layout = QVBoxLayout(self)
        if rerun_of is not None:
            import json

            params = json.loads(str(rerun_of["params"] or "{}"))
            covered, total = repo.reduction_coverage(
                conn=None, reduction_id=0, source_model="") if False else (0, 0)
            with self._db.transaction() as conn:
                covered, total = repo.reduction_coverage(
                    conn, int(rerun_of["id"]), str(rerun_of["source_model"]))
            stale = covered < total
            header = QLabel(
                f"Refreshing '{rerun_of['name']}' — the stored projection "
                f"covers {covered:,} of {total:,} current chunk vectors"
                + ("" if stale else " (up to date)") + ".")
            header.setWordWrap(True)
            header.setStyleSheet(
                "color: #8a6d3b; font-size: 11px;" if stale
                else "color: gray; font-size: 11px;")
            layout.addWidget(header)
        else:
            params = {}
        form = QFormLayout()

        self._source_combo = QComboBox()
        with self._db.transaction() as conn:
            chunk_models = repo.get_chunk_vector_models(conn)
            self._vector_counts = repo.chunk_vector_counts(conn)
        for info in plugin_info():
            if info["name"] in chunk_models:
                self._source_combo.addItem(info["display_name"], info["name"])
        self._source_combo.currentIndexChanged.connect(self._update_name)
        self._source_combo.currentIndexChanged.connect(
            lambda _index: self._sync_method_availability())
        form.addRow("Source dataset:", self._source_combo)

        self._method_combo = QComboBox()
        available = dim_reduction.availability()
        for method in dim_reduction.METHODS:
            label = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[method]
            self._method_combo.addItem(label, method)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self._method_combo)

        self._components_spin = QSpinBox()
        self._components_spin.setRange(2, 128)
        self._components_spin.setValue(32)
        self._components_spin.setToolTip(
            "Target dimensionality of the reduced dataset (2–128).")
        self._components_spin.valueChanged.connect(self._update_name)
        form.addRow("Components:", self._components_spin)

        self._neighbors_spin = QSpinBox()
        self._neighbors_spin.setRange(2, 200)
        self._neighbors_spin.setValue(15)
        self._neighbors_spin.setToolTip(
            "UMAP n_neighbors: size of the local neighbourhood UMAP "
            "optimises for (small = local detail, large = global structure).")
        form.addRow("UMAP neighbors:", self._neighbors_spin)

        self._min_dist_spin = QDoubleSpinBox()
        self._min_dist_spin.setRange(0.0, 1.0)
        self._min_dist_spin.setSingleStep(0.05)
        self._min_dist_spin.setValue(0.1)
        self._min_dist_spin.setToolTip(
            "UMAP min_dist: minimum spacing between points in the "
            "projection.")
        form.addRow("UMAP min dist:", self._min_dist_spin)

        self._perplexity_spin = QSpinBox()
        self._perplexity_spin.setRange(5, 100)
        self._perplexity_spin.setValue(30)
        self._perplexity_spin.setToolTip(
            "t-SNE perplexity: effective number of neighbours per point.")
        form.addRow("t-SNE perplexity:", self._perplexity_spin)

        self._name_edit = QLineEdit()
        self._name_edit.textEdited.connect(lambda: self._mark_name_dirty())
        form.addRow("Dataset name:", self._name_edit)
        layout.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)      # idle: busy-style
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._stage_label = QLabel(
            "The reduced dataset becomes searchable in the Similar tab "
            "(dataset picker) right after the run.")
        self._stage_label.setWordWrap(True)
        layout.addWidget(self._stage_label)

        buttons = QHBoxLayout()
        self._run_button = QPushButton("&Reduce")
        self._run_button.clicked.connect(self._run)
        buttons.addWidget(self._run_button)
        buttons.addStretch(1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        # Elapsed-time ticker for the (indeterminate) fit phase.
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_elapsed)

        if rerun_of is not None:
            # Prefill from the stored row and lock the identity fields: a
            # refresh re-fits the SAME projection over the current data.
            self._source_combo.setCurrentIndex(
                self._source_combo.findData(str(rerun_of["source_model"])))
            self._source_combo.setEnabled(False)
            index = self._method_combo.findData(str(rerun_of["method"]))
            if index >= 0:
                self._method_combo.setCurrentIndex(index)
            self._components_spin.setValue(int(rerun_of["n_components"]))
            self._neighbors_spin.setValue(int(params.get("n_neighbors", 15)))
            self._min_dist_spin.setValue(float(params.get("min_dist", 0.1)))
            self._perplexity_spin.setValue(int(params.get("perplexity", 30)))
            self._name_edit.setText(str(rerun_of["name"]))
            self._name_edit.setEnabled(False)
            self._name_dirty = True     # never auto-suggest over the name
            self._run_button.setText("&Refresh")
        self._on_method_changed()
        self._sync_method_availability()
        self._update_name()

    # ---- helpers ------------------------------------------------------------
    def _current_method(self) -> str:
        return self._method_combo.currentData() or "pca"

    def _sync_method_availability(self) -> None:
        """Enable method items by library availability AND dataset size.

        A method over the dataset's point cap is greyed out with the
        reason on the tooltip — better than a runtime error after the
        user presses Reduce.
        """
        available = dim_reduction.availability()
        source = self._source_combo.currentData()
        count = self._vector_counts.get(source, 0)
        caps = {"tsne": dim_reduction.TSNE_MAX_POINTS,
                "umap": dim_reduction.UMAP_MAX_POINTS}
        for index in range(self._method_combo.count()):
            method = self._method_combo.itemData(index)
            label = self._method_combo.itemText(index)
            item = self._method_combo.model().item(index)
            if item is None:
                continue
            missing = available.get(method)
            cap = caps.get(method)
            over = cap is not None and count > cap
            item.setEnabled(missing is None and not over)
            if missing:
                item.setToolTip(missing)
            elif over:
                item.setToolTip(
                    f"This dataset has {count:,} chunk vectors — {label} "
                    f"is limited to {cap:,} per run. Use PCA, or analyze "
                    "fewer tracks.")
            else:
                item.setToolTip(
                    f"{count:,} chunk vectors" if method != "pca" else "")
        # If the selected method just became unavailable (source switch),
        # fall back to the first usable one so Reduce stays runnable.
        current = self._method_combo.model().item(
            self._method_combo.currentIndex())
        if current is not None and not current.isEnabled():
            for index in range(self._method_combo.count()):
                if self._method_combo.model().item(index).isEnabled():
                    self._method_combo.setCurrentIndex(index)
                    break

    def _suggested_name(self) -> str:
        source = self._source_combo.currentData() or "model"
        method = self._current_method()
        label = {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}[method]
        return f"{label}-{self._components_spin.value()} of {source}"

    def _update_name(self) -> None:
        if not self._name_dirty:
            self._name_edit.setText(self._suggested_name())

    def _mark_name_dirty(self) -> None:
        self._name_dirty = True

    def _on_method_changed(self) -> None:
        method = self._current_method()
        lo, hi = _COMPONENT_RANGES[method]
        self._components_spin.setRange(lo, hi)
        if not (lo <= self._components_spin.value() <= hi):
            self._components_spin.setValue(min(32, hi))
        is_umap = method == "umap"
        is_tsne = method == "tsne"
        self._neighbors_spin.setEnabled(is_umap)
        self._min_dist_spin.setEnabled(is_umap)
        self._perplexity_spin.setEnabled(is_tsne)
        self._update_name()

    def _tick_elapsed(self) -> None:
        if self._fit_started_at is not None:
            elapsed = int(time.monotonic() - self._fit_started_at)
            self._stage_label.setText(
                f"Fitting… {elapsed // 60}m {elapsed % 60:02d}s elapsed "
                "(the progress bar is busy while the solver runs)")

    # ---- run ----------------------------------------------------------------
    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        source = self._source_combo.currentData()
        if not source:
            self._stage_label.setText(
                "No model has chunk vectors yet — analyze tracks first.")
            return
        name = self._name_edit.text().strip() or self._suggested_name()
        self._worker = ReductionWorker(
            self._db.db_path,
            source, self._current_method(),
            self._components_spin.value(), name,
            n_neighbors=self._neighbors_spin.value(),
            min_dist=self._min_dist_spin.value(),
            perplexity=self._perplexity_spin.value(),
            replace_reduction_id=self._rerun_id)
        self._worker.stage.connect(self._stage_label.setText)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._set_controls_enabled(False)
        self._progress.setRange(0, 0)      # busy while loading+fitting
        self._fit_started_at = time.monotonic()
        self._timer.start()
        self._worker.start()

    def _on_progress(self, value: int) -> None:
        if self._progress.maximum() != 100:
            self._progress.setRange(0, 100)
        self._progress.setValue(max(self._progress.value(), value))

    def _on_finished(self, reduction_id: int, name: str,
                     n_vectors: int) -> None:
        self._timer.stop()
        self._fit_started_at = None
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        note = ""
        with self._db.transaction() as conn:
            row = repo.get_reduction(conn, reduction_id)
        if row is not None and row["explained_variance"]:
            import json

            ratios = json.loads(row["explained_variance"])
            total = sum(ratios) * 100.0
            note = f" — variance captured: {total:.1f}%"
        verb = ("refreshed" if self._rerun_id is not None else "created")
        self._stage_label.setText(
            f"✓ Dataset '{name}' {verb}: {n_vectors:,} chunk vectors, "
            f"searchable as red:{reduction_id}{note}")
        self._set_controls_enabled(True)
        self.reduction_created.emit(f"red:{reduction_id}")

    def _on_failed(self, message: str) -> None:
        self._timer.stop()
        self._fit_started_at = None
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._stage_label.setText(f"⚠ {message}")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self._method_combo,
                       self._components_spin, self._neighbors_spin,
                       self._min_dist_spin, self._perplexity_spin,
                       self._run_button):
            widget.setEnabled(enabled)
        # Source + name are locked in refresh mode, always.
        identity = enabled and self._rerun_id is None
        self._source_combo.setEnabled(identity)
        self._name_edit.setEnabled(identity)
