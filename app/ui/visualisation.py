"""The "Visualisation" dialog: scatter plots of the analysis datasets.

Each point is one analysed chunk.  Two viewing modes:

* **Raw components** — pick any component of any model for X and any
  component of any model for Y (cross-model combinations allowed).
* **Reduced to 2-D** — pick PCA / t-SNE / UMAP and the models to feed in
  (any subset of the models that produced data); every chunk that has all
  selected models' vectors is projected to 2-D and plotted.

Data comes straight from the ``embeddings`` table, so the plot always
reflects the current analysis state.  t-SNE / UMAP need their optional
libraries (scikit-learn / umap-learn); PCA is built in.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from app.analysis import dim_reduction
from app.db import repo
from app.models.registry import plugin_info
from app.ui.scatter_plot import ScatterCanvas

#: Cap for raw-component plots (beyond this, points are subsampled).
RAW_MAX_POINTS = 50_000

#: Percentage-of-points formatting for the status note.
def _fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


_DISPLAY_NAMES: dict[str, str] = {}


def _display_name(model: str) -> str:
    """Display name for a plugin id, cached — plugin_info() probes optional
    dependencies via importlib, which is far too slow to call per point."""
    if model not in _DISPLAY_NAMES:
        for info in plugin_info():
            _DISPLAY_NAMES[info["name"]] = info["display_name"]
    return _DISPLAY_NAMES.get(model, model)


class VisualisationDialog(QDialog):
    """Pick X/Y sources (model + component, or a reduction) and plot."""

    def __init__(self, db, config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Visualisation")
        self.setModal(False)
        self.resize(1000, 700)
        self._db = db
        self._config = config
        self._data_loaded = False
        # Per-model view of the gathered rows, all aligned by position:
        self._model_names: list[str] = []
        self._vectors: dict[str, np.ndarray] = {}       # model → (n, dim)
        self._chunk_keys: dict[str, list[tuple]] = {}   # model → per-row key
        self._track_colors: dict[str, int] = {}

        layout = QVBoxLayout(self)

        # ---- controls -----------------------------------------------------
        controls = QHBoxLayout()

        def _source_box(title: str) -> tuple[QComboBox, QSpinBox]:
            combo = QComboBox()
            spin = QSpinBox()
            spin.setMinimum(0)
            spin.setMaximum(0)
            spin.setToolTip(f"Component index of the {title} source vector")
            box = QHBoxLayout()
            box.addWidget(QLabel(title))
            box.addWidget(combo, 1)
            box.addWidget(QLabel("component"))
            box.addWidget(spin)
            controls.addLayout(box)
            return combo, spin

        self._x_model, self._x_comp = _source_box("X:")
        self._y_model, self._y_comp = _source_box("Y:")
        self._x_model.currentIndexChanged.connect(self._on_axis_changed)
        self._y_model.currentIndexChanged.connect(self._on_axis_changed)
        self._x_comp.valueChanged.connect(self._on_axis_changed)
        self._y_comp.valueChanged.connect(self._on_axis_changed)
        layout.addLayout(controls)
        # Parameter changes re-plot live (debounced; instant methods only).
        self._replot_timer = QTimer(self)
        self._replot_timer.setSingleShot(True)
        self._replot_timer.setInterval(250)
        self._replot_timer.timeout.connect(self._auto_plot)

        reduce_row = QHBoxLayout()
        reduce_row.addWidget(QLabel("Dimensionality reduction:"))
        self._reduce_combo = QComboBox()
        self._reduce_combo.addItem("None (raw components)", "none")
        self._reduce_combo.addItem("PCA", "pca")
        self._reduce_combo.addItem("t-SNE (slow)", "tsne")
        self._reduce_combo.addItem("UMAP", "umap")
        self._reduce_combo.setToolTip(
            "Project the selected models' chunk vectors to 2-D before "
            "plotting. t-SNE needs scikit-learn, UMAP needs umap-learn.")
        self._reduce_combo.currentIndexChanged.connect(
            self._on_reduction_changed)
        reduce_row.addWidget(self._reduce_combo)
        reduce_row.addSpacing(14)
        reduce_row.addWidget(QLabel("Models to reduce:"))
        self._reduce_checks: dict[str, QCheckBox] = {}
        for info in plugin_info():
            check = QCheckBox(info["display_name"])
            check.setChecked(True)
            check.toggled.connect(self._on_reduction_changed)
            self._reduce_checks[info["name"]] = check
            reduce_row.addWidget(check)
        self._plot_button = QPushButton("&Plot")
        self._plot_button.setDefault(True)
        self._plot_button.clicked.connect(self.refresh_and_plot)
        reduce_row.addWidget(self._plot_button)
        reduce_row.addStretch(1)
        layout.addLayout(reduce_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._canvas = ScatterCanvas()
        layout.addWidget(self._canvas, 1)

        self._sync_reduction_availability()

    # ---- data ---------------------------------------------------------------
    def _load_data(self) -> None:
        """(Re-)read every chunk embedding from the database."""
        with self._db.transaction() as conn:
            rows = repo.get_chunk_embedding_rows(conn)
        self._model_names = []
        per_model: dict[str, list[tuple]] = {}
        for row in rows:
            model = row["model"]
            if model not in per_model:
                per_model[model] = []
                self._model_names.append(model)
            per_model[model].append((row["vec"], int(row["chunk_id"]),
                                     row["track_path"], row["track_filename"],
                                     int(row["chunk_idx"]),
                                     float(row["start_sec"])))
        self._vectors = {}
        self._chunk_keys = {}
        for model, items in per_model.items():
            self._vectors[model] = np.stack([item[0] for item in items]) \
                if items else np.zeros((0, 0))
            self._chunk_keys[model] = [item[1:] for item in items]
        self._data_loaded = True
        self._populate_model_combos()

    def _populate_model_combos(self) -> None:
        """Fill X/Y model combos from the models that actually have data.

        The user's selection SURVIVES a reload (every "Plot" click re-reads
        the database): X/Y model and component are restored — clamped to
        what still exists — and reduction checkboxes keep their checked
        state.  Only the very first population applies the defaults
        (first model, components 0/1, every data-bearing model checked).
        """
        prev = {
            "x_model": self._x_model.currentData(),
            "y_model": self._y_model.currentData(),
            "x_comp": self._x_comp.value(),
            "y_comp": self._y_comp.value(),
        }
        for combo in (self._x_model, self._y_model):
            prev_model = prev["x_model" if combo is self._x_model else "y_model"]
            combo.blockSignals(True)
            combo.clear()
            for model in self._model_names:
                combo.addItem(_display_name(model), model)
            index = combo.findData(prev_model) if prev_model else -1
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)
        if self._model_names:
            # Maxima first, then values — a QSpinBox clamps to its maximum.
            for combo, spin, model_key, comp_key, default_comp in (
                    (self._x_model, self._x_comp, "x_model", "x_comp", 0),
                    (self._y_model, self._y_comp, "y_model", "y_comp", 1)):
                model = combo.currentData()
                top = max(0, self._dim(model) - 1)
                # programmatic restores must not schedule a re-plot
                spin.blockSignals(True)
                spin.setMaximum(top)
                if prev[model_key] is not None and prev[model_key] == model:
                    spin.setValue(min(prev[comp_key], top))
                else:
                    spin.setValue(min(default_comp, top))
                spin.blockSignals(False)
        self._sync_component_range()
        # Reduction checkboxes: only models that produced data are usable;
        # the user's checked state survives a reload.
        for model, check in self._reduce_checks.items():
            has_data = model in self._vectors and self._vectors[model].size
            check.setEnabled(has_data)
            if not has_data:
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)

    def _dim(self, model: str) -> int:
        vectors = self._vectors.get(model)
        return int(vectors.shape[1]) if vectors is not None and vectors.size else 0

    # ---- live parameter updates ------------------------------------------
    def _on_axis_changed(self, *_args) -> None:
        """X/Y model or component changed — re-sync ranges, re-plot soon."""
        self._sync_component_range()
        self._replot_timer.start()

    def _on_reduction_changed(self, *_args) -> None:
        """Reduction method or model subset changed."""
        method = self._reduce_combo.currentData()
        if method == "none" and self.sender() in self._reduce_checks.values():
            return   # checkboxes do not affect raw-component plots
        self._replot_timer.start()

    def _auto_plot(self) -> None:
        """Live re-plot after a parameter change (debounced).

        Only the instant methods re-plot automatically; t-SNE / UMAP take
        seconds to minutes, so they wait for an explicit Plot click (the
        status line says so).
        """
        if not self.isVisible() or not self._data_loaded:
            return
        method = self._reduce_combo.currentData()
        if method in ("tsne", "umap"):
            self._status_label.setText(
                f"{method.upper()} is slow — press Plot to (re-)run it.")
            return
        self._plot()

    def _sync_component_range(self) -> None:
        for combo, spin in ((self._x_model, self._x_comp),
                            (self._y_model, self._y_comp)):
            model = combo.currentData()
            top = max(0, self._dim(model) - 1) if model else 0
            # programmatic changes must not schedule a re-plot
            spin.blockSignals(True)
            spin.setMaximum(top)
            if spin.value() > top:
                spin.setValue(0)
            spin.blockSignals(False)

    def _sync_reduction_availability(self) -> None:
        """Disable reduction methods whose library is missing + tooltips."""
        available = dim_reduction.availability()
        for index in range(self._reduce_combo.count()):
            method = self._reduce_combo.itemData(index)
            missing = available.get(method)
            item = self._reduce_combo.model().item(index)
            if item is not None:
                item.setEnabled(missing is None)
                item.setToolTip(missing or "")

    # ---- plotting -----------------------------------------------------------
    def refresh_and_plot(self) -> None:
        """Reload the data from the database and (re-)plot."""
        self._status_label.setText("Loading embeddings…")
        self._load_data()
        if not self._model_names:
            self._canvas.clear_plot("No analyzed chunks yet — run an "
                                    "analysis first")
            self._status_label.setText(
                "No chunk embeddings found. Analyze some tracks first.")
            return
        self._plot()

    def _plot(self) -> None:
        method = self._reduce_combo.currentData()
        if method == "none":
            self._plot_raw()
        else:
            self._plot_reduced(method)

    def _track_color(self, track_path: str) -> int:
        if track_path not in self._track_colors:
            self._track_colors[track_path] = len(self._track_colors)
        return self._track_colors[track_path]

    def _plot_raw(self) -> None:
        model_x = self._x_model.currentData()
        model_y = self._y_model.currentData()
        for model in (model_x, model_y):
            if not model or not self._vectors.get(model, np.zeros((0, 0))).size:
                self._canvas.clear_plot("No data for the selected model")
                return
        comp_x = int(self._x_comp.value())
        comp_y = int(self._y_comp.value())
        if comp_x >= self._dim(model_x) or comp_y >= self._dim(model_y):
            self._canvas.clear_plot()
            self._status_label.setText("Component index out of range.")
            return
        if model_x == model_y:
            # Same model: plot every chunk of that model directly.
            keys = self._chunk_keys[model_x]
            xs = self._vectors[model_x][:, comp_x].astype(np.float64)
            ys = self._vectors[model_x][:, comp_y].astype(np.float64)
        else:
            # Cross-model combination: only chunks carrying BOTH models,
            # aligned row by row.
            common = set(self._chunk_keys[model_x]) \
                & set(self._chunk_keys[model_y])
            ix = {key: i for i, key in enumerate(self._chunk_keys[model_x])}
            iy = {key: i for i, key in enumerate(self._chunk_keys[model_y])}
            keys = [key for key in self._chunk_keys[model_x] if key in common]
            xs = self._vectors[model_x][[ix[k] for k in keys],
                                        comp_x].astype(np.float64)
            ys = self._vectors[model_y][[iy[k] for k in keys],
                                        comp_y].astype(np.float64)
        if not xs.size:
            self._canvas.clear_plot()
            self._status_label.setText(
                "No chunk carries both selected models' vectors.")
            return
        # Subsample the rows BEFORE building hover labels — label strings
        # for 141k chunks dominate re-plot time on real libraries.
        note = ""
        if xs.size > RAW_MAX_POINTS:
            total = int(xs.size)
            keep = VisualisationDialog._subsample_indices(
                xs.size, RAW_MAX_POINTS)
            xs, ys = xs[keep], ys[keep]
            keys = [keys[i] for i in keep]
            note = (f"subsampled {_fmt_int(RAW_MAX_POINTS)} of "
                    f"{_fmt_int(total)} points for this method")
        labels = [
            f"{key[2]} — chunk {key[3]} @ {key[4]:.0f}s "
            f"({_display_name(model_x)}[{comp_x}])"
            for key in keys
        ]
        colors = [self._track_color(key[1]) for key in keys]
        x_label = f"{_display_name(model_x)}[{comp_x}]"
        y_label = f"{_display_name(model_y)}[{comp_y}]"
        self._canvas.set_plot(xs, ys, x_label, y_label, labels, colors,
                              status=note)
        n = xs.size
        cross = "" if model_x == model_y else \
            f" (cross-model: Y from {_display_name(model_y)})"
        self._status_label.setText(
            f"{_fmt_int(n)} chunks — X: {x_label}, Y: {y_label}{cross}"
            + (f" — {note}" if note else ""))

    def _plot_reduced(self, method: str) -> None:
        chosen = [model for model, check in self._reduce_checks.items()
                  if check.isEnabled() and check.isChecked()]
        if not chosen:
            self._canvas.clear_plot()
            self._status_label.setText(
                "Select at least one model to reduce.")
            return
        # Chunks that have every chosen model, aligned row by row.
        key_sets = [set(self._chunk_keys[model]) for model in chosen]
        common = set.intersection(*key_sets) if key_sets else set()
        if not common:
            self._canvas.clear_plot()
            self._status_label.setText(
                "No chunk carries all the selected models' vectors. "
                "Analyze the tracks with all of them first.")
            return
        primary = chosen[0]
        order = [key for key in self._chunk_keys[primary] if key in common]
        # Subsample the chunk list BEFORE stacking matrices / building
        # labels — real libraries have far more chunks than any method's cap.
        cap = {"pca": RAW_MAX_POINTS,
               "tsne": dim_reduction.TSNE_MAX_POINTS,
               "umap": dim_reduction.UMAP_MAX_POINTS}[method]
        note = ""
        if len(order) > cap:
            total = len(order)
            keep = VisualisationDialog._subsample_indices(len(order), cap)
            order = [order[i] for i in keep]
            note = (f"subsampled {_fmt_int(cap)} of "
                    f"{_fmt_int(total)} points for this method")
        index_of = {model: {key: i for i, key in
                            enumerate(self._chunk_keys[model])}
                    for model in chosen}
        matrix_parts = []
        for model in chosen:
            rows = [index_of[model][key] for key in order]
            matrix_parts.append(self._vectors[model][rows])
        matrix = np.hstack(matrix_parts).astype(np.float64)
        models_label = " + ".join(_display_name(m) for m in chosen)

        labels = [
            f"{key[2]} — chunk {key[3]} @ {key[4]:.0f}s" for key in order]
        colors = [self._track_color(key[1]) for key in order]

        self._status_label.setText(
            f"Reducing {_fmt_int(matrix.shape[0])} chunks × "
            f"{matrix.shape[1]} dims ({models_label}) with "
            f"{method.upper()}…")
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            coords, axis_labels = dim_reduction.reduce_2d(matrix, method)
        except dim_reduction.DimReductionError as exc:
            QApplication.restoreOverrideCursor()
            self._canvas.clear_plot(str(exc))
            self._status_label.setText(str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        xs, ys = coords[:, 0], coords[:, 1]
        self._canvas.set_plot(xs, ys, axis_labels[0], axis_labels[1],
                              labels, colors, status=note)
        self._status_label.setText(
            f"{method.upper()} of {_fmt_int(matrix.shape[0])} chunks × "
            f"{matrix.shape[1]} dims ({models_label}) — "
            f"X: {axis_labels[0]}, Y: {axis_labels[1]}"
            + (f" — {note}" if note else ""))

    # ---- subsampling --------------------------------------------------------
    @staticmethod
    def _subsample_indices(n: int, cap: int) -> np.ndarray:
        """Fixed-seed random subset — stable across re-plots."""
        rng = np.random.default_rng(0)
        return np.sort(rng.choice(n, size=cap, replace=False))
