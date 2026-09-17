"""Settings dialog: chunking, models, Ollama, playlist length."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPlainTextEdit, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.models.registry import plugin_info
from app.similarity.ollama import OllamaClient

#: Preferred Ollama embedding models for music descriptions (code models last).
_PREFERRED = (
    "nomic-embed-text", "bge-m3", "mxbai-embed-large", "snowflake-arctic-embed",
    "all-minilm", "embeddinggemma", "qwen3-embedding",
)


def rank_embedding_models(models: list[str]) -> list[str]:
    """Sort detected embedding models: general text models first, code models last."""
    def score(name: str) -> int:
        lowered = name.lower()
        if "code" in lowered:
            return 50
        for i, hint in enumerate(_PREFERRED):
            if hint in lowered:
                return i
        return len(_PREFERRED) + 10
    return sorted(models, key=score)


class SettingsDialog(QDialog):
    """Edits an :class:`app.config.AppConfig` in place (applied on accept)."""

    #: Sidebar entries, in order: (list label, builder method suffix).
    _PAGES = ("General", "Analysis models", "CLAP", "MERT", "MERT-330M",
              "FFT", "Ollama")

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self.setWindowTitle("Settings")
        self.setMinimumSize(720, 540)

        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        outer.addLayout(body, 1)

        # Sidebar navigation: the per-model groups outgrew a single
        # scrolling column, so each area now gets its own page.
        self._category_list = QListWidget()
        self._category_list.setFixedWidth(170)
        self._category_list.setSizePolicy(
            self._category_list.sizePolicy().horizontalPolicy(),
            self._category_list.sizePolicy().verticalPolicy())
        body.addWidget(self._category_list)
        self._pages = QStackedWidget()
        body.addWidget(self._pages, 1)

        builders = (
            self._build_general_page, self._build_models_page,
            self._build_clap_page, self._build_mert_page,
            self._build_mert330_page, self._build_fft_page,
            self._build_ollama_page)
        for label, builder in zip(self._PAGES, builders):
            self._category_list.addItem(label)
            self._pages.addWidget(builder(config))
        self._category_list.currentRowChanged.connect(
            self._pages.setCurrentIndex)
        self._category_list.setCurrentRow(0)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ---- pages ---------------------------------------------------------------
    def _build_general_page(self, config: AppConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        chunk_box = QGroupBox("Chunking")
        form = QFormLayout(chunk_box)
        self._chunk_seconds = QDoubleSpinBox()
        self._chunk_seconds.setRange(1.0, 120.0)
        self._chunk_seconds.setDecimals(1)
        self._chunk_seconds.setSingleStep(1.0)
        self._chunk_seconds.setValue(float(config.chunk_seconds))
        self._overlap = QSpinBox()
        self._overlap.setRange(0, 95)
        self._overlap.setValue(int(config.overlap_percent))
        form.addRow("Chunk length (seconds):", self._chunk_seconds)
        form.addRow("Overlap (%):", self._overlap)
        layout.addWidget(chunk_box)

        perf_box = QGroupBox("Performance")
        perf_form = QFormLayout(perf_box)
        # Parallel analysis is automatic now: FFT-only runs parallelize
        # across every CPU core but one; runs involving any other model are
        # strictly sequential (model inference serializes on the GPU/MPS
        # anyway).  The old manual spinbox is gone — the legacy
        # analysis_parallelism config field stays only for load-compat.
        self._skip_long = QCheckBox("Skip files longer than 20 minutes")
        self._skip_long.setChecked(
            bool(getattr(config, "analysis_skip_long_files", False)))
        self._skip_long.setToolTip(
            "Batch analysis (Analyze All, folder Analyze) skips files longer "
            "than 20 minutes so giant live sets never block a run. The files "
            "stay in the library and remain playable; explicitly analyzing a "
            "single selected file ignores this skip.")
        perf_form.addRow("", self._skip_long)
        layout.addWidget(perf_box)

        playlist_row = QFormLayout()
        self._playlist_length = QSpinBox()
        self._playlist_length.setRange(5, 100)
        self._playlist_length.setValue(int(config.playlist_length))
        playlist_row.addRow("Playlist length:", self._playlist_length)
        layout.addLayout(playlist_row)
        layout.addStretch(1)
        return page

    def _build_models_page(self, config: AppConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        models_box = QGroupBox("Analysis models (checked = run during analysis)")
        models_layout = QVBoxLayout(models_box)
        self._model_checks: dict[str, QCheckBox] = {}
        for info in plugin_info():
            label = f"{info['display_name']} — dim {info['embedding_dim']}"
            if not info["available"]:
                # quiet rows: a short "(unavailable)" marker only — the full
                # dependency error stays reachable via the checkbox tooltip
                label += "  (unavailable)"
            check = QCheckBox(label)
            if info["error"]:
                check.setToolTip(info["error"])
            check.setChecked(info["name"] in config.models)
            check.setEnabled(info["available"])
            self._model_checks[info["name"]] = check
            models_layout.addWidget(check)
        layout.addWidget(models_box)
        layout.addStretch(1)
        return page

    def _build_clap_page(self, config: AppConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        clap_box = QGroupBox("CLAP (audio tagging + embeddings)")
        clap_form = QFormLayout(clap_box)
        self._clap_model_combo = QComboBox()
        self._clap_model_combo.setEditable(True)
        for mid in ("laion/clap-htsat-unfused", "laion/clap-htsat-fused"):
            self._clap_model_combo.addItem(mid)
        if config.clap_model_id not in ("laion/clap-htsat-unfused",
                                        "laion/clap-htsat-fused"):
            self._clap_model_combo.addItem(config.clap_model_id)
        self._clap_model_combo.setCurrentText(config.clap_model_id)
        clap_form.addRow("Model id:", self._clap_model_combo)
        self._clap_top_k = QSpinBox()
        self._clap_top_k.setRange(1, 20)
        self._clap_top_k.setValue(int(config.clap_tag_top_k))
        clap_form.addRow("Tag top K:", self._clap_top_k)
        self._clap_batch = QSpinBox()
        self._clap_batch.setRange(1, 64)
        self._clap_batch.setValue(int(config.clap_batch_size))
        clap_form.addRow("Batch size:", self._clap_batch)
        # Candidate-tag editor: the zero-shot tagger scores the chunks
        # against exactly this list (one tag per line). Empty lines are
        # dropped and duplicates collapse, so users can paste a bulk list.
        tags_layout = QVBoxLayout()
        tags_layout.setContentsMargins(0, 0, 0, 0)
        from app.models.clap_model import CANDIDATE_TAGS
        self._clap_default_tags = tuple(CANDIDATE_TAGS)
        self._clap_tags_edit = QPlainTextEdit()
        self._clap_tags_edit.setPlainText(
            "\n".join(getattr(config, "clap_tags", None) or CANDIDATE_TAGS))
        self._clap_tags_edit.setToolTip(
            "One candidate tag per line — CLAP scores every chunk against "
            "this list and reports the top matches (Settings → top K). "
            "Empty lines are ignored. Changing the list takes effect on the "
            "next analysis run.")
        self._clap_tags_edit.setMaximumHeight(150)
        tags_layout.addWidget(self._clap_tags_edit)
        tag_buttons = QHBoxLayout()
        self._clap_tags_reset = QPushButton("Restore default list")
        self._clap_tags_reset.setToolTip(
            "Replace the text with the built-in ~90-tag candidate list.")
        self._clap_tags_reset.clicked.connect(self._on_reset_clap_tags)
        tag_buttons.addWidget(self._clap_tags_reset)
        tag_buttons.addStretch(1)
        tags_layout.addLayout(tag_buttons)
        clap_form.addRow("Candidate tags:", tags_layout)
        layout.addWidget(clap_box)
        layout.addStretch(1)
        return page

    def _build_mert_page(self, config: AppConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        mert_box = QGroupBox("MERT (audio embeddings)")
        mert_form = QFormLayout(mert_box)
        self._mert_model_combo = QComboBox()
        self._mert_model_combo.setEditable(True)
        self._mert_model_combo.addItem("m-a-p/MERT-v1-95M")
        if config.mert_model_id != "m-a-p/MERT-v1-95M":
            self._mert_model_combo.addItem(config.mert_model_id)
        self._mert_model_combo.setCurrentText(config.mert_model_id)
        mert_form.addRow("Model id:", self._mert_model_combo)
        self._mert_window = QDoubleSpinBox()
        self._mert_window.setRange(1.0, 30.0)
        self._mert_window.setDecimals(1)
        self._mert_window.setSingleStep(1.0)
        self._mert_window.setValue(float(config.mert_window_sec))
        mert_form.addRow("Window length (s):", self._mert_window)
        self._mert_overlap = QDoubleSpinBox()
        self._mert_overlap.setRange(0.0, 5.0)
        self._mert_overlap.setDecimals(1)
        self._mert_overlap.setSingleStep(0.5)
        self._mert_overlap.setValue(float(config.mert_window_overlap_sec))
        mert_form.addRow("Window overlap (s):", self._mert_overlap)
        self._mert_batch = QSpinBox()
        self._mert_batch.setRange(1, 64)
        self._mert_batch.setValue(int(config.mert_batch_size))
        mert_form.addRow("Batch size:", self._mert_batch)
        layout.addWidget(mert_box)
        layout.addStretch(1)
        return page

    def _build_mert330_page(self, config: AppConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        mert330_box = QGroupBox("MERT-330M (large audio embeddings)")
        mert330_form = QFormLayout(mert330_box)
        self._mert330_model_combo = QComboBox()
        self._mert330_model_combo.setEditable(True)
        self._mert330_model_combo.addItem("m-a-p/MERT-v1-330M")
        if config.mert330_model_id != "m-a-p/MERT-v1-330M":
            self._mert330_model_combo.addItem(config.mert330_model_id)
        self._mert330_model_combo.setCurrentText(config.mert330_model_id)
        self._mert330_model_combo.setToolTip(
            "m-a-p/MERT-v1-330M — ~1.3 GB download, 1024-d embeddings, "
            "roughly 4x slower than MERT-95M. Its vectors are stored "
            "under their own model key and coexist with MERT-95M.")
        mert330_form.addRow("Model id:", self._mert330_model_combo)
        self._mert330_window = QDoubleSpinBox()
        self._mert330_window.setRange(1.0, 30.0)
        self._mert330_window.setDecimals(1)
        self._mert330_window.setSingleStep(1.0)
        self._mert330_window.setValue(float(config.mert330_window_sec))
        mert330_form.addRow("Window length (s):", self._mert330_window)
        self._mert330_overlap = QDoubleSpinBox()
        self._mert330_overlap.setRange(0.0, 5.0)
        self._mert330_overlap.setDecimals(1)
        self._mert330_overlap.setSingleStep(0.5)
        self._mert330_overlap.setValue(float(config.mert330_window_overlap_sec))
        mert330_form.addRow("Window overlap (s):", self._mert330_overlap)
        self._mert330_batch = QSpinBox()
        self._mert330_batch.setRange(1, 64)
        self._mert330_batch.setValue(int(config.mert330_batch_size))
        mert330_form.addRow("Batch size:", self._mert330_batch)
        layout.addWidget(mert330_box)
        layout.addStretch(1)
        return page

    def _build_fft_page(self, config: AppConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        fft_box = QGroupBox("FFT (spectral band statistics)")
        fft_form = QFormLayout(fft_box)
        self._fft_window = QDoubleSpinBox()
        self._fft_window.setRange(1.0, 60.0)
        self._fft_window.setDecimals(1)
        self._fft_window.setSingleStep(1.0)
        self._fft_window.setValue(
            float(getattr(config, "fft_window_sec", 10.0)))
        self._fft_window.setToolTip(
            "Each analysis chunk is split into windows of this length before "
            "the FFT; per-window spectra are averaged into one feature vector "
            "per chunk.")
        fft_form.addRow("Window length (s):", self._fft_window)
        layout.addWidget(fft_box)
        layout.addStretch(1)
        return page

    def _build_ollama_page(self, config: AppConfig) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        ollama_box = QGroupBox("Ollama (text-embedding similarity)")
        ollama_form = QFormLayout(ollama_box)
        self._use_ollama = QCheckBox("Use Ollama for text embeddings of descriptions")
        self._use_ollama.setChecked(config.use_ollama)
        ollama_form.addRow(self._use_ollama)
        self._host_edit = QLineEdit(config.ollama_host)
        ollama_form.addRow("Host:", self._host_edit)
        model_row = QHBoxLayout()
        self._emb_model_combo = QComboBox()
        self._emb_model_combo.addItem("(auto — first detected)", None)
        if config.ollama_embedding_model:
            self._emb_model_combo.addItem(config.ollama_embedding_model,
                                          config.ollama_embedding_model)
            self._emb_model_combo.setCurrentIndex(1)
        detect_btn = QPushButton("Detect")
        detect_btn.clicked.connect(self._on_detect)
        model_row.addWidget(self._emb_model_combo, 1)
        model_row.addWidget(detect_btn)
        holder = QWidget()
        holder.setLayout(model_row)
        ollama_form.addRow("Embedding model:", holder)
        self._detect_status = QLabel("")
        ollama_form.addRow("", self._detect_status)
        layout.addWidget(ollama_box)
        layout.addStretch(1)
        return page

    def _on_detect(self) -> None:
        self._detect_status.setText("Detecting…")
        client = OllamaClient(self._host_edit.text().strip() or AppConfig().ollama_host)
        if not client.is_running():
            self._detect_status.setText("Ollama is not reachable at this host.")
            return
        models = rank_embedding_models(client.list_embedding_models())
        if not models:
            self._detect_status.setText("Ollama is running but no embedding models found.")
            return
        current = self._emb_model_combo.currentData()
        self._emb_model_combo.blockSignals(True)
        self._emb_model_combo.clear()
        self._emb_model_combo.addItem("(auto — first detected)", None)
        for name in models:
            self._emb_model_combo.addItem(name, name)
        if current:
            idx = self._emb_model_combo.findData(current)
            self._emb_model_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self._emb_model_combo.blockSignals(False)
        self._detect_status.setText(f"Detected {len(models)} embedding model(s).")

    def apply(self) -> AppConfig:
        """Write dialog values back into the config object (also returned)."""
        self._config.chunk_seconds = float(self._chunk_seconds.value())
        self._config.overlap_percent = float(self._overlap.value())
        self._config.analysis_skip_long_files = self._skip_long.isChecked()
        self._config.models = [name for name, check in self._model_checks.items()
                               if check.isChecked()]
        # CLAP
        self._config.clap_model_id = self._clap_model_combo.currentText().strip() \
            or AppConfig().clap_model_id
        self._config.clap_tag_top_k = int(self._clap_top_k.value())
        self._config.clap_batch_size = int(self._clap_batch.value())
        self._config.clap_tags = self._parse_clap_tags()
        if (not self._config.clap_tags
                or self._config.clap_tags == list(self._clap_default_tags)):
            # Empty (use defaults) or identical to the built-in list: store
            # None so a future plugin update can ship an improved default.
            self._config.clap_tags = None
        # MERT (overlap is clamped silently to stay below the window length)
        self._config.mert_model_id = self._mert_model_combo.currentText().strip() \
            or AppConfig().mert_model_id
        window_sec = float(self._mert_window.value())
        overlap_sec = float(self._mert_overlap.value())
        overlap_sec = max(0.0, min(window_sec - 0.5, overlap_sec))
        self._config.mert_window_sec = window_sec
        self._config.mert_window_overlap_sec = overlap_sec
        self._config.mert_batch_size = int(self._mert_batch.value())
        # MERT-330M (same clamping as MERT)
        self._config.mert330_model_id = \
            self._mert330_model_combo.currentText().strip() \
            or AppConfig().mert330_model_id
        window_sec = float(self._mert330_window.value())
        overlap_sec = float(self._mert330_overlap.value())
        overlap_sec = max(0.0, min(window_sec - 0.5, overlap_sec))
        self._config.mert330_window_sec = window_sec
        self._config.mert330_window_overlap_sec = overlap_sec
        self._config.mert330_batch_size = int(self._mert330_batch.value())
        # FFT
        self._config.fft_window_sec = float(self._fft_window.value())
        self._config.use_ollama = self._use_ollama.isChecked()
        self._config.ollama_host = self._host_edit.text().strip() or AppConfig().ollama_host
        self._config.ollama_embedding_model = self._emb_model_combo.currentData()
        self._config.playlist_length = int(self._playlist_length.value())
        return self._config

    def _parse_clap_tags(self) -> list[str]:
        """One tag per line, trimmed; blanks dropped; duplicates collapse
        (first occurrence wins, order preserved)."""
        seen: set[str] = set()
        tags: list[str] = []
        for raw in self._clap_tags_edit.toPlainText().splitlines():
            tag = raw.strip()
            if tag and tag.casefold() not in seen:
                seen.add(tag.casefold())
                tags.append(tag)
        return tags

    def _on_reset_clap_tags(self) -> None:
        """Restore the built-in candidate list into the text editor."""
        self._clap_tags_edit.setPlainText("\n".join(self._clap_default_tags))
