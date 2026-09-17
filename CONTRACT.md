# HoloSmart Music Explorer — Implementation Contract (v1)

Single source of truth for all modules. **Read this fully before writing code.**
Files already written by the orchestrator (do NOT modify, implement against them):
`app/config.py`, `app/models/base.py`, `app/db/schema.sql`.

## v1.1 addenda (implemented)

* New model plugin **fft** (`app/models/fft_model.py`, registered last in
  `app.models.registry`): FFT spectral band statistics per chunk — 6 band
  stats x 6 bands (0-50, 50-300, 300-1000, 1000-5000, 5000-15000,
  15000 Hz-Nyquist) + dominant frequency / spectral centroid / Shannon
  entropy / Hurst exponent, one 40-dim loudness-invariant vector per chunk;
  similarity methods `fft` and `auto` include it. Config: `fft_window_sec`
  (default 10 s), one-time auto-enable for legacy configs via
  `models_version`.
* File tree shows one status column per registered plugin (column 0 name,
  column 1 overall status, columns 2.. per-model ✓/✗ and per-folder
  percentage-only progress cells — `ready/total` counts live in the cell
  tooltips); column 0 auto-sizes to fit filenames plus
  their nesting depth.
* `repo.get_track_embedding_counts` / `repo.get_all_track_embedding_counts`
  feed the per-model tree columns; `pipeline.aggregate_description` is
  public for reuse. (The v1.1 per-file tag editor was removed in v1.3 —
  see below; `repo.set_chunk_tags` and `app/ui/tag_editor.py` are gone.)

## v1.3 addenda (implemented)

* **Per-file tag editor removed**: `app/ui/tag_editor.py` deleted,
  `repo.set_chunk_tags` deleted, DetailPane lost the *Edit tags…* button,
  the `tags_edited` signal and `_on_edit_tags`; chunk tags in the Chunks
  tab are view-only. `pipeline.aggregate_description` stays public.
* **CLAP candidate-tag editor (Settings)**: `config.clap_tags: list[str] |
  None = None` (None = built-in `clap_model.CANDIDATE_TAGS`). The Settings
  dialog's CLAP group has a one-tag-per-line `QPlainTextEdit` +
  *Restore default list*; apply() trims, drops blanks, dedupes
  (case-insensitive) and stores `None` when the result is empty or equal
  to the built-in list. `ClapPlugin.apply_config` adopts it as
  `tag_candidates`; `_load()` computes the zero-shot text features from
  exactly that list; changing the list (or model id) while loaded unloads
  the weights so the next analysis recomputes them.
* **Softer activity highlight**: `_SCANNING_BRUSH` renamed to
  `_ACTIVITY_BRUSH = QBrush(QColor(255, 200, 0, 60))` — a semi-transparent
  amber tint readable with white text in dark palettes (solid yellow was
  not).
* **Visualisation window** (`app/ui/visualisation.py`,
  non-modal from the new toolbar *Visualisation* action;
  `MainWindow.open_visualisation`): scatter plots of every analyzed chunk.
  Raw mode: any component of any model per axis (cross-model axes align
  chunks that carry both models). Reduction mode: PCA / t-SNE / UMAP
  (user's choice) over any subset of data-bearing models, chunks missing
  any chosen model excluded.
* **Dimensionality reduction** (`app/analysis/dim_reduction.py`):
  `pca_2d` (numpy SVD, labels carry explained-variance %), `tsne_2d`
  (scikit-learn, optional, `TSNE_MAX_POINTS=15000`), `umap_2d`
  (umap-learn, optional, `UMAP_MAX_POINTS=200000`), `standardize`,
  `availability()` (install hints), `reduce_2d` dispatch,
  `DimReductionError`; fixed seed → deterministic re-plots.
* **Scatter canvas** (`app/ui/scatter_plot.py::ScatterCanvas`): painter-only
  (no matplotlib dependency) — auto-ranged axes with grid/tick labels,
  palette-aware colours (night mode safe), `drawPoints` for large N,
  per-track colour cycling, nearest-point hover tooltips
  (`QToolTip`), `RAW_MAX_POINTS=100_000` deterministic subsampling in the
  dialog with a status-note.
* **Repo**: `repo.get_chunk_embedding_rows(conn, models=None)` — joined
  embeddings + chunk idx/start + track filename/path rows with decoded
  `vec`, feeding the dialog.

## v1.2 addenda (implemented)

* Chunk-set similarity methods in `app/similarity/chunk_distances.py`:
  `chamfer_distance` (symmetric mean nearest-chunk cosine distance) and
  `earth_movers_distance` (Sinkhorn-regularized OT over cosine costs,
  uniform per-chunk mass), plus `similar_tracks_chunk_distance` —
  similarity `1/(1+distance)`, dispatched from `search.similar_tracks` as
  methods `emd` / `chamfer` (Similar-tab combo entries included).
* Playback shortcuts: `app/ui/system_player.py::open_in_system_player`
  (QDesktopServices) plays a track on double-click in the file tree and in
  the Similar results table; the Similar tab's *Create .m3u & play* button
  creates + persists a playlist, exports it via the new
  `playlist/generator.py::default_m3u_path` (`data/playlists/<name>.m3u`)
  and opens it in the system player immediately.
* Settings → Performance checkbox *Skip files longer than 20 minutes*
  (`config.analysis_skip_long_files`): `AnalysisWorker._partition_requested`
  (now returning 3 values + the `skipped_long` signal) drops batch tracks
  longer than `workers.LONG_TRACK_SEC` (20 min); explicit single-file
  re-analysis and unknown durations never skip.
* Tree activity highlight: `FolderTree.set_analyzing_paths` paints tracks
  under analysis — and every ancestor folder up to the library root —
  yellow (one combined pass with the scanning highlight,
  `_apply_highlights`); `MainWindow._analyzing` feeds it per signal.
  `FolderTree.refresh` now preserves the current selection (file OR
  folder), all expansion states and the scroll offset across rebuilds, so
  the end-of-run refresh never resets the user's place.

## v1.4 addenda (implemented)

* **Seed-first results**: `similar_tracks` prepends the seed track itself
  as row one with score `1.0` (100 %) for every algorithm
  (`_prepend_seed`/`_seed_result`); delegated leaf searches
  (`pareto_similar_tracks`, `similar_tracks_chunk_distance`) keep their
  seed-excluded contract — the seed is added at the `similar_tracks` layer,
  so playlists built from results start with the seed too.
* **Dataset/Algorithm split**: `similar_tracks(conn, seed, dataset=,
  algorithm=, limit=, ollama=)` — *dataset* ∈ model name | `red:<id>` |
  `ollama:<model>` | `auto`; *algorithm* ∈ `centroid` | `pareto`/`psvi` |
  `emd` | `chamfer` (ollama datasets force centroid). Legacy `method=`
  kwarg still maps onto the pair via `_split_legacy_method`. Dataset
  validation with friendly errors lives in `_resolve_dataset`. The Similar
  tab now has a **Dataset** picker (models with chunk vectors + stored
  reductions + Ollama description entries; algorithm locked to Centroid
  for Ollama) and an **Algorithm** picker (Centroid / PSVI / EMD /
  Chamfer); `DetailPane.search_requested` is now
  `Signal(int, str, str, int)`; `SimilarSearchWorker` takes dataset +
  algorithm; playlist metadata stores `"algorithm:dataset"`.
* **Dataset-restricted chunk searches**: `pareto_similar_tracks` and
  `similar_tracks_chunk_distance` accept `dataset=`; helpers
  `pareto._dataset_chunk_vectors` / `_dataset_candidate_ids` /
  `_dataset_surface_ids` treat a reduction exactly like a model.
* **Reduction datasets**: new tables `reductions` (name UNIQUE,
  source_model, method, params JSON, n_components, n_vectors,
  explained_variance) and `reduced_embeddings` (UNIQUE(reduction_id,
  chunk_id), FK cascades). Repo: `create_reduction`,
  `set_reduction_result`, `list_reductions`, `get_reduction`,
  `delete_reduction`, `replace_reduced_embeddings`,
  `get_reduced_chunk_embeddings`, `tracks_with_reduced_chunks`,
  `get_chunk_vector_models`. A reduction stores per-track centroids in
  `track_embeddings` under the model name `red:<id>` at creation time, so
  the centroid algorithm works unchanged; it is a snapshot — re-run to
  include newly analyzed tracks.
* **`app/analysis/dim_reduction.py` generalized**: `pca_reduce` /
  `tsne_reduce` / `umap_reduce` (n components + parameters) with
  `fit_reduce(x, method, n_components, *, n_neighbors, min_dist,
  perplexity)` dispatch (info carries PCA explained-variance ratios);
  `pca_2d`/`tsne_2d`/`umap_2d`/`reduce_2d` remain for the Visualisation
  dialog. Caps: t-SNE 15 000 points, UMAP 200 000 — the reduction dialog
  greys out any method whose cap the selected dataset exceeds (tooltip
  explains). The optional libraries must be installed into the app's venv
  (`<venv>/bin/python -m pip install scikit-learn umap-learn`), not the
  system Python. `ReductionWorker` sets a 256 MB QThread stack: numba's
  JIT (UMAP) bus-errors on Qt's ~512 KB default thread stack.
* **ReductionWorker** (`app/ui/workers.py`): background fit + storage with
  `stage`/`progress`/`finished_ok(reduction_id, name, n_vectors)`/`failed`
  signals; friendly errors (not enough vectors, duplicate dataset name,
  missing optional library, point caps).
* **Noise filters (HDBSCAN / OPTICS)**: new tables `noise_filters`
  (UNIQUE(dataset, method), params/n_vectors/n_noise) and `noise_chunks`
  (UNIQUE(filter_id, chunk_id), FK cascades). Repo: `set_noise_filter_result`,
  `get_noise_filter`, `list_noise_filters`, `get_noise_chunk_ids`,
  `delete_noise_filter`; `delete_reduction` cascades the filters of
  `red:<id>` datasets. `app/similarity/noise_filter.py` performs
  **per-song** density outlier detection: each track's chunks are
  L2-normalized (the app's cosine space) and scored with HDBSCAN's
  mutual-reachability density or OPTICS' ordering reachability
  (`DENSITY_NEIGHBORS = 4`); a chunk is noise when its score exceeds
  `NOISE_REACH_FACTOR = 4` x its own song's median score — label-based
  noise (`labels_ < 0`) is not used because tiny point clouds fragment.
  Tracks with fewer than `MIN_TRACK_CHUNKS = 6` chunks are skipped.
  Global caps 200 000 (HDBSCAN) / 100 000 (OPTICS) vectors per run; a
  fit of a 5k-track library takes ~30 s per method, and `progress_cb`
  receives `(message, fraction)` for a real progress bar. Stored
  params: `{"scope": "per-track", ...}`. Runs are one-time and cached.
  `similar_tracks(..., discard_noise=("hdbscan", "optics"))` drops flagged
  chunks before every algorithm — vector loaders
  (`pareto._dataset_chunk_vectors`, `pareto._chunk_vectors_by_model`,
  `search._chunk_vectors_by_model`) accept a `noise` id set, and the
  centroid algorithm recomputes per-track centroids from surviving chunks
  (`noise_filter.filtered_centroids`); a fully-noisy seed raises a friendly
  error. `NoiseFilterWorker` runs the fit off the UI thread (64 MB stack);
  the Similar tab's two checkboxes emit `noise_filter_toggled(dataset,
  method, on)` — checking one without a cached run starts the clustering
  (modal progress dialog with a real 0-100 bar from the worker) and
  re-runs the search when it finishes — and
  `search_requested` is now `Signal(int, str, str, int, object)` carrying
  the active filter names; playlist labels append `+noise`.
* **Multi-reference similarity**: `similar_tracks_multi(conn, seed_ids, …)`
  searches each reference individually over the untruncated candidate set
  (auto dataset resolved once so percentages share a vector space) and
  combines per-candidate percentages via `geometric_mean` (any percentage
  <= 0 floors the combined score; references are excluded from the
  results; a candidate must be comparable to every reference). A single
  reference delegates to `similar_tracks` unchanged (seed first at 100%).
  `DetailPane.search_requested` now carries a LIST of reference ids; the
  Similar tab's `_seed_list` auto-follows the tree selection until
  "Add selected" pins it; `SimilarSearchWorker` accepts one id or a list.
* **Analysis skip policy**: a track counts as "already analyzed" only when
  every enabled, available model has a vector on every chunk
  (`repo.track_model_coverage` + `AnalysisWorker._fully_analyzed`). The
  incremental pipeline fills in newly enabled models without touching
  stored results.
* **FFT per-component normalization**: after EVERY finished analysis run —
  regardless of which models it used (Analyze All, Analyze Selected,
  folder / single-file / context-menu, forced, stopped) —
  `AnalysisWorker._post_run_normalization` calls
  `app.analysis.normalization.normalize_model(db, "fft")`, which
  re-standardizes ALL stored FFT chunk vectors per component (z-score,
  zero-variance components flattened, guard `_STD_FLOOR`) and rebuilds
  the FFT track centroids. Unconditional/database-wise: a cheap
  COUNT pre-check skips libraries without FFT vectors; an already
  standardized dataset rewrites nothing (idempotent). Reductions over FFT
  should be refreshed (values are stale, coverage is not).
* **Learning** (`app/learning/weights.py`): pairs live in `learning_pairs`
  (undirected, UNIQUE), learned weights in `learning_weights` (per model +
  component). `learn_weights` solves min Σ c_j w_j + λ Σ w_j² s.t. Σw=1,
  w≥0 via KKT/water-filling (`c_j` = mean squared pair difference per
  component, λ = mean(c)/2); the stored vector is rescaled to mean 1.
  Weights are applied as `vec * sqrt(w)` BEFORE every similarity algorithm:
  centroid path + noise-aware path (`search.py`), chunk vectors of
  EMD/Chamfer (`chunk_distances.py`) and Pareto (`pareto.py`) —
  `weight_vector_for_dataset` never applies weights to `red:`/`ollama:`
  datasets. UI: Learning tab in `DetailPane` + `LearningWorker`.
* **Folder references**: Similar tab "Add folder…" emits
  `folder_references_requested`; `MainWindow._on_add_folder_references`
  uses `FolderTree.folder_track_ids_for_references()` (selected folder
  subtree, or the parent folder for a file; cap
  `MAX_FOLDER_REFERENCES = 400`) and `DetailPane.add_reference_tracks`
  appends deduplicated. `similar_tracks_multi` skips references without
  dataset vectors and has a vectorized one-fetch fast path for the plain
  centroid algorithm.
* **Visualisation selections survive a reload**: every "Plot" click in
  `VisualisationDialog` re-reads the embeddings from the database, and
  `_populate_model_combos` restores the previous X/Y model + component
  (clamped to what still exists; fallback = first model + defaults when a
  model's data vanished) and keeps the reduction checkboxes' checked
  state (models without data are disabled + unchecked). First population
  applies the defaults (first model, components 0/1, all data-bearing
  models checked).
* **ScatterCanvas caches are keyed on the DATA**: `set_plot` drops the
  data→pixel cache AND the per-colour-group `QPolygonF` cache (they were
  previously only invalidated on widget resize, so a re-plot at an
  unchanged window size re-drew the stale point cloud — labels/ticks
  updated, points never did).  `VisualisationDialog._plot_raw` /
  `_plot_reduced` subsample the chunk list BEFORE building hover labels
  and stacking matrices (deterministic `_subsample_indices`, seed 0), so
  re-plots on 100k+-chunk libraries stay fast.
* **Visualisation live updates**: X/Y model combos, component spins, the
  reduction-method combo and the reduction-model checkboxes schedule a
  debounced (250 ms) auto-replot (`_on_axis_changed` /
  `_on_reduction_changed` → `_auto_plot`).  Programmatic restores (the
  reload path) block signals so they never arm the timer.  Only instant
  methods auto-plot — raw components and PCA; t-SNE / UMAP changes show a
  "press Plot" hint instead (they take seconds to minutes).
* **Noise checkboxes are method toggles**: they are never auto-unchecked
  on dataset switches (that silent reset was a bug). Checked = the
  detector's outliers are active; Chunks-tab highlighting pools EVERY
  dataset's stored run of the method (`noise_ids_for(conn, "auto", ...)`)
  and is independent of the Similar tab's dataset. A freshly checked
  filter with no cached run anywhere clusters the dataset resolved by
  `DetailPane._resolve_noise_run_dataset` (Similar's model/red selection,
  else the displayed track's first vector model, else any vector model).
* **GUI responsiveness during analysis**: FFT-only runs use
  `cpu_count - 1` (one core reserved for the GUI); analysis pool threads
  call `os.nice(5)` once (`_lower_thread_priority`) and every worker
  QThread runs at `QThread.Priority.LowPriority`; per-track chunk ticks
  are throttled to ~10 Hz in `AnalysisWorker.progress_cb` (stop-check
  stays first, coarse phase changes always delivered).
* **Chunk vector read-out**: double-clicking a model cell in the Chunks
  table opens a modal dialog listing every dimension (name + value);
  FFT vectors use `fft_model.feature_names()` (band + statistic names),
  other models use `dim N`. Column→model and row→chunk mappings are
  rebuilt by `_refresh_chunks` (`_chunks_model_columns`,
  `_chunks_chunk_ids`).
* **Incremental re-analysis** (`analyze_track`): the freshly computed
  chunk plan is compared against the stored chunks; when boundaries match
  AND the file's analysis-time snapshot (`tracks.source_mtime`/
  `source_size`, migrated via `database._migrate`) is unchanged, the
  stored chunks — and every model's vectors on them — are KEPT, and
  models that already cover all chunks are skipped with an
  "already analyzed (kept)" note (a skipped text model's stored tags still
  feed the description). A fully-covered run is a success, not an error;
  a mid-batch plugin failure (only partial coverage) still errors.
  `force=True` keeps the chunks but re-embeds every enabled model. Chunk
  parameter changes or a changed file re-chunk destructively as before.
* **Recursive folder analysis**: the folder context-menu Analyze passes
  `skip_analyzed=False` (`AnalysisWorker` visits `analyzed` tracks too);
  the per-track incremental logic above makes that safe and cheap.
  `Analyze All` keeps the skip policy (`skip_analyzed=True` default).
* **Excluded tracks never count in progress**: `FolderTree` receives
  `excluded_extensions` (`config.analysis_excluded_extensions`; `(".wav",)`
  unless `analyze_wav`). Items with `EXCLUDED_ROLE` are greyed out
  (`_apply_excluded_look`: italic grey name, grey `–` in the Status and
  every model column, explanatory tooltips) and `_subtree_status_counts`
  skips them entirely, so folder/root percentages ("N of M files
  analyzed") cover only analyzable files. The look survives
  `update_track_status` updates.
* **WAV analysis exclusion** (default): `AppConfig.analyze_wav = False`
  and `MainWindow._split_wav_tracks` drop `.wav` tracks from batch runs
  (status message reports the exclusion); a forced single-file analysis
  ignores it. WAV files still scan into the library and stay playable.
* **Reduction refresh (rerun) mechanism**: reductions store
  `source_model` + `n_vectors`; staleness = `repo.reduction_coverage`
  (live covered chunk ids < the source dataset's current vector count —
  re-analysis replaces chunk ids, so dangling rows stop counting too).
  Stale reductions are marked `⚠ outdated` in the Dataset picker
  (selection is preserved across repopulation) and the Similar tab's
  *Refresh* button opens `ReductionDialog(..., rerun_of=row)`: prefilled
  and identity-locked (source + name), it runs `ReductionWorker` with
  `replace_reduction_id` — `repo.update_reduction` overwrites the row,
  `replace_reduced_embeddings` swaps the vectors, and the `red:<id>`
  per-track centroids are deleted and rebuilt. The dataset id (and every
  stored reference to it) survives a refresh.
* **Noise filters live on the Chunks tab**: the HDBSCAN/OPTICS checkboxes
  moved from the Similar tab into the Chunks tab's header row (same
  `noise_filter_toggled` wiring and search `discard_noise` semantics);
  checked filters tint flagged chunks' rows — amber `#ffd54f` (HDBSCAN),
  blue `#90caf9` (OPTICS), pink `#f48fb1` (both) — with the responsible
  detector(s) in the row tooltip (`DetailPane.refresh_chunks` re-renders).
* **FFT-only analysis parallelism**: `AnalysisWorker._effective_parallelism`
  returns `os.cpu_count() - 1` (clamped to the workload) ONLY when
  `config.models == {"fft"}`; every other model set returns 1 — strictly
  sequential. The legacy `config.analysis_parallelism` field is kept for
  load-compat but no longer widens any run, and the Settings dialog no
  longer exposes a parallelism control.
* **SettingsDialog sidebar**: one `QListWidget` + `QStackedWidget` with
  pages General / Analysis models / CLAP / MERT / MERT-330M / FFT /
  Ollama; all widget attribute names and `apply()` semantics unchanged.
* **ReductionDialog** (`app/ui/reduction_dialog.py`): source dataset +
  method + parameter widgets (components 2–128; UMAP neighbors/min_dist;
  t-SNE perplexity, components locked to 2), auto-suggested editable
  dataset name, progress bar (busy during the fit with elapsed time),
  stage readout, `reduction_created("red:<id>")` signal. Opened from the
  Similar tab's *Reduce…* button; the picker selects the new dataset when
  it finishes.

## Project layout & conventions

```
/Users/apple/Documents/HOLOSMART/          <- project root
  app/
    __init__.py        config.py          (done)
    db/                database.py repo.py schema.sql (schema done)
    audio/             decode.py chunking.py resample.py
    models/            base.py (done) clap_model.py mert_model.py openl3_model.py registry.py
    analysis/          pipeline.py         (orchestrator writes)
    similarity/        ollama.py search.py
    playlist/          generator.py
    ui/                main_window.py folder_tree.py detail_pane.py workers.py settings_dialog.py
    main.py            entry point: python -m app.main
  tests/               unittest-style tests per module
  data/                library.db, config.json (created at runtime, never commit)
  bin/ffmpeg           static ffmpeg (auto-detected)
  .venv/               Python 3.14 venv — ALL code runs/tests with .venv/bin/python
```

Conventions: Python 3.14, `from __future__ import annotations`, full type hints,
docstrings on public functions, `logging` module (no prints except in `main.py`),
NO GUI-blocking work on the UI thread (all heavy work in QThread workers),
**never hard-fail on missing optional deps** (torch/transformers/tensorflow/openl3/
ffmpeg/Ollama may all be absent — degrade gracefully), English UI text.
Dependencies available in `.venv`: numpy, PySide6, soundfile, mutagen, requests,
torch, transformers. NOT installed: openl3/tensorflow, librosa, pytest (use
unittest), scikit-learn (t-SNE in the Visualisation dialog is optional,
`pip install scikit-learn`), umap-learn (UMAP, `pip install umap-learn`).

## App behavior (requirements)

1. Scan music folders (default suggestion: user's Desktop) for
   `.mp3 .wav .flac .ape .aac .m4a`; left pane shows folders→files tree from DB.
2. Every discovered track stored in SQLite with path, size, mtime, container,
   codec, sample rate, channels, bit depth, bitrate, duration + tags.
3. Analysis decodes each track and splits it into chunks: `chunk_seconds` (default
   20.0) with `overlap_percent` (default 50 → hop = 10 s).
4. Chunks are embedded by enabled model plugins (default list: clap, mert, openl3);
   per-chunk vectors stored in `embeddings`, human-readable tags in `chunk_tags`.
5. Chunk results are grouped under their track in the UI (Chunks tab: one row per
   chunk with its tags and per-model embedding previews).
6. Context menu / toolbar "Analyze" works on any selected file or folder, not only
   during bulk operations.
7. Similar-track search: methods = `ollama:<embedding-model>` over Ollama text
   embeddings of track descriptions, or `clap`/`mert`/`openl3` centroids, or
   `auto` (best available). Ollama embedding models are detected at startup via
   `/api/tags` `capabilities` containing `"embedding"`.
8. Playlists generated from similarity results; exportable as .m3u.
9. Everything persisted in SQLite at `data/library.db` (WAL mode).

## Module APIs (implement exactly)

### app/db/database.py
```python
class Database:
    def __init__(self, db_path: Path | str) -> None
    def connect(self) -> sqlite3.Connection
        # FRESH connection per call (thread-safe pattern): row_factory=sqlite3.Row,
        # PRAGMA foreign_keys=ON, journal_mode=WAL, synchronous=NORMAL.
        # Parent dirs created; schema (schema.sql next to this file) applied idempotently.
    def transaction(self) -> contextlib.AbstractContextManager[sqlite3.Connection]
        # yields a connection; commits on success, rolls back on exception.

def vec_to_blob(vec: np.ndarray) -> bytes      # np.asarray(vec, '<f4').tobytes()
def blob_to_vec(blob: bytes) -> np.ndarray     # np.frombuffer(blob, '<f4').copy()
```

### app/db/repo.py — module-level functions, `conn` first arg, sqlite3.Row out
```python
# folders
def add_folder(conn, path: str) -> int                    # INSERT OR IGNORE -> id
def remove_folder(conn, folder_id: int) -> None           # cascades tracks/chunks/etc
def list_folders(conn) -> list[sqlite3.Row]
def get_folder_by_path(conn, path: str) -> sqlite3.Row | None
# tracks — meta dict keys (all optional): filename, extension, size_bytes, mtime,
# container, codec, sample_rate, channels, bit_depth, bitrate_kbps, duration_sec,
# title, artist, album, genre, year, track_no
def upsert_track(conn, folder_id: int, path: str, meta: dict) -> int
def list_tracks(conn, folder_id: int | None = None) -> list[sqlite3.Row]  # ORDER BY path
def get_track(conn, track_id: int) -> sqlite3.Row | None
def get_track_by_path(conn, path: str) -> sqlite3.Row | None
def set_track_status(conn, track_id: int, status: str, message: str | None = None) -> None
def set_track_description(conn, track_id: int, description: str) -> None
def delete_tracks_missing(conn, folder_id: int, existing_paths: set[str]) -> int  # -> count
# chunks — items: list[(idx:int, start_sec:float, end_sec:float)]
def replace_chunks(conn, track_id: int, items: list[tuple[int, float, float]]) -> list[int]
def get_chunks(conn, track_id: int) -> list[sqlite3.Row]                  # ORDER BY idx
def get_chunks_for_tracks(conn, track_ids: list[int]) -> list[sqlite3.Row]
# embeddings — vec: np.ndarray 1-D float32
def add_chunk_embedding(conn, chunk_id: int, model: str, vec: np.ndarray) -> None
def get_chunk_embeddings(conn, chunk_id: int, model: str | None = None) -> list[sqlite3.Row]
    # rows have extra key 'vec' -> decoded np.ndarray
def set_track_embedding(conn, track_id: int, model: str, vec: np.ndarray) -> None  # upsert (centroids & ollama text emb)
def get_track_embedding(conn, track_id: int, model: str) -> np.ndarray | None
def get_track_embeddings(conn, model: str, track_ids: list[int] | None = None) -> dict[int, np.ndarray]
def tracks_with_embeddings(conn, model: str) -> list[int]
# tags — tags: list[(text:str, score:float)]
def add_chunk_tags(conn, chunk_id: int, model: str, tags: list[tuple[str, float]]) -> None
def get_chunk_tags(conn, chunk_id: int, model: str | None = None) -> list[sqlite3.Row]  # score DESC
def get_track_tags(conn, track_id: int, model: str | None = None) -> list[sqlite3.Row]  # joined w/ chunk idx, score DESC
# playlists — items: list[(track_id:int, similarity:float)]
def create_playlist(conn, name: str, seed_track_id: int | None = None, method: str | None = None) -> int
def add_playlist_items(conn, playlist_id: int, items: list[tuple[int, float]]) -> None  # position = 1..n
def list_playlists(conn) -> list[sqlite3.Row]
def get_playlist(conn, playlist_id: int) -> sqlite3.Row | None
def get_playlist_items(conn, playlist_id: int) -> list[sqlite3.Row]  # joined track cols, ORDER BY position
def delete_playlist(conn, playlist_id: int) -> None
# settings (TEXT values; JSON-encode lists)
def set_setting(conn, key: str, value: str) -> None
def get_setting(conn, key: str, default: str | None = None) -> str | None
def save_ollama_models(conn, models: list[str]) -> None   # settings key 'ollama_embedding_models'
def get_ollama_models(conn) -> list[str]
```

### app/audio/decode.py
```python
def ffmpeg_path() -> str | None   # project-local bin/ffmpeg first, then shutil.which('ffmpeg')
def ffmpeg_available() -> bool
def probe_audio(path: str | Path) -> dict
    # keys: container(str|None), codec(str|None), sample_rate(int|None), channels(int|None),
    # bit_depth(int|None), bitrate_kbps(float|None), duration_sec(float|None). Never raises for
    # readable files; missing values None. Strategy: ffprobe (JSON) if ffmpeg present,
    # else mutagen, else soundfile.info. container from suffix.
def decode_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]
    # -> (float32 samples 1-D if mono else (n, ch), sample_rate). Strategy: ffmpeg pipe
    # '-f f32le -acodec pcm_f32le [-ac 1] [-ar target_sr] -'; else soundfile.read; else wave.
    # Raises RuntimeError with actionable message (mention ffmpeg install) when format
    # unsupported (aac/ape without ffmpeg).
```

### app/audio/resample.py
```python
def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray
    # linear interpolation via np.interp (mono 1-D); passthrough if equal; cheap & dependency-free
```

### app/audio/chunking.py
```python
@dataclass
class Chunk:
    idx: int; start_sec: float; end_sec: float; samples: np.ndarray  # float32 1-D

def chunk_audio(samples: np.ndarray, sr: int, chunk_seconds: float = 20.0,
                overlap_percent: float = 50.0) -> list[Chunk]
    # hop = chunk_seconds * (1 - overlap/100). Final partial chunk: if its length >= 25%
    # of chunk -> keep as shorter final chunk; else extend previous chunk to end of audio.
    # Handles audio shorter than one chunk (single chunk, zero-padded not needed — just shorter).
def format_duration(seconds: float | None) -> str   # "mm:ss" / "hh:mm:ss", None -> "?"
def format_size(nbytes: int | None) -> str          # "1.2 MB" human readable
```

### app/models/* — implement `ModelPlugin` subclasses + registry
Lazy imports only (inside `_load`). `is_available()` via `find_spec` only.
```python
class ClapPlugin(ModelPlugin):    # name='clap', display_name='CLAP', embedding_dim=512,
    # provides_text=True, preferred_sample_rate=48000, requirements=('torch','transformers')
    # _load: transformers ClapModel + ClapProcessor 'laion/clap-htsat-unfused' (local_files_only
    #   fallback ok), device 'mps' if available else 'cpu', eval mode.
    # _embed: processor(audios=chunks, sampling_rate=resampled-to-48000, return_tensors='pt',
    #   padding=True) -> model.get_audio_features -> L2-normalized float32 numpy vectors.
    # _describe: zero-shot audio tagging. CANDIDATE_TAGS module constant (~90 music labels:
    #   genres/genres-moods/instruments e.g. 'rock','pop','jazz','classical','hip hop',
    #   'electronic','ambient','folk','metal','blues','reggae','country','funk','soul',
    #   'disco','techno','house','acoustic','piano','guitar','violin','drums','synthesizer',
    #   'orchestral','vocal','male vocal','female vocal','choir','dance','sad','happy','calm',
    #   'energetic','aggressive','chill','dark','upbeat','lo-fi', ...). Text features computed
    #   once at load; per chunk cosine sim -> softmax(scores*100) -> top_k [(tag, prob)].

class MertPlugin(ModelPlugin):    # name='mert', display_name='MERT', embedding_dim=768,
    # provides_text=False, preferred_sample_rate=24000, requirements=('torch','transformers')
    # _load: Wav2Vec2FeatureExtractor + HubertModel 'm-a-p/MERT-v1-95M', device like CLAP;
    #   after load the instance embedding_dim is refreshed from model.config.hidden_size.
    # _embed: resample to 24000 -> extractor -> model -> last_hidden_state mean over time
    #   -> L2-normalized float32 vectors. Chunks longer than the window are split into
    #   overlapping windows (MPS conv limit) and their embeddings averaged.
    # settings_prefix='mert': apply_config reads <prefix>_model_id/_window_sec/
    #   _window_overlap_sec/_batch_size from the AppConfig.

class Mert330Plugin(MertPlugin):  # name='mert330', display_name='MERT-330M', dim 1024,
    # default_model_id='m-a-p/MERT-v1-330M', settings_prefix='mert330', BATCH_SIZE=4.
    # A separate PLUGIN KEY (not just a model id) so 1024-d vectors never share the
    # 'mert' key with 768-d ones; opt-in via Settings (not auto-enabled).

class OpenL3Plugin(ModelPlugin):  # name='openl3', display_name='OpenL3', embedding_dim=512,
    # provides_text=False, preferred_sample_rate=48000,
    # requirements=('openl3','tensorflow','numpy')
    # is_available additionally returns False loudly (availability_error) — TF/OpenL3 are
    #   optional and NOT installed in this venv; code must still be correct.
    # _load: import openl3; _embed: openl3.get_audio_embedding(chunk, sr, input_repr='mel256',
    #   content_type='music', embedding_size=512, verbose=False) -> mean over frames ->
    #   L2-normalized vector. Resample to 48000 first.

# app/models/registry.py
def get_plugin(name: str) -> ModelPlugin           # KeyError with friendly message listing known names
def list_plugins() -> list[ModelPlugin]            # fresh singleton-per-process instances
def plugin_info() -> list[dict]                    # {'name','display_name','embedding_dim',
                                                   #  'provides_text','available','error','loaded'}
```
Import heavy libs only inside methods. Use `torch.no_grad()` + `model.eval()`.
Batch chunks through the model when memory allows (batch size ~8 for CLAP/MERT).

### app/similarity/ollama.py
```python
class OllamaError(RuntimeError): ...
class OllamaClient:
    def __init__(self, host: str = 'http://127.0.0.1:11434', timeout: float = 30.0) -> None
    def is_running(self) -> bool                                    # GET /api/tags, False on any error
    def list_models(self) -> list[dict]                             # raw entries; raise OllamaError if down
    def list_embedding_models(self) -> list[str]                    # names where capabilities contains
                                                                    # 'embedding'; fallback heuristic:
                                                                    # name contains embed/minilm/bge/mxbai/
                                                                    # arctic; fallback2: probe /api/embed
    def embed(self, text: str | list[str], model: str) -> list[list[float]]
        # POST /api/embed {'model': model, 'input': text}; on 404 retry legacy
        # /api/embeddings {'model': model, 'prompt': t} per item. Raise OllamaError w/ message.
def detect_ollama(host: str) -> tuple[bool, list[str]]  # (running, embedding_model_names)
```
Use `requests`. All calls short-timeout (connect 2s) so startup never hangs.

### app/similarity/search.py
```python
@dataclass
class SimilarResult:
    track_id: int; path: str; filename: str; artist: str | None; title: str | None
    score: float; method: str

def cosine(a: np.ndarray, b: np.ndarray) -> float          # 0.0 if either norm==0
def compute_centroid(chunk_vecs: list[np.ndarray]) -> np.ndarray   # mean, L2-normalized; zeros if empty
def ensure_track_embeddings(conn, track_ids: list[int],
                            ollama: OllamaClient | None, emb_model: str | None) -> None
    # For each track: recompute centroids for every model found in chunk embeddings and
    # store via repo.set_track_embedding; if ollama+emb_model and track.description:
    # embed description -> set_track_embedding(track_id, f'ollama:{emb_model}', vec).
def resolve_method(conn, requested: str, ollama: OllamaClient | None) -> str | None
    # 'auto': prefer 'ollama:<cfg model>' if seed track has one & ollama alive, else first
    # model ('clap','mert','openl3') with >=2 tracks having embeddings; None if nothing works.
def similar_tracks(conn, seed_track_id: int, method: str = 'auto', limit: int = 20,
                   ollama: OllamaClient | None = None) -> list[SimilarResult]
    # Excludes seed; only tracks with an embedding for the resolved method; sorted by
    # cosine desc; result.method = concrete method used. Raises RuntimeError(friendly) when none.
```

### app/playlist/generator.py
```python
def generate_playlist(conn, name: str, seed_track_id: int, limit: int = 15,
                      method: str = 'auto', ollama: OllamaClient | None = None) -> int
    # similar_tracks -> create_playlist + add_playlist_items -> playlist_id
def export_m3u(conn, playlist_id: int, out_path: str | Path) -> str
    # '#EXTM3U' + '#EXTINF:<dur>,<artist - title>' + absolute paths; returns path
```

### app/ui/ + app/main.py (PySide6)
- `app/main.py`: `main()` — QApplication, `MainWindow(config)` show; `if __name__ == '__main__': main()`.
- `MainWindow` (main_window.py): central `QSplitter(horizontal)`:
  - LEFT: `FolderTree` (folder_tree.py, QTreeWidget). Top-level: tracked folders (path as
    text, `folder_id` in Qt.ItemDataRole.UserRole); children: tracks (filename + status
    icon/annotation, `track_id` in UserRole, tooltip = full path). Populate from repo.
    Methods: `refresh()`, `selected_track_id() -> int | None`,
    `selected_folder_id() -> int | None`, signal `track_selected(int)`.
    Context menu: Analyze this file, Find similar, Reveal in Finder.
  - RIGHT: `DetailPane` (detail_pane.py, QTabWidget):
    - "Overview": track metadata table (all DB cols, human formatted) + status label +
      description (QTextEdit readonly).
    - "Chunks": table rows = chunks of selected track: idx, start–end, tags per
      text-model ("clap: pop 0.91, rock 0.05"), one column per embedding model showing
      `dim=512 [0.021, -0.113, 0.442, ...]` (first 4 values). Chunk outputs are thus
      grouped under the track. Refreshed on track_selected and after analysis.
    - "Similar": controls row (method QComboBox: Auto/Ollama model names/CLAP/MERT/OpenL3;
      limit QSpinBox 5–100 default 15; Search button), results table (filename, artist,
      title, score 0–100%, method), button "Create playlist from results".
    - "Playlists": QListWidget of playlists; table of items (position, filename, artist,
      similarity); buttons "Export .m3u…", "Delete playlist".
  - Toolbar: "Add Folder…" "Remove Folder" | "Rescan" | "Analyze Selected" "Analyze All" | "Settings…".
  - Status bar: messages + QProgressBar (indeterminate while scanning/analyzing).
- `SettingsDialog` (settings_dialog.py): chunk_seconds (QDoubleSpinBox 1–120, 1 decimal),
  overlap_percent (QSpinBox 0–95), model checkboxes generated from registry.plugin_info()
  with availability shown ("(unavailable: missing deps)"), use_ollama QCheckBox,
  ollama_host QLineEdit, embedding-model QComboBox + "Detect" button (fills from
  OllamaClient), playlist_length QSpinBox. OK/Cancel; saves AppConfig and DB settings.
- `workers.py` — QThread workers (signals with int/str payloads only):
```python
class ScanWorker(QThread):   # __init__(db_path, folder_paths: list[str])
    folder_scanned = Signal(str, int)      # path, tracks_found
    track_upserted = Signal(int, str)      # track_id, path
    finished_scan = Signal(int, int, int)  # added, updated, removed
    failed = Signal(str)
class AnalysisWorker(QThread):  # __init__(db_path, config: AppConfig, track_ids: list[int])
    track_started = Signal(int, str)       # track_id, path
    track_finished = Signal(int, bool, str)  # track_id, ok, message
    all_finished = Signal()
    failed = Signal(str)
    # Internally calls analysis.pipeline.analyze_track(db, track_id, config) per track
```
MainWindow on startup: open Database, refresh tree, background-detect Ollama
(QThread or QThreadPool) -> save_ollama_models + status bar note + enable Similar tab.
Analysis runs enabled plugins sequentially per track; unavailable plugin = skipped with
warning in status_message (NOT an error) as long as at least one plugin ran.

## Thread-safety rules
- Every thread creates its own `Database.connect()` connection; never share
  connections across threads.
- UI updates only via signals; workers never touch widgets.

## Tests (unittest, runnable headless: `QT_QPA_PLATFORM=offscreen`)
- tests/test_db.py — in-memory-ish temp file DB: schema init, track upsert, chunks
  replace, embeddings roundtrip blob<->vec, tags, playlist, settings.
- tests/test_audio.py — chunking counts/hops/overlap on synthetic sine (5 s, 23 s),
  decode on generated WAV (soundfile write), probe on that wav, resample length checks.
- tests/test_models.py — registry lists clap/mert/openl3; plugin_info() keys; available()
  False does not raise; embed/describe skipped unless truly available.
- tests/test_similarity.py — cosine, centroid, ensure_track_embeddings + similar_tracks
  with vectors injected via repo; playlist generation; ollama client only against real
  server (skip if not running).
- tests/test_ui.py — MainWindow constructs offscreen with temp DB; FolderTree.refresh;
  SettingsDialog constructs.

Run: `cd /Users/apple/Documents/HOLOSMART && .venv/bin/python -m unittest discover -s tests -v`
