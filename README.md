# HoloSmart Music Explorer

A PySide6 desktop application that scans your music folders, catalogs every track in
SQLite, splits each track into overlapping chunks, runs deep-audio models
(**CLAP**, **MERT**, **MERT-330M**, **OpenL3**) and a dependency-light **FFT spectral
statistics** model over the chunks, and lets you browse the per-chunk
embeddings/tags, find **similar tracks**, generate **playlists** from them,
and explore the analysis datasets as **scatter plots**.

## Features

- **Library scan** — indexes `mp3 / wav / flac / ape / aac / m4a` files under any
  folders you add (Desktop suggested by default). The left pane shows a real
  folder tree (files nested under their subfolders) with a live filter box on
  top for partial-match lookup over files and folders, plus Fold all /
  Unfold all buttons. Folders currently being indexed are highlighted in
  yellow. The filename column auto-sizes to the space left over after the
  status area (long names stay reachable via tooltips; drag the splitter
  for more name room), and the tree shows **one status column per analysis
  model** (Status for overall state, then CLAP, MERT, MERT-330M, OpenL3, FFT, …):
  file rows show a ✓ when that model produced embeddings, a ✗ when the track
  was analyzed but the model produced nothing, and folder/root rows show
  live analysis-progress percentage per model (hover a cell for the exact
  `ready/total` counts). **Double-click a file to
  play it in the system-wide music player.** Files currently under
  analysis — and every folder above them, up to the library root — are
  highlighted in yellow for the whole duration, and refreshing the tree
  (e.g. when an analysis run finishes) never moves your selection,
  expansion states or scroll position.
- **Catalog** — path, size, mtime, container, codec, sample rate, channels, bit depth,
  bitrate, duration and tags (title/artist/album/...) stored in SQLite (`data/library.db`).
- **Chunking** — each track is decoded and split into chunks of `chunk_seconds`
  (default **20 s**) with `overlap_percent` overlap (default **50 %**, i.e. a 10 s hop).
- **Model analysis** — every chunk is embedded by the enabled model plugins and the
  vectors are stored per chunk. CLAP additionally produces human-readable zero-shot
  tags; the top tags are aggregated into a per-track description. The **FFT**
  plugin (always available, numpy-only) computes per-band spectral statistics
  (mean amplitude, std, skew, kurtosis, RMS, crest factor for 0–50, 50–300,
  300–1000, 1000–5000, 5000–15000 and 15000 Hz+; dominant frequency, spectral
  centroid, Shannon entropy and Hurst exponent for the whole spectrum) over
  windows of `fft_window_sec` (default 10 s) and stores a 40-dim,
  loudness-invariant feature vector per chunk.
  Unavailable plugins (missing deps) are skipped with a clear note — the app never
  requires all models to be installed.
- **Chunk results under each track** — the *Chunks* tab lists every chunk with its
  time range, tags, and per-model embedding previews (`dim=512 [0.021, -0.113, …]`).
- **Analyze any file** — right-click a file (or folder) in the tree, or use
  *Analyze Selected / Analyze All* in the toolbar. Runs with only the FFT
  model enabled analyze **in parallel** across all but one CPU core (FFT
  is numpy-bound and scales); **every run involving another model is
  strictly sequential** — the torch models (CLAP/MERT/MERT-330M/OpenL3)
  serialize on the GPU anyway, so parallel tracks would only add
  contention and memory spikes. Analysis threads run at a lower
  scheduling priority with
  progress ticks capped at ~10 Hz per track, so the GUI stays responsive
  while analysis churns, and the
  status line tracks each file by name and position in the batch
  (`Analyzing 3/12 — song.mp3 — CLAP: chunk 7/24`), with a determinate
  progress bar and a `Speed: 42 min/h` label (audio-minutes per wall-hour).
  Batch runs (**Analyze All**) **skip already-analyzed files**; the folder
  context-menu **Analyze** walks the selected subtree **recursively and
  incrementally** — already-analyzed tracks are visited too, their chunks
  are kept, and only models that do not cover every chunk yet run, so
  enabling a new model and analyzing a folder fills it in everywhere
  without losing the other models' results. Explicitly analyzing a single
  selected file forces a full re-analysis (escape hatch after changing
  model settings). **WAV files are excluded from batch analysis by
  default** (raw-PCM decode + embed is disproportionately expensive); they
  still scan into the library and stay playable — set `analyze_wav: true`
  in `data/config.json` to include them. In the tree such files are
  **greyed out** (italic name, `–` status dashes) with an explanatory
  tooltip, and they **do not count** toward the per-folder/per-model
  analysis-progress percentages. Your tree selection is never reset while
  a run progresses. *Stop Analysis* aborts at the next batch boundary —
  already-analyzed tracks stay analyzed and the interrupted track is marked
  resumable, so you can continue later with *Analyze Selected / Analyze All*.
  *Clear Analysis* (toolbar or tree context menu) wipes chunks, embeddings,
  tags and descriptions for the selected file or folder — after a
  confirmation — so the tracks can be re-analyzed from scratch.
- **Similar tracks** — two pickers, one search. The **Dataset** picker
  chooses what is compared: `CLAP / MERT / MERT-330M / OpenL3 / FFT` (chunk/centroid
  vectors), any stored dimensionality reduction (`red:<id>`, created in the
  *Reduce…* dialog), or `Description (Ollama)` — text embeddings of the
  track descriptions (via a local [Ollama](https://ollama.com) server).
  The **Algorithm** picker chooses how two tracks are compared:
  - **Centroid** — cosine between the per-track chunk centroids, or
  - **PSVI** (Pareto surface volume intersection) — each track's chunks
    form a Pareto surface (non-dominated by typicality, i.e. cosine to the
    track's own centroid); the seed's surface chunks are matched against
    *every* chunk of each candidate track (mean-of-best-match), or
  - **EMD** (Earth Mover's Distance) — transports the seed's chunk mass
    onto the candidate's chunks (Sinkhorn, cosine ground cost), or
  - **Chamfer** — symmetric mean nearest-chunk distance; EMD and Chamfer
    report the similarity `1 / (1 + distance)`.
  Chunk-level algorithms can also be restricted to a single dataset via the
  Dataset picker. Two **noise filter** checkboxes (HDBSCAN / OPTICS) live on
  the **Chunks tab** and drop junk chunks (silence, fades, transitions,
  spectral flukes) before any comparison. Noise is judged **within each
  song**: every track's chunks are scored with the selected density method
  (HDBSCAN mutual reachability / OPTICS ordering reachability) and a chunk
  is flagged when its score is an outlier relative to its own song (>4x the
  song's median) — silence in an ambient piece is noise, the same spectrum
  in a noise-collage track is not. The run is one-time per (dataset,
  method), cached, fast (~1 minute for a 5k-track library), shown with a
  real progress bar; every algorithm (including Centroid, whose per-track
  centroids get recomputed without the noise chunks) then searches over the
  clean chunk set. With both checkboxes on, a chunk is discarded when
  either detector flags it. The checkboxes are method toggles — checked
  means the detector's outliers are active; they never silently reset
  when the Similar-tab dataset changes. The same checkboxes drive the
  Chunks-tab highlighting: chunks flagged by HDBSCAN are tinted
  **amber**, OPTICS outliers **blue**, and chunks flagged by both
  **pink** (pooling every dataset's stored run of that method), each
  row's tooltip naming the responsible detector(s). **Double-click a
  model cell in the chunks grid** to open a read-out of that chunk's
  full embedding — every dimension with its name (FFT vectors carry
  their band/stat names, e.g. `1–5 kHz RMS`; other models list `dim N`).

  Reductions can age: chunks analyzed (or re-analyzed) after a reduction
  was fitted are missing from its projection. Stale reductions are marked
  in the Dataset picker (`⚠ outdated`) and the Similar tab's **Refresh**
  button re-runs the same reduction over the current chunk vectors **in
  place** — the dataset keeps its `red:<id>` id, name and search
  references, and its per-track centroids are rebuilt.

- **Analysis skip policy** — Analyze Selected / Analyze All skip only
  **fully analyzed** tracks (every enabled model has a vector on every
  chunk). A track analyzed with FFT only is revisited after enabling
  MERT-330M and the incremental pipeline fills in the new model without
  touching stored results. After **every** finished analysis run — no
  matter which models it used (Analyze All, Analyze Selected, single file
  or folder) — the whole FFT dataset in the database is re-standardized
  **per component** (z-score across all stored vectors) and the per-track
  centroids are rebuilt, so band statistics living on different scales
  contribute comparably.

- **Multi-reference similarity** — The Similar tab holds a
  **Reference tracks** list: it is auto-filled with the file selected in
  the tree (and follows the selection until you press *Add selected*,
  which pins the list), and *Add selected / Remove / Clear / Add folder…*
  manage further entries — **Add folder…** adds every track of the folder
  selected in the tree (a selected file's parent folder counts too; capped
  at 400 tracks). References without vectors in the search dataset are
  skipped during the search. With **multiple references** the search With **multiple references** the search
  runs per reference over the untruncated candidate set and combines the
  per-reference match percentages via their **geometric mean** — a track
  must be similar to ALL references to rank high, and the references
  themselves never appear in the results. With a single reference the
  **seed track itself is always the first result row at 100%**, followed
  by the best matches. The results table shows each
  match's full path; **double-click a row to play that file in the
  system-wide music player**. All Ollama embedding models are detected at
  application startup.
- **Dimensionality reduction** — the Similar tab's **Reduce…** button opens
  a dialog that turns one model's chunk vectors into a new searchable
  dataset: PCA (built-in, 2–128 components, reports captured variance) or
  UMAP / t-SNE (optional libraries, with `n_neighbors` / `min_dist` /
  `perplexity` parameters). The fit runs on a background thread with a
  progress bar, stage text and elapsed-time readout; the result appears in
  the Dataset picker immediately and is searched like any model. A
  reduction is a snapshot of the chunks analyzed when it was created —
  re-run it after new analyses to include them. UMAP and t-SNE are
  optional libraries and must be installed into the app's own environment
  (`<app>/\.venv/bin/python -m pip install umap-learn scikit-learn`) —
  a `pip install` in a terminal usually targets the system Python, which
  the app cannot see. Size limits: t-SNE up to 15,000 chunk vectors per
  run, UMAP up to 200,000 (an 84k-chunk library reduces in about a
  minute); the dialog greys out methods a dataset is too large for.
  PCA is built-in and instant at any size.
- **Playlists** — one click turns similarity results into a named playlist,
  exportable as `.m3u`; the **Create .m3u & play** button in the Similar
  tab additionally writes the playlist to `data/playlists/` and opens it in
  the system music player immediately (single click, no dialogs).
- **Visualisation** — the toolbar *Visualisation* button opens a scatter-plot
  window over every analyzed chunk. Pick any **component of any model** for
  the X and Y axes (CLAP, MERT, MERT-330M, OpenL3, FFT — cross-model combinations
  allowed; each point is one chunk, hover for track name, chunk index and
  time), or reduce to 2-D first with **PCA** (built-in, instant), **t-SNE**
  (needs `pip install scikit-learn`) or **UMAP** (needs
  `pip install umap-learn`) over any subset of the models — chunks missing
  one of the selected models' vectors are excluded, slow methods are
  deterministically subsampled, and everything re-plots straight from the
  current database on every click — your axis/component/model selections
  survive the re-plot (clamped to what still exists). Parameter changes
  update the chart **live**: axis components, X/Y models and the PCA
  method/selection re-plot immediately (debounced 250 ms); t-SNE and UMAP
  wait for an explicit Plot click because they take seconds to minutes.
  Large libraries are subsampled to 50 000 points before drawing, so each
  re-plot stays snappy.

## Install & run

```bash
cd /Users/apple/Documents/HOLOSMART
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_app.sh            # or: .venv/bin/python -m app.main
```

Optional extras:

- **AAC/APE decoding & rich probing** — a static `ffmpeg`/`ffprobe` ships in `bin/`
  (auto-detected). Otherwise install one: `brew install ffmpeg`.
- **OpenL3 plugin** — `pip install openl3 tensorflow` (large download).
- **CLAP / MERT plugins** — enabled automatically; weights download from HuggingFace
  on first analysis (`laion/clap-htsat-unfused`, `m-a-p/MERT-v1-95M`) and run on
  Apple Metal (MPS) when available. **MERT-330M** (`m-a-p/MERT-v1-330M`,
  1024-d) is an additional opt-in model — enable it under Settings → Analysis
  models; its ~1.3 GB weights download on first use and its embeddings are
  stored under their own model key, alongside (not replacing) MERT-95M.
  Apple Silicon via MPS when available.
- **Ollama similarity** — start `ollama serve` with at least one embedding model
  (e.g. `ollama pull nomic-embed-text`).

## Configuration

The *Settings…* dialog is organized as a sidebar (General, Analysis
models, CLAP, MERT, MERT-330M, FFT, Ollama) so every page fits on screen.
General holds the chunk seconds, overlap percent and a *Skip files longer
than 20 minutes* checkbox (parallel analysis is automatic: FFT-only runs
use every core but one, all other models run sequentially) (batch runs leave oversized files unanalyzed — they stay
in the library and remain playable; explicitly analyzing a single selected
file ignores this skip); the model pages hold which models to run,
per-model options for **CLAP** (model id, tag top-K, batch size, and a
**candidate-tags editor** — one tag per line — that replaces the built-in
~90-tag list CLAP scores chunks against; *Restore default list* puts the
originals back, and an empty or default list defers to the plugin's built-in
list), **MERT**
(model id, window length/overlap for the MPS long-chunk limit, batch size)
and **FFT** (window length for the spectral statistics), Ollama host +
embedding model (+ *Detect* button), playlist length.
Persisted to `data/config.json`; the library lives in `data/library.db`
(override with `HOLOSMART_DATA_DIR` / `HOLOSMART_DB`). Configs written before
the FFT plugin existed get `fft` enabled exactly once on first load; after
that your own model on/off choices are kept.

## Architecture

```
app/
  config.py               paths + AppConfig (JSON persistence)
  db/                     schema.sql, Database (WAL, per-thread connections), repo (DAO)
  audio/                  decode.py (ffmpeg/soundfile/wave), chunking.py, resample.py
  models/                 base.py (ModelPlugin contract), clap/mert/openl3/fft, registry
  analysis/pipeline.py    decode -> chunk -> plugins -> embeddings/tags -> description
  similarity/             ollama.py (client + embedding-model detection), search.py,
                          pareto.py, chunk_distances.py (EMD / Chamfer)
  playlist/generator.py   playlist creation + .m3u export
  ui/                     main_window, folder_tree, detail_pane, workers,
                          settings_dialog, system_player
  main.py                 entry point
```

Design rules: heavy imports are lazy (the GUI starts instantly without torch);
every thread opens its own SQLite connection; workers talk to the UI only via
Qt signals; missing optional dependencies degrade gracefully.

## Scanning progress & folder permissions

- Scans run in two phases: files are discovered first (status bar shows the total),
  then each file is probed and indexed with a live determinate progress bar and
  `Scanning 42/1200 — filename` updates. Unreadable/corrupt files are reported and
  skipped without aborting the scan. Scanning is fast: files are probed in
  parallel (bounded thread pool), the index uses one batched DB connection per
  scan, and a rescan skips re-probing files whose size+mtime are unchanged
  (a full rescan of an unchanged library is near-instant).
- On macOS, protected folders (Desktop, Documents, Downloads, …) can deny read
  access (`PermissionError`). HoloSmart pre-checks access when you add a folder —
  which also triggers the OS permission prompt on first use — and if any directory
  is denied during a scan it shows a dialog with **Open Privacy & Security**
  (jumps to the right System Settings pane), **Show in Finder** and **Retry**.
  Accessible sub-trees are still scanned even when sibling directories are denied
  (see `app/fs_utils.py`).

## Learning: similarity weights from your own ears

The **Learning** tab lets you teach the app what *you* consider similar.
Mark pairs of tracks that sound alike ("Set A from selection" / "Set B
from selection" + **Add pair**), pick the model (primarily FFT) and press
**Learn weights**. The optimizer solves a small ridge-regularized
quadratic program over the pairs: components on which your similar songs
consistently agree get large weights, components that vary get zero — the
table ranks the components from *most* to *least* similarity-carrying.
The learned weights persist in the database and scale every vector
comparison of that model in the Similar search (weighted cosine for
centroids, weighted distances for EMD/Chamfer/Pareto), so future searches
emphasize the features that match your perception. **Clear learned
weights** removes them again.

## Tests & verification

```bash
.venv/bin/python -m unittest discover -s tests -v   # 341 tests (db/audio/models/pipeline/similarity/UI/perf)
.venv/bin/python scripts/e2e_check.py               # end-to-end: scan -> chunk -> CLAP/MERT -> Ollama -> playlist
```

The E2E script generates real `wav`/`mp3`/`aac` files, analyzes them with the actual
CLAP + MERT models, stores Ollama text embeddings of the model descriptions, runs
similarity search with every method, and exports a playlist.

