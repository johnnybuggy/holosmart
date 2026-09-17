-- HoloSmart Music Explorer - SQLite schema (single source of truth).
CREATE TABLE IF NOT EXISTS folders (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    path      TEXT NOT NULL UNIQUE,
    added_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id     INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    path          TEXT NOT NULL UNIQUE,
    filename      TEXT NOT NULL,
    extension     TEXT,
    size_bytes    INTEGER,
    mtime         REAL,
    container     TEXT,
    codec         TEXT,
    sample_rate   INTEGER,
    channels      INTEGER,
    bit_depth     INTEGER,
    bitrate_kbps  REAL,
    duration_sec  REAL,
    -- Snapshot of the audio file taken by the LAST analysis run (distinct
    -- from mtime/size_bytes, which the scanner stores): analyze_track uses
    -- it to detect on-disk changes and keep prior models' results when the
    -- file and the chunk plan are unchanged.
    source_mtime  REAL,
    source_size   INTEGER,
    title         TEXT,
    artist        TEXT,
    album         TEXT,
    genre         TEXT,
    year          TEXT,
    track_no      TEXT,
    status        TEXT NOT NULL DEFAULT 'new',   -- new | analyzing | analyzed | error
    status_message TEXT,
    description   TEXT,                          -- aggregated human-readable description from model tags
    last_analyzed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_folder ON tracks(folder_id);
CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);

CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id   INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    start_sec  REAL NOT NULL,
    end_sec    REAL NOT NULL,
    UNIQUE(track_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_chunks_track ON chunks(track_id);

CREATE TABLE IF NOT EXISTS embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id   INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,          -- float32 little-endian bytes
    norm       REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(chunk_id, model)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

CREATE TABLE IF NOT EXISTS chunk_tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id   INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    text       TEXT NOT NULL,
    score      REAL,
    UNIQUE(chunk_id, model, text)
);

-- Per-track aggregate vectors: centroids of chunk embeddings ("clap"/"mert"/"openl3")
-- and text embeddings of the track description ("ollama:<model>").
CREATE TABLE IF NOT EXISTS track_embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id   INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    norm       REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(track_id, model)
);
CREATE INDEX IF NOT EXISTS idx_track_embeddings_model ON track_embeddings(model);

CREATE TABLE IF NOT EXISTS playlists (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    seed_track_id  INTEGER REFERENCES tracks(id) ON DELETE SET NULL,
    method         TEXT
);

CREATE TABLE IF NOT EXISTS playlist_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    track_id    INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    similarity  REAL,
    UNIQUE(playlist_id, position)
);
CREATE INDEX IF NOT EXISTS idx_playlist_items_playlist ON playlist_items(playlist_id);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Dimensionality-reduced chunk vectors: stored "analysis datasets" the
-- Similar tab can search like a model (dataset id "red:<reductions.id>").
-- A reduction is a snapshot over the chunks analyzed at creation time; the
-- per-track centroids are additionally stored in track_embeddings under the
-- model name "red:<id>" so centroid search works unchanged.
CREATE TABLE IF NOT EXISTS reductions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL UNIQUE,
    source_model   TEXT NOT NULL,          -- raw dataset: 'clap', 'mert', ...
    method         TEXT NOT NULL,          -- 'pca' | 'umap' | 'tsne'
    params         TEXT,                   -- JSON of the fit parameters
    n_components   INTEGER NOT NULL,
    n_vectors      INTEGER NOT NULL DEFAULT 0,
    explained_variance TEXT,               -- JSON list of ratios (PCA) or NULL
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reduced_embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    reduction_id  INTEGER NOT NULL REFERENCES reductions(id) ON DELETE CASCADE,
    chunk_id      INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,           -- float32 little-endian bytes
    norm          REAL,
    UNIQUE(reduction_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_reduced_embeddings_chunk
    ON reduced_embeddings(chunk_id);

-- Noise-filter runs: density-based clustering (HDBSCAN / OPTICS) over one
-- dataset's chunk vectors; chunks labelled as cluster noise (-1) are stored
-- so similarity searches can discard them without re-clustering per search.
-- One run per (dataset, method); re-running replaces it.
CREATE TABLE IF NOT EXISTS noise_filters (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset        TEXT NOT NULL,          -- 'fft', 'mert', 'red:3', ...
    method         TEXT NOT NULL,          -- 'hdbscan' | 'optics'
    params         TEXT,                   -- JSON of the clustering parameters
    n_vectors      INTEGER NOT NULL,
    n_noise        INTEGER NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (dataset, method)
);

CREATE TABLE IF NOT EXISTS noise_chunks (
    filter_id      INTEGER NOT NULL REFERENCES noise_filters(id) ON DELETE CASCADE,
    chunk_id       INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    UNIQUE (filter_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_noise_chunks_chunk
    ON noise_chunks(chunk_id);

-- Learning: user-supplied song pairs + learned per-component weights.
-- A pair says "these two tracks sound similar to me"; the weight optimizer
-- searches for the vector components (per model, primarily FFT) whose
-- agreement across the pairs best explains that perceived similarity.
-- The weights persist and scale every vector comparison in the similarity
-- search (weighted cosine / weighted chunk distances).
CREATE TABLE IF NOT EXISTS learning_pairs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_a    INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    track_b    INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (track_a, track_b)
);

CREATE TABLE IF NOT EXISTS learning_weights (
    model           TEXT NOT NULL,           -- dataset the weights apply to
    component_index INTEGER NOT NULL,
    weight          REAL NOT NULL,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (model, component_index)
);
