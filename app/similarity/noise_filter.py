"""Density-based noise filtering for chunk datasets (HDBSCAN / OPTICS).

Some chunks of a track are junk — silence, fades, transitions, spectral
flukes — and they skew every chunk-level comparison.  Noise is judged
WITHIN each song: every track's chunks are clustered separately (HDBSCAN
or OPTICS from scikit-learn) and a chunk is flagged when its density
score is an outlier relative to its OWN song — silence in an ambient
piece is noise; the same spectrum in a noise-collage track is not.
Similarity searches then simply drop the flagged chunks before
comparing, which is cheap — the clustering is a one-time cached run per
(dataset, method).

* ``dataset`` is a model name (``"fft"``) or a stored reduction
  (``"red:3"``); the clustering sees the dataset's vectors, so a
  reduction can be filtered exactly like a model.
* With BOTH methods enabled, a chunk is discarded when EITHER detector
  flags it (union of the noise sets).
* Re-analyzing a track replaces its chunks; the FK cascade drops dead
  noise flags and fresh chunks simply start unfiltered (re-run the
  filter to include them).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from app.db import repo
from app.db.database import Database

__all__ = [
    "METHODS",
    "HDBSCAN_MAX_POINTS",
    "OPTICS_MAX_POINTS",
    "availability",
    "fit_noise_filter",
    "noise_ids_for",
    "filters_for_dataset",
    "filtered_centroids",
]

#: Noise detectors offered to the user.
METHODS = ("hdbscan", "optics")

#: Point caps — one-time runs over many small per-track fits; refuse the
#: truly absurd instead of running for hours.
HDBSCAN_MAX_POINTS = 200_000
OPTICS_MAX_POINTS = 100_000

#: Neighborhood size for the per-track density scores (clamped to the
#: track's chunk count).
DENSITY_NEIGHBORS = 4

#: A chunk is noise when its density score exceeds this multiple of its
#: song's median score.  Real music chunks sit within ~2x of each other;
#: junk (silence, dropouts, spectral flukes) lands an order of magnitude
#: away, so 4x separates them cleanly while flagging ~0.1 % of clean
#: libraries.
NOISE_REACH_FACTOR = 4.0

#: Tracks with fewer chunks than this are never clustered (a 3-chunk
#: song has no meaningful density structure to violate).
MIN_TRACK_CHUNKS = 6


def availability() -> dict[str, str | None]:
    """``{method: missing-dependency message or None}`` for the detectors.

    Both detectors come from scikit-learn, which is an optional
    dependency of the app (PCA/UMAP need nothing here).
    """
    try:
        import sklearn  # noqa: F401
    except Exception as exc:   # ImportError or a broken install
        message = ("Noise filtering needs scikit-learn — install it into "
                   "the app's environment with "
                   "`.venv/bin/python -m pip install scikit-learn`")
        return {method: message for method in METHODS}
    return {method: None for method in METHODS}


def _dataset_vectors_by_track(conn, dataset: str,
                              ) -> dict[int, list[tuple[int, np.ndarray]]]:
    """All chunk vectors of *dataset*, grouped per song.

    Returns ``{track_id: [(chunk_id, vector), ...]}`` with tracks and
    chunks in stable (id / timeline) order.
    """
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    if dataset.startswith("red:"):
        rows = repo.get_reduced_chunk_embeddings(conn, int(dataset[4:]))
        for row in rows:
            grouped.setdefault(int(row["track_id"]), []).append(
                (int(row["chunk_id"]),
                 np.asarray(row["vec"], dtype=np.float64)))
    else:
        rows = conn.execute(
            "SELECT c.track_id AS track_id, c.id AS chunk_id, "
            "e.vector AS vector "
            "FROM embeddings e JOIN chunks c ON c.id = e.chunk_id "
            "WHERE e.model = ? ORDER BY c.track_id, c.idx",
            (dataset,)).fetchall()
        for row in rows:
            grouped.setdefault(int(row["track_id"]), []).append(
                (int(row["chunk_id"]), repo.blob_to_vec(row["vector"])))
    if not grouped:
        raise RuntimeError(
            f"Dataset '{dataset}' has no chunk vectors yet — analyze "
            "tracks first.")
    return dict(sorted(grouped.items()))


def _l2(matrix: np.ndarray) -> np.ndarray:
    """Rows L2-normalized (zero rows stay zero) — the app's cosine space."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix),
                     where=norms > 0)


def _hdbscan_density_scores(x: np.ndarray) -> np.ndarray:
    """HDBSCAN's mutual-reachability density, one score per chunk.

    The score is the mean mutual reachability
    ``max(core_i, core_j, dist(i, j))`` from each chunk to its
    ``DENSITY_NEIGHBORS`` nearest same-song chunks — the exact quantity
    the HDBSCAN hierarchy is built on.  Junk chunks are far from
    everything, so their score spikes.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(x)
    k = min(DENSITY_NEIGHBORS + 1, n - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(x)
    dist, _idx = nn.kneighbors(x)
    core = dist[:, -1]
    mutual = np.maximum(dist[:, 1:], core[:, None])
    return mutual.mean(axis=1)


def _optics_density_scores(x: np.ndarray) -> np.ndarray:
    """OPTICS' ordering reachability, one score per chunk.

    ``reachability_[i]`` is the distance from chunk *i* to its
    predecessor in the OPTICS ordering — the algorithm's own density
    measure.  The first chunk has no predecessor (NaN → never flagged).
    """
    from sklearn.cluster import OPTICS

    optics = OPTICS(min_samples=min(DENSITY_NEIGHBORS, len(x) - 1),
                    n_jobs=1).fit(x)
    reach = optics.reachability_.astype(np.float64).copy()
    reach[0] = np.nan
    return reach


def fit_noise_filter(db_path: Path | str, dataset: str, method: str, *,
                     progress_cb: Callable[[str, float], None] | None = None,
                     ) -> dict:
    """Flag noise chunks WITHIN each song and store them.

    Every track's chunks are clustered separately — a chunk is judged
    against its own song, never against the whole library.  Chunks are
    L2-normalized (the app's cosine space) and scored with the requested
    density method (HDBSCAN mutual reachability or OPTICS ordering
    reachability); a chunk is noise when its score exceeds
    ``NOISE_REACH_FACTOR`` x its song's median score.  Tracks with fewer
    than ``MIN_TRACK_CHUNKS`` chunks are skipped (no flags).

    ``progress_cb`` receives ``(message, fraction)`` with fraction in
    ``[0, 1]``.  Returns ``{"filter_id", "dataset", "method",
    "n_vectors", "n_noise"}``.  Raises ``RuntimeError`` with a friendly,
    actionable message when the dataset is missing/too small/too large
    or the method is unknown.  Deterministic for a given dataset state.
    """
    if method not in METHODS:
        raise ValueError(f"unknown noise-filter method: {method!r}")
    missing = availability().get(method)
    if missing:
        raise RuntimeError(missing)

    def announce(message: str, fraction: float = 0.0) -> None:
        if progress_cb is not None:
            progress_cb(message, fraction)

    db = Database(db_path)
    announce(f"Loading '{dataset}' chunk vectors…", 0.0)
    with db.transaction() as conn:
        by_track = _dataset_vectors_by_track(conn, dataset)
    n_total = sum(len(chunks) for chunks in by_track.values())
    if n_total < 20:
        raise RuntimeError(
            f"'{dataset}' has only {n_total} chunk vectors — noise "
            "filtering needs at least 20 chunk vectors to be meaningful.")
    cap = HDBSCAN_MAX_POINTS if method == "hdbscan" else OPTICS_MAX_POINTS
    if n_total > cap:
        raise RuntimeError(
            f"{method.upper()} is limited to {cap:,} chunk vectors per "
            f"run — '{dataset}' has {n_total:,}.")

    scorer = (_hdbscan_density_scores if method == "hdbscan"
              else _optics_density_scores)
    tracks = [chunks for chunks in by_track.values()
              if len(chunks) >= MIN_TRACK_CHUNKS]
    total_tracks = len(tracks)
    announce(f"Clustering {total_tracks:,} songs with {method.upper()}… "
             "(one-time run — the result is cached and reused by every "
             "search)", 0.02)

    noise_ids: list[int] = []
    for position, chunks in enumerate(tracks, start=1):
        ids = [chunk_id for chunk_id, _vec in chunks]
        matrix = _l2(np.stack([np.asarray(vec, dtype=np.float64)
                               for _cid, vec in chunks]))
        scores = scorer(matrix)
        finite = scores[np.isfinite(scores)]
        if finite.size:
            median = float(np.median(finite))
            if median > 0.0:
                flags = np.where(np.isfinite(scores),
                                 scores > median * NOISE_REACH_FACTOR,
                                 False)
                noise_ids.extend(chunk_id for chunk_id, flagged
                                 in zip(ids, flags) if flagged)
        if position % 25 == 0 or position == total_tracks:
            announce(f"Clustering song {position:,}/{total_tracks:,} "
                     f"with {method.upper()}…",
                     0.02 + 0.95 * position / total_tracks)

    params = ('{"scope": "per-track", "min_samples": %d, '
              '"reach_factor": %g, "min_track_chunks": %d}'
              % (DENSITY_NEIGHBORS, NOISE_REACH_FACTOR, MIN_TRACK_CHUNKS))
    announce("Storing noise flags…", 0.98)
    with db.transaction() as conn:
        filter_id = repo.set_noise_filter_result(
            conn, dataset, method, params, n_total, len(noise_ids),
            noise_ids)
    announce(f"Done — {len(noise_ids):,} of {n_total:,} chunks flagged.",
             1.0)
    return {"filter_id": filter_id, "dataset": dataset, "method": method,
            "n_vectors": n_total, "n_noise": len(noise_ids)}


def filters_for_dataset(conn, dataset: str) -> dict[str, dict]:
    """``{method: summary}`` of the stored noise runs usable on *dataset*.

    ``dataset="auto"`` pools every stored run — noise chunk ids are
    global, so a filter fitted on any dataset is applicable to a
    cross-model comparison.
    """
    summaries: dict[str, dict] = {}
    for row in repo.list_noise_filters(conn):
        row_dataset = str(row["dataset"])
        if dataset != "auto" and row_dataset != dataset:
            continue
        summaries.setdefault(str(row["method"]), {
            "id": int(row["id"]),
            "dataset": row_dataset,
            "n_vectors": int(row["n_vectors"]),
            "n_noise": int(row["n_noise"]),
            "created_at": str(row["created_at"]),
        })
    return summaries


def noise_ids_for(conn, dataset: str, methods: Iterable[str]) -> set[int]:
    """Union of the stored noise chunk ids for the requested *methods*.

    Missing runs contribute nothing, so a caller never fails because a
    filter has not been fitted yet.
    """
    wanted = set(methods)
    if not wanted:
        return set()
    noise: set[int] = set()
    for row in repo.list_noise_filters(conn):
        if str(row["method"]) not in wanted:
            continue
        if dataset != "auto" and str(row["dataset"]) != dataset:
            continue
        noise |= repo.get_noise_chunk_ids(conn, int(row["id"]))
    return noise


def filtered_centroids(conn, dataset: str,
                       noise: set[int]) -> dict[int, np.ndarray]:
    """Per-track centroids over the NON-noise chunks of *dataset*.

    One query groups every chunk vector by track (noise chunks skipped);
    returns ``{track_id: float32 unit centroid}`` for tracks that keep at
    least one chunk.  Used by the centroid algorithm when noise filtering
    is active — the stored per-track centroids include the noise chunks.
    """
    from app.similarity.search import compute_centroid  # lazy: cycle

    by_track: dict[int, list[np.ndarray]] = {}
    if dataset.startswith("red:"):
        for row in repo.get_reduced_chunk_embeddings(conn, int(dataset[4:])):
            if int(row["chunk_id"]) in noise:
                continue
            by_track.setdefault(int(row["track_id"]), []).append(row["vec"])
    else:
        rows = conn.execute(
            "SELECT c.track_id AS track_id, c.id AS chunk_id, "
            "e.vector AS vector "
            "FROM embeddings e JOIN chunks c ON c.id = e.chunk_id "
            "WHERE e.model = ?", (dataset,)).fetchall()
        for row in rows:
            if int(row["chunk_id"]) in noise:
                continue
            by_track.setdefault(int(row["track_id"]), []).append(
                repo.blob_to_vec(row["vector"]))
    centroids: dict[int, np.ndarray] = {}
    for track_id, vectors in by_track.items():
        centroid = compute_centroid(vectors)
        if centroid.size:
            centroids[track_id] = centroid
    return centroids
