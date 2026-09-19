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

import logging

import numpy as np

from app.db import repo
from app.db.database import Database

log = logging.getLogger(__name__)

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

#: Point caps — one-time runs over many small per-track fits (cost scales
#: ~linearly, a 141k-chunk library fits in ~1-2 min per method); refuse
#: the truly absurd instead of running for hours.
HDBSCAN_MAX_POINTS = 1_000_000
OPTICS_MAX_POINTS = 1_000_000

#: A model needs at least this many chunk vectors before the post-run
#: clustering bothers with it (HDBSCAN/OPTICS find nothing meaningful in
#: a handful of points).
MIN_NOISE_VECTORS = 20

#: Neighborhood size for the per-track density scores (clamped to the
#: track's chunk count).
DENSITY_NEIGHBORS = 4

#: A chunk is only noise when the density algorithm leaves it unclustered
#: AND its local reachability is an outlier vs its own song's median
#: (k-NN distance above this multiple).  The cross-check kills the false
#: positives HDBSCAN/OPTICS produce inside degenerate near-uniform songs
#: (sklearn's HDBSCAN happily marks arbitrary points of a perfectly tight
#: blob as -1); real junk (silence, fades, dropouts) fails BOTH tests.
NOISE_REACH_FACTOR = 2.0

#: Tracks with fewer chunks than this are never clustered (a 3-chunk
#: song has no meaningful density structure to violate).
MIN_TRACK_CHUNKS = 6


def _reachability_outliers(x: np.ndarray) -> np.ndarray:
    """``True`` where the k-NN distance exceeds ``NOISE_REACH_FACTOR`` x
    the song's median k-NN distance (junk lands an order of magnitude
    away; texture chunks sit within ~2x of each other)."""
    from sklearn.neighbors import NearestNeighbors

    n = len(x)
    k = max(1, min(DENSITY_NEIGHBORS, n - 1))
    dist, _idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    reach = dist[:, 1:].mean(axis=1)
    median = float(np.median(reach))
    if median <= 0.0:
        return np.zeros(n, dtype=bool)
    return reach > median * NOISE_REACH_FACTOR


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
                              track_ids=None
                              ) -> dict[int, list[tuple[int, np.ndarray]]]:
    """All chunk vectors of *dataset*, grouped per song.

    With *track_ids* only those songs are returned (the incremental
    post-analysis refit).  Returns ``{track_id: [(chunk_id, vector), ...]}``
    with tracks and chunks in stable (id / timeline) order.
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
    if track_ids is not None:
        wanted = {int(t) for t in track_ids}
        grouped = {t: chunks for t, chunks in grouped.items()
                   if t in wanted}
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


def _track_noise_labels(x: np.ndarray, method: str) -> np.ndarray:
    """Density-cluster one song's chunk matrix; ``True`` marks outliers.

    The REAL algorithms from scikit-learn, run per song:

    * ``hdbscan`` — :class:`sklearn.cluster.HDBSCAN` finds the song's
      dense texture cluster(s); points not belonging to any cluster
      (``label == -1``) are the outlier candidates — silence, fades,
      dropouts, spectral flukes.
    * ``optics`` — :class:`sklearn.cluster.OPTICS` ordering, extracted
      DBSCAN-style at a per-song epsilon (2x the median k-NN distance —
      OPTICS' own reachability machinery drives the extraction).

    A candidate only becomes noise when its local reachability is ALSO
    an outlier vs the song's median (see
    :func:`_reachability_outliers`) — the two-factor rule keeps the
    algorithms' degenerate false positives (arbitrary -1s inside a
    perfectly tight blob) out of the results while real junk fails both
    tests by an order of magnitude.
    """
    n = len(x)
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        labels = np.asarray(HDBSCAN(
            min_cluster_size=max(3, min(5, n - 2)), min_samples=1,
            metric="euclidean").fit(x).labels_)
    elif method == "optics":
        from sklearn.cluster import OPTICS, cluster_optics_dbscan

        k = max(1, min(DENSITY_NEIGHBORS, n - 1))
        dist, _idx = _knn_distances(x, k)
        eps = float(np.median(dist[:, 1:].mean(axis=1))) * 2.0
        model = OPTICS(min_samples=max(2, min(3, n - 2)),
                       metric="euclidean").fit(x)
        labels = np.asarray(cluster_optics_dbscan(
            reachability=model.reachability_,
            core_distances=model.core_distances_,
            ordering=model.ordering_, eps=eps))
    else:
        raise ValueError(f"unknown noise-filter method: {method!r}")
    return (labels == -1) & _reachability_outliers(x)


def _knn_distances(x: np.ndarray, k: int):
    """Distances to the k nearest neighbors (column 0 = self, dropped by
    callers via ``[:, 1:]``)."""
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)


def fit_noise_filter(db_path: Path | str, dataset: str, method: str, *,
                     progress_cb: Callable[[str, float], None] | None = None,
                     ) -> dict:
    """Cluster every song's chunks with the requested density algorithm
    and store the outlier (noise) chunks.

    Every track's chunks are clustered separately — a chunk is judged
    against its own song, never against the whole library.  Chunks are
    L2-normalized (the app's cosine space) and clustered with the REAL
    scikit-learn algorithm (HDBSCAN or OPTICS); chunks the algorithm
    leaves unclustered (``label == -1``) are stored as noise.  Tracks
    with fewer than ``MIN_TRACK_CHUNKS`` chunks are skipped (no flags).

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
        flags = _track_noise_labels(matrix, method)
        noise_ids.extend(chunk_id for chunk_id, flagged
                         in zip(ids, flags) if flagged)
        if position % 25 == 0 or position == total_tracks:
            announce(f"Clustering song {position:,}/{total_tracks:,} "
                     f"with {method.upper()}…",
                     0.02 + 0.95 * position / total_tracks)

    params = ('{"scope": "per-track", "algorithm": "%s", '
              '"min_track_chunks": %d}'
              % (method, MIN_TRACK_CHUNKS))
    announce("Storing noise flags…", 0.98)
    signatures = {track_id: (len(chunks), sum(chunk_id for chunk_id, _ in
                                              chunks))
                  for track_id, chunks in by_track.items()
                  if len(chunks) >= MIN_TRACK_CHUNKS}
    with db.transaction() as conn:
        filter_id = repo.set_noise_filter_result(
            conn, dataset, method, params, n_total, len(noise_ids),
            noise_ids)
        # everything clustered here — the incremental bookkeeping catches up
        repo.record_noise_run_tracks(conn, dataset, method, signatures)
    announce(f"Done — {len(noise_ids):,} of {n_total:,} chunks flagged.",
             1.0)
    return {"filter_id": filter_id, "dataset": dataset, "method": method,
            "n_vectors": n_total, "n_noise": len(noise_ids)}


def fit_noise_filter_tracks(db_path: Path | str, dataset: str, method: str,
                            track_ids, *,
                            progress_cb: Callable[[str, float], None]
                            | None = None) -> dict:
    """Re-cluster ONLY the given songs and merge into the stored filter.

    The incremental post-analysis path: a song's noise flags depend only
    on that song's own chunks, so an analysis run that (re-)embedded
    tracks *T* under model *M* only has to recompute the *T* rows of the
    ``(M, method)`` filter — every other song's flags stay untouched and
    the update costs milliseconds per song instead of a full-library
    pass.

    Replaces the previous flags of the given tracks with the fresh ones
    (a re-analyzed song's new chunks cannot keep old flags), creates the
    filter row when none exists yet, and refreshes the stored vector
    count.  Raises the same friendly ``RuntimeError`` messages as
    :func:`fit_noise_filter` (unknown method, missing dependency, dataset
    under ``MIN_NOISE_VECTORS`` vectors).
    """
    if method not in METHODS:
        raise ValueError(f"unknown noise-filter method: {method!r}")
    missing = availability().get(method)
    if missing:
        raise RuntimeError(missing)
    wanted_tracks = sorted({int(t) for t in track_ids})
    if not wanted_tracks:
        return {"filter_id": None, "dataset": dataset, "method": method,
                "n_vectors": 0, "n_noise": 0, "n_refit_tracks": 0}

    def announce(message: str, fraction: float = 0.0) -> None:
        if progress_cb is not None:
            progress_cb(message, fraction)

    db = Database(db_path)
    with db.transaction() as conn:
        try:
            if dataset.startswith("red:"):
                n_total = len(repo.get_reduced_chunk_embeddings(
                    conn, int(dataset[4:])))
            else:
                n_total = int(conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model = ?",
                    (dataset,)).fetchone()[0])
        except Exception as exc:
            raise RuntimeError(f"Cannot count '{dataset}' vectors: {exc}")
    if n_total < MIN_NOISE_VECTORS:
        raise RuntimeError(
            f"'{dataset}' has only {n_total} chunk vectors — noise "
            "filtering needs at least "
            f"{MIN_NOISE_VECTORS} chunk vectors to be meaningful.")
    cap = HDBSCAN_MAX_POINTS if method == "hdbscan" else OPTICS_MAX_POINTS
    if n_total > cap:
        raise RuntimeError(
            f"{method.upper()} is limited to {cap:,} chunk vectors per "
            f"run — '{dataset}' has {n_total:,}.")

    announce(f"Re-clustering {len(wanted_tracks)} song(s) with "
             f"{method.upper()}…", 0.1)
    with db.transaction() as conn:
        by_track = _dataset_vectors_by_track(conn, dataset, wanted_tracks)
    fresh_flags: set[int] = set()
    refit_tracks = 0
    for chunks in by_track.values():
        if len(chunks) < MIN_TRACK_CHUNKS:
            continue          # too few chunks to judge a song's density
        ids = [chunk_id for chunk_id, _vec in chunks]
        matrix = _l2(np.stack([np.asarray(vec, dtype=np.float64)
                               for _cid, vec in chunks]))
        flags = _track_noise_labels(matrix, method)
        fresh_flags.update(chunk_id for chunk_id, flagged
                           in zip(ids, flags) if flagged)
        refit_tracks += 1

    params = ('{"scope": "per-track", "algorithm": "%s", '
              '"min_track_chunks": %d}'
              % (method, MIN_TRACK_CHUNKS))
    signatures = {track_id: (len(chunks), sum(chunk_id for chunk_id, _ in
                                              chunks))
                  for track_id, chunks in by_track.items()
                  if len(chunks) >= MIN_TRACK_CHUNKS}
    with db.transaction() as conn:
        row = repo.get_noise_filter(conn, dataset, method)
        if row is None:
            filter_id = repo.set_noise_filter_result(
                conn, dataset, method, params, n_total, len(fresh_flags),
                sorted(fresh_flags))
        else:
            # old flags of the re-clustered tracks are dropped: their
            # chunks either changed or were re-judged just now
            with_raw = conn.execute(
                "SELECT nc.chunk_id AS chunk_id FROM noise_chunks nc "
                "JOIN chunks c ON c.id = nc.chunk_id "
                "WHERE nc.filter_id = ? AND c.track_id IN "
                f"({', '.join('?' * len(wanted_tracks))})",
                [int(row["id"]), *wanted_tracks]).fetchall()
            stale = {int(r["chunk_id"]) for r in with_raw}
            kept = {int(r["chunk_id"]) for r in conn.execute(
                "SELECT chunk_id FROM noise_chunks WHERE filter_id = ?",
                (int(row["id"]),)).fetchall()} - stale
            merged = kept | fresh_flags
            repo.update_noise_filter_result(conn, int(row["id"]), params,
                                            n_total, sorted(merged))
            filter_id = int(row["id"])
        repo.record_noise_run_tracks(conn, dataset, method, signatures)
    announce(f"Done — {len(fresh_flags):,} of the re-clustered chunks "
             "flagged.", 1.0)
    return {"filter_id": filter_id, "dataset": dataset, "method": method,
            "n_vectors": n_total, "n_noise": len(fresh_flags),
            "n_refit_tracks": refit_tracks}


def fit_noise_filter_pending(db_path: Path | str, datasets, methods=METHODS,
                             *, progress_cb: Callable[[str, float], None]
                             | None = None) -> list[dict]:
    """Cluster ONLY the tracks not yet clustered for (dataset, method).

    The button-driven noise pass: after every analysis run nothing runs
    automatically; when the user asks for outliers, each covered dataset's
    PENDING songs — newly analyzed tracks, and re-analyzed tracks whose
    chunk ids changed (see ``repo.pending_noise_tracks``) — are re-judged
    incrementally and merged into the stored filters.  Datasets below
    ``MIN_NOISE_VECTORS`` vectors (or over their cap) are skipped
    silently.  Returns one summary dict per actually-fitted
    (dataset, method).
    """
    def announce(message: str, fraction: float = 0.0) -> None:
        if progress_cb is not None:
            progress_cb(message, fraction)

    results: list[dict] = []
    dataset_list = sorted({str(d) for d in datasets})
    for d_position, dataset in enumerate(dataset_list):
        db = Database(db_path)
        try:
            with db.transaction() as conn:
                if dataset.startswith("red:"):
                    n_total = len(repo.get_reduced_chunk_embeddings(
                        conn, int(dataset[4:])))
                else:
                    n_total = int(conn.execute(
                        "SELECT COUNT(*) FROM embeddings WHERE model = ?",
                        (dataset,)).fetchone()[0])
                pending = {method: repo.pending_noise_tracks(
                               conn, dataset, method, MIN_TRACK_CHUNKS)
                           for method in methods}
        except Exception as exc:
            log.warning("Noise sweep pre-check for %s skipped: %s",
                        dataset, exc)
            continue
        if n_total < MIN_NOISE_VECTORS:
            continue
        for m_position, method in enumerate(methods):
            track_ids = pending.get(method, [])
            if not track_ids:
                continue
            announce(f"{dataset} / {method.upper()}: {len(track_ids):,} "
                     "new song(s)…",
                     (d_position + m_position / max(1, len(methods)))
                     / max(1, len(dataset_list)))
            result = fit_noise_filter_tracks(db_path, dataset, method,
                                             track_ids,
                                             progress_cb=progress_cb)
            results.append(result)
    return results


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
