"""Chunk-set distance metrics: Chamfer distance and Earth Mover's Distance.

Where :mod:`app.similarity.search` compares whole tracks through one
centroid vector per model, this module compares tracks through their FULL
chunk-embedding sets, per model common to both tracks:

* **Chamfer distance** — for every chunk of one track, the distance to the
  NEAREST chunk of the other track; the symmetric mean of both directions.
  Cheap (one cost matrix) and robust when the two tracks have different
  numbers of chunks.
* **Earth Mover's Distance (EMD)** — the minimal total cost of transporting
  the seed's chunk mass onto the candidate's chunks, where every chunk of
  each track carries the uniform mass ``1/n`` and the ground cost between
  two chunks is their cosine distance (``1 - cos``).  Solved with Sinkhorn
  iterations (entropic-regularized optimal transport), which needs only
  numpy and converges in a few hundred cheap matrix operations for the
  small chunk counts a library track produces.

Both operate on cosine distances, so values live in ``[0, 2]``.  The
search in :func:`similar_tracks_chunk_distance` converts the averaged
per-model distance into a similarity score ``1 / (1 + distance)`` so the
result rows sort "best first" like every other method.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

from app.db import repo

if TYPE_CHECKING:  # static-only: keeps the runtime import cycle-free
    from app.similarity.search import SimilarResult

__all__ = [
    "chamfer_distance",
    "cosine_cost_matrix",
    "earth_movers_distance",
    "similar_tracks_chunk_distance",
]

#: Sinkhorn regularization strength (smaller = closer to the exact EMD but
#: slower/less stable; clamped into this range).
EMD_REG_MIN, EMD_REG_MAX = 0.01, 2.0
#: Sinkhorn iteration cap and convergence tolerance on the row potentials.
EMD_MAX_ITER = 500
EMD_TOL = 1e-9


def _as_unit_matrix(vecs) -> np.ndarray:
    """``(n, d)`` float64 matrix of *vecs* with rows L2-normalized.

    Zero-norm rows (e.g. silent-chunk zero vectors) stay zero: their cosine
    cost against anything is ``1.0`` (maximally noncommittal mid-distance).
    Raises ``ValueError`` on an empty input — callers skip empty sets.
    """
    rows = [np.asarray(vec, dtype=np.float64).reshape(-1) for vec in vecs]
    if not rows:
        raise ValueError("chunk distance needs at least one vector per side")
    width = rows[0].shape[0]
    if any(row.shape[0] != width for row in rows):
        raise ValueError("chunk vectors of one track must share one dimension")
    matrix = np.vstack(rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix),
                     where=norms > 0.0)


def cosine_cost_matrix(a, b) -> np.ndarray:
    """Pairwise cosine distances (``1 - cos``, clipped to ``[0, 2]``).

    ``a`` is ``(n, d)``, ``b`` is ``(m, d)``; returns ``(n, m)``.  Inputs
    need not be normalized.  Raises ``ValueError`` on empty inputs or
    mismatched dimensions.
    """
    unit_a = _as_unit_matrix(a)
    unit_b = _as_unit_matrix(b)
    if unit_a.shape[1] != unit_b.shape[1]:
        raise ValueError(
            f"chunk vector dimensions differ: {unit_a.shape[1]} vs "
            f"{unit_b.shape[1]}")
    cost = 1.0 - unit_a @ unit_b.T
    return np.clip(cost, 0.0, 2.0)


def chamfer_distance(a, b) -> float:
    """Symmetric Chamfer distance between two chunk-vector sets.

    ``mean_a(min_b d(a, b))`` and ``mean_b(min_a d(b, a))`` are averaged,
    so the value is independent of which side has more chunks.  Identical
    sets score ``0.0``; disjoint unit vectors (cos ``0``) score ``1.0``.
    """
    cost = cosine_cost_matrix(a, b)
    return float((cost.min(axis=1).mean() + cost.min(axis=0).mean()) / 2.0)


def earth_movers_distance(a, b, reg: float = 0.05) -> float:
    """Entropic-regularized Earth Mover's Distance between two chunk sets.

    Both sides carry uniform mass (each chunk ``1/n`` of its track); the
    ground cost is the pairwise cosine distance.  Solved with Sinkhorn
    iterations on ``K = exp(-cost / reg)``; returns the transported cost
    ``sum(plan * cost)`` (marginals sum to 1, so the value is the mean
    per-mass cost).  Symmetric up to solver noise, deterministic, and
    ``~0`` for identical sets.  ``reg`` is clamped to
    ``[EMD_REG_MIN, EMD_REG_MAX]``.
    """
    cost = cosine_cost_matrix(a, b)
    reg = float(min(max(reg, EMD_REG_MIN), EMD_REG_MAX))
    n, m = cost.shape
    p = np.full(n, 1.0 / n)
    q = np.full(m, 1.0 / m)
    kernel = np.exp(-cost / reg)
    u = np.full(n, 1.0 / n)
    v = np.full(m, 1.0 / m)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(EMD_MAX_ITER):
            u_prev = u
            u = p / (kernel @ v)
            v = q / (kernel.T @ u)
            if np.all(np.isfinite(u)) and np.linalg.norm(
                    u - u_prev, ord=1) < EMD_TOL:
                break
    plan = u[:, None] * kernel * v[None, :]
    if not np.all(np.isfinite(plan)):
        # Degenerate kernel (pathological inputs): fall back to the mean
        # ground cost rather than reporting a broken number.
        return float(cost.mean())
    return float((plan * cost).sum())


def _apply_noise(vecs: dict, noise: set[int]) -> dict:
    """Drop noise chunk ids from a vector container (lists or id maps)."""
    filtered: dict = {}
    for source, vectors in vecs.items():
        if isinstance(vectors, dict):
            kept = {chunk_id: vec for chunk_id, vec in vectors.items()
                    if int(chunk_id) not in noise}
        else:
            # plain lists carry no chunk ids; they were already filtered
            # at load time via _chunk_vectors_by_model(noise=...)
            kept = vectors
        if kept:
            filtered[source] = kept
    return filtered


def similar_tracks_chunk_distance(
    conn: sqlite3.Connection,
    seed_track_id: int,
    method: str = "emd",
    limit: int = 20,
    progress_cb: Callable[[str], None] | None = None,
    dataset: str = "auto",
    discard_noise: Iterable[str] = (),
) -> list["SimilarResult"]:
    """Tracks most similar to the seed by chunk-set distance (EMD/Chamfer).

    For every candidate track that shares at least one embedding model with
    the seed, the two tracks' FULL chunk-vector sets are compared per common
    model (dimension mismatches drop the model), the per-model distances are
    averaged with equal weights, and the result is reported as the
    similarity ``1 / (1 + distance)`` so bigger = closer, matching the rest
    of the UI.  ``method`` is ``"emd"`` or ``"chamfer"`` and rides on
    ``SimilarResult.method``; the seed is excluded; results sort by
    descending similarity, then track id.

    *dataset* restricts the comparison to one analysis dataset — a model
    name (``"clap"``) or a stored reduction (``"red:<id>"``); ``"auto"``
    (default) compares across every shared model, as before.

    Raises ``RuntimeError`` with a friendly, actionable message when the
    seed has no chunks or no other track shares a comparable model.
    """
    from app.similarity.search import (  # lazy: avoid import cycle
        SimilarResult, _chunk_vectors_by_model, _validated_noise_methods)
    from app.similarity.noise_filter import noise_ids_for

    if method not in ("emd", "chamfer"):
        raise ValueError(f"unknown chunk-distance method: {method!r}")
    distance_fn = (earth_movers_distance if method == "emd"
                   else chamfer_distance)
    noise_methods = _validated_noise_methods(discard_noise)

    seed = repo.get_track(conn, seed_track_id)
    if seed is None:
        raise RuntimeError(f"Seed track {seed_track_id} not found in database")
    if not repo.get_chunks(conn, seed_track_id):
        raise RuntimeError(
            f"Seed track {seed_track_id} has no chunks yet — analyze it first "
            "(right-click the track in the tree and choose Analyze).")

    restricted = dataset not in ("auto", "", None)
    if restricted:
        from app.similarity.pareto import _dataset_chunk_vectors, \
            _dataset_candidate_ids

        seed_vecs = _dataset_chunk_vectors(conn, seed_track_id, dataset)
        if not seed_vecs.get(dataset):
            raise RuntimeError(
                f"Seed track has no '{dataset}' chunk vectors yet — analyze "
                "it first (right-click the track in the tree and choose "
                "Analyze).")
        candidate_ids = _dataset_candidate_ids(conn, seed_track_id, dataset)
    else:
        seed_vecs = _chunk_vectors_by_model(conn, seed_track_id)
        track_models = repo.get_track_chunk_models(conn)
        seed_models = set(track_models.get(int(seed_track_id), ()))
        candidate_ids = sorted(
            track_id for track_id, models in track_models.items()
            if track_id != int(seed_track_id) and set(models) & seed_models
        )
    # Noise chunks are dropped from every vector set before comparing;
    # missing noise runs contribute an empty set (no filtering).
    noise = noise_ids_for(conn, dataset, noise_methods) or None
    if noise:
        seed_vecs = _apply_noise(seed_vecs, noise)
        if not any(seed_vecs.values()):
            raise RuntimeError(
                "The noise filter discarded every chunk of the seed "
                "track — uncheck a noise filter (or refit it) and "
                "search again.")
    # Learned component weights scale every compared vector (weighted
    # chunk distances); models without weights pass through untouched.
    from app.learning.weights import scale_model_vectors

    seed_vecs = scale_model_vectors(conn, seed_vecs)

    scored: list[tuple[int, float]] = []
    total = len(candidate_ids)
    for position, cand_id in enumerate(candidate_ids, start=1):
        if progress_cb is not None:
            progress_cb(f"comparing track {cand_id} ({position}/{total})")
        cand_vecs = (_dataset_chunk_vectors(conn, cand_id, dataset)
                     if restricted
                     else _chunk_vectors_by_model(conn, cand_id))
        if noise:
            cand_vecs = _apply_noise(cand_vecs, noise)
        cand_vecs = scale_model_vectors(conn, cand_vecs)
        per_model: list[float] = []
        for model in sorted(set(seed_vecs) & set(cand_vecs)):
            # The "auto" path stores plain vector lists per model, the
            # dataset-restricted path stores {chunk_id: vec} maps.  Only
            # the VALUES carry vectors — iterating the container itself
            # would silently compare chunk ids (every cosine cost 0).
            seed_container = seed_vecs[model]
            cand_container = cand_vecs[model]
            seed_list = list(seed_container.values() if isinstance(
                seed_container, dict) else seed_container)
            cand_list = list(cand_container.values() if isinstance(
                cand_container, dict) else cand_container)
            if not seed_list or not cand_list:
                continue
            if (np.asarray(seed_list[0]).size
                    != np.asarray(cand_list[0]).size):
                continue  # mixed-model dimensions — drop this model
            try:
                per_model.append(distance_fn(seed_list, cand_list))
            except ValueError:  # ragged/dimensionless vectors — skip model
                continue
        if per_model:
            distance = float(np.mean(per_model))
            scored.append((cand_id, 1.0 / (1.0 + distance)))

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    scored = scored[: max(1, int(limit))]

    results: list[SimilarResult] = []
    for cand_id, score in scored:
        track = repo.get_track(conn, cand_id)
        if track is None:
            continue
        results.append(SimilarResult(
            track_id=cand_id,
            path=track["path"],
            filename=track["filename"],
            artist=track["artist"],
            title=track["title"],
            score=score,
            method=method,
        ))
    if not results:
        raise RuntimeError(
            "No other tracks share an embedding model with this track's "
            "chunks yet — analyze more tracks first (Analyze All).")
    return results
