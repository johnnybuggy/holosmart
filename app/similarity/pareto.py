"""Pareto-surface selection over chunk embeddings and chunk-level similarity search.

Where :mod:`app.similarity.search` compares whole tracks through their
per-track aggregate vectors, this module drills down to the chunk level:

* every chunk of a track is described by one objective per embedding model —
  how representative the chunk is of the track for that model (cosine of the
  chunk vector against the track's own centroid for that model).  A chunk
  missing a model's embedding scores ``-1.0`` there, so sparse chunks never
  masquerade as representative ones;
* the ``pareto_surface`` of those objective vectors keeps the chunks that are
  not beaten by another chunk in every model at once ("typical chunks");
* ``pareto_similar_tracks`` then compares candidates against the seed's
  surface chunks with a cartesian chunk-level comparison.

Scoring decisions (documented per the feature spec):

* For each model present on both the seed's surface chunks and the
  candidate's chunks, every seed surface chunk is compared against EVERY
  chunk of the candidate track and the best match wins; the per-model value
  is the mean of those best matches over the surface chunks
  (mean-of-best-match per surface chunk).  Pairs with mismatched vector
  dimensions contribute ``0.0``, mirroring :func:`app.similarity.search.cosine`'s
  tolerance.
* The final score is the unweighted mean over the common models (equal
  weights).  A model that none of the seed's *surface* chunks carry is
  dropped from the comparison, and a candidate left without any comparable
  model is skipped.
* Unlike the track-level search, this feature reads only ``chunks`` and
  ``embeddings`` — no ``track_embeddings`` aggregates are needed.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Callable, Iterable, TypeVar

import numpy as np

from app.db import repo

if TYPE_CHECKING:  # static-only: keeps the runtime import cycle-free
    from app.similarity.search import SimilarResult

__all__ = [
    "chunk_objectives",
    "pareto_chunk_ids",
    "pareto_similar_tracks",
    "pareto_surface",
]

#: Objective value reported for a chunk that lacks a model's embedding.
MISSING_MODEL_SCORE = -1.0

K = TypeVar("K")


def _chunk_vectors_by_model(
    conn: sqlite3.Connection, track_id: int,
    noise: set[int] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Chunk embedding vectors of one track as ``{model: {chunk_id: vec}}``.

    ``noise`` is an optional set of chunk ids to skip (noise-filtered
    chunks are never compared).
    """
    by_model: dict[str, dict[int, np.ndarray]] = {}
    for chunk in repo.get_chunks(conn, track_id):
        chunk_id = int(chunk["id"])
        if noise and chunk_id in noise:
            continue
        for row in repo.get_chunk_embeddings(conn, chunk_id):
            by_model.setdefault(row["model"], {})[chunk_id] = row["vec"]
    return by_model


def chunk_objectives(
    conn: sqlite3.Connection, track_id: int
) -> dict[int, list[float]]:
    """Map each chunk id of *track_id* to its per-model objective vector.

    One dimension per model that has embeddings on the track's chunks, the
    models sorted by name so the vector layout is deterministic.  The value
    for model *m* is ``cosine(chunk_vec_m, centroid_m)`` where ``centroid_m``
    is the L2-normalized mean of the track's chunk vectors for *m* — i.e. how
    representative that chunk is of the track under *m*.  A chunk missing the
    embedding for *m* (or lacking embeddings entirely) gets
    :data:`MISSING_MODEL_SCORE` (``-1.0``) in that dimension.  Returns ``{}``
    when the track has no chunks.
    """
    from app.similarity.search import compute_centroid, cosine  # lazy: avoid cycle

    chunks = repo.get_chunks(conn, track_id)
    if not chunks:
        return {}
    by_model = _chunk_vectors_by_model(conn, track_id)
    models = sorted(by_model)
    centroids = {
        model: compute_centroid(list(by_model[model].values())) for model in models
    }
    objectives: dict[int, list[float]] = {}
    for chunk in chunks:
        chunk_id = int(chunk["id"])
        objectives[chunk_id] = [
            cosine(by_model[model][chunk_id], centroids[model])
            if chunk_id in by_model[model] else MISSING_MODEL_SCORE
            for model in models
        ]
    return objectives


def pareto_surface(points: dict[K, list[float]]) -> list[K]:
    """Keys of the non-dominated points, maximizing every objective.

    A point dominates another iff it is ``>=`` in every objective and
    strictly ``>`` in at least one; points with identical objective vectors do
    not dominate each other, so both survive.  Pure numpy, no DB access.
    The result is sorted by key (ints sort numerically); empty input yields
    an empty list.  Ragged objective vectors are zero-padded on the right
    with ``-inf`` so a missing dimension can never look competitive.
    """
    if not points:
        return []
    keys = list(points.keys())
    rows = [np.asarray(points[key], dtype=np.float64).reshape(-1) for key in keys]
    width = max((row.size for row in rows), default=0)
    if width:
        rows = [
            np.pad(row, (0, width - row.size), mode="constant",
                   constant_values=-np.inf)
            for row in rows
        ]
    matrix = np.vstack(rows) if width else np.zeros((len(rows), 0), dtype=np.float64)

    ge = matrix[:, None, :] >= matrix[None, :, :]   # [j, i, d]: j >= i on dim d
    gt = matrix[:, None, :] > matrix[None, :, :]    # [j, i, d]: j >  i on dim d
    dominates = ge.all(axis=2) & gt.any(axis=2)     # j dominates i
    np.fill_diagonal(dominates, False)              # a point never dominates itself
    dominated = dominates.any(axis=0)               # i beaten by some other point
    return sorted(key for key, beaten in zip(keys, dominated) if not beaten)


def pareto_chunk_ids(conn: sqlite3.Connection, track_id: int,
                     noise: set[int] | None = None) -> list[int]:
    """Ids of *track_id*'s chunks that sit on its Pareto surface.

    Ordered by chunk ``idx`` (via :func:`app.db.repo.get_chunks`), so the
    result follows the track's timeline rather than chunk insertion order.
    Noise chunks (``noise`` id set) never reach the surface.
    """
    objectives = chunk_objectives(conn, track_id)
    if noise:
        objectives = {chunk_id: point for chunk_id, point in
                      objectives.items() if int(chunk_id) not in noise}
    surface = set(pareto_surface(objectives))
    if not surface:
        return []
    return [
        int(chunk["id"]) for chunk in repo.get_chunks(conn, track_id)
        if int(chunk["id"]) in surface
    ]


def _dataset_chunk_vectors(
    conn: sqlite3.Connection, track_id: int, dataset: str,
    noise: set[int] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Chunk vectors of one track for a *dataset* as ``{source: {chunk_id: vec}}``.

    ``dataset`` is a model name (raw ``embeddings`` rows) or ``"red:<id>"``
    (stored dimensionality-reduced vectors).  The returned dict maps that
    source id to its ``{chunk_id: vec}`` vectors, so callers can treat a
    reduction exactly like a model.  ``noise`` is an optional set of chunk
    ids to skip (noise-filtered chunks are never compared).
    """
    if dataset.startswith("red:"):
        by_source: dict[str, dict[int, np.ndarray]] = {dataset: {}}
        for row in repo.get_reduced_chunk_embeddings(conn, int(dataset[4:]),
                                                     track_id):
            chunk_id = int(row["chunk_id"])
            if noise and chunk_id in noise:
                continue
            by_source[dataset][chunk_id] = row["vec"]
        return by_source
    by_model: dict[str, dict[int, np.ndarray]] = {}
    for chunk in repo.get_chunks(conn, track_id):
        chunk_id = int(chunk["id"])
        if noise and chunk_id in noise:
            continue
        for row in repo.get_chunk_embeddings(conn, chunk_id):
            if row["model"] != dataset:
                continue
            by_model.setdefault(dataset, {})[chunk_id] = row["vec"]
    return by_model


def _dataset_candidate_ids(conn: sqlite3.Connection, seed_track_id: int,
                           dataset: str) -> list[int]:
    """Other tracks carrying vectors in *dataset*, ascending."""
    if dataset.startswith("red:"):
        ids = set(repo.tracks_with_reduced_chunks(conn, int(dataset[4:])))
    else:
        rows = conn.execute(
            "SELECT DISTINCT c.track_id AS track_id "
            "FROM chunks c JOIN embeddings e ON e.chunk_id = c.id "
            "WHERE e.model = ? ORDER BY c.track_id",
            (dataset,),
        ).fetchall()
        ids = {int(r["track_id"]) for r in rows}
    return sorted(ids - {int(seed_track_id)})


def _dataset_surface_ids(conn: sqlite3.Connection, seed_track_id: int,
                         dataset: str,
                         noise: set[int] | None = None) -> list[int]:
    """Pareto surface of the seed's chunks within one dataset.

    Noise chunks never reach the surface: with *noise* set they are
    excluded from both the typicality objective and the surface itself.

    One dataset means one objective per chunk — its representativeness
    (cosine against the dataset centroid) — so the surface keeps the chunks
    that are not beaten on that single objective (ties survive).  ``[]``
    when the seed has no vectors in the dataset.
    """
    from app.similarity.search import compute_centroid, cosine  # lazy: avoid cycle

    vecs = _dataset_chunk_vectors(conn, seed_track_id, dataset,
                                  noise=noise).get(dataset, {})
    if not vecs:
        return []
    centroid = compute_centroid(list(vecs.values()))
    objectives = {cid: [cosine(vec, centroid)] for cid, vec in vecs.items()}
    return sorted(pareto_surface(objectives))


def pareto_similar_tracks(
    conn: sqlite3.Connection,
    seed_track_id: int,
    limit: int = 20,
    progress_cb: Callable[[str], None] | None = None,
    dataset: str = "auto",
    discard_noise: Iterable[str] = (),
) -> list[SimilarResult]:
    """Tracks most similar to the seed via chunk-level Pareto comparison.

    The seed's Pareto-surface chunks (its most representative chunks) are
    compared — per embedding model — against EVERY chunk of every candidate
    track (best match per surface chunk, averaged, then averaged across the
    common models with equal weights; see the module docstring).  Candidates
    are the tracks that carry an embedding for at least one model found on
    the seed's chunks; the seed itself is excluded.  Results are sorted by
    descending score, then track id, truncated to *limit*, and carry
    ``method="pareto"``.

    *dataset* restricts the comparison to one analysis dataset — a model
    name (``"clap"``) or a stored reduction (``"red:<id>"``); ``"auto"``
    (default) compares across every model the tracks share, as before.

    ``progress_cb`` optionally receives coarse human-readable progress
    messages (e.g. ``"comparing track 12 (3/8)"``) and stays cheap.
    Raises ``RuntimeError`` with a friendly, actionable message when the
    seed has no chunks or no other track shares a comparable model.
    """
    from app.similarity.search import (  # lazy: avoid cycle
        SimilarResult, _validated_noise_methods, cosine)
    from app.similarity.noise_filter import noise_ids_for

    noise_methods = _validated_noise_methods(discard_noise)
    if dataset in ("auto", "", None):
        dataset = None   # compare across every shared model (legacy behaviour)

    # Noise chunks are dropped from every vector set below; missing runs
    # contribute an empty set, i.e. no filtering.
    noise = noise_ids_for(conn, dataset if dataset is not None else "auto",
                          noise_methods) or None

    seed = repo.get_track(conn, seed_track_id)
    if seed is None:
        raise RuntimeError(f"Seed track {seed_track_id} not found in database")
    if not repo.get_chunks(conn, seed_track_id):
        raise RuntimeError(
            f"Seed track {seed_track_id} has no chunks yet — analyze it first "
            "(right-click the track in the tree and choose Analyze).")

    if dataset is None:
        seed_surface = pareto_chunk_ids(conn, seed_track_id, noise=noise)
        seed_vecs = _chunk_vectors_by_model(conn, seed_track_id, noise=noise)
        track_models = repo.get_track_chunk_models(conn)
        seed_sources = set(track_models.get(int(seed_track_id), ()))
        candidate_ids = sorted(
            track_id for track_id, models in track_models.items()
            if track_id != int(seed_track_id) and set(models) & seed_sources
        )
    else:
        seed_surface = _dataset_surface_ids(conn, seed_track_id, dataset,
                                            noise=noise)
        seed_vecs = _dataset_chunk_vectors(conn, seed_track_id, dataset)
        candidate_ids = _dataset_candidate_ids(conn, seed_track_id, dataset)
        if not seed_vecs.get(dataset):
            raise RuntimeError(
                f"Seed track has no '{dataset}' chunk vectors yet — analyze "
                "it first (right-click the track in the tree and choose "
                "Analyze).")
        if noise:
            # a fully-noisy seed cannot be compared at all
            seed_vecs = {source: {chunk_id: vec for chunk_id, vec in vectors
                                  .items() if chunk_id not in noise}
                         for source, vectors in seed_vecs.items()}
            if not any(seed_vecs.values()):
                raise RuntimeError(
                    "The noise filter discarded every chunk of the seed "
                    "track — uncheck a noise filter (or refit it) and "
                    "search again.")

    # Learned component weights (Learning tab) scale every compared vector:
    # chunk vectors enter the weighted space BEFORE the cosine comparison.
    from app.learning.weights import scale_model_vectors

    seed_vecs = scale_model_vectors(conn, seed_vecs)

    scored: list[tuple[int, float]] = []
    total = len(candidate_ids)
    for position, cand_id in enumerate(candidate_ids, start=1):
        if progress_cb is not None:
            progress_cb(f"comparing track {cand_id} ({position}/{total})")
        cand_vecs = (_dataset_chunk_vectors(conn, cand_id, dataset,
                                            noise=noise)
                     if dataset is not None
                     else _chunk_vectors_by_model(conn, cand_id, noise=noise))
        cand_vecs = scale_model_vectors(conn, cand_vecs)
        per_model: list[float] = []
        for model in sorted(set(seed_vecs) & set(cand_vecs)):
            surface_vecs = [
                seed_vecs[model][chunk_id] for chunk_id in seed_surface
                if chunk_id in seed_vecs[model]
            ]
            cand_list = list(cand_vecs[model].values())
            if not surface_vecs or not cand_list:
                continue  # model not comparable at the surface — drop it
            best = [max(cosine(seed_vec, cand_vec) for cand_vec in cand_list)
                    for seed_vec in surface_vecs]
            per_model.append(float(np.mean(best)))
        if per_model:
            scored.append((cand_id, float(np.mean(per_model))))

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
            method="pareto",
        ))
    if not results:
        raise RuntimeError(
            "No other tracks share an embedding model with this track's "
            "chunks yet — analyze more tracks first (Analyze All).")
    return results
