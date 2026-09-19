"""Similarity search over stored track embeddings."""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.db import repo
from app.similarity.ollama import OllamaClient, OllamaError

log = logging.getLogger(__name__)

#: Audio-model methods tried by ``auto``, in preference order.
_MODEL_ORDER = ("clap", "mert", "mert330", "m2dclap", "muq", "muqlan",
                "lpmc", "qwen2audio", "openl3", "fft")

__all__ = [
    "SimilarResult",
    "cosine",
    "compute_centroid",
    "ensure_track_embeddings",
    "resolve_method",
    "similar_tracks",
]


@dataclass
class SimilarResult:
    """One row of a similar-track search."""

    track_id: int
    path: str
    filename: str
    artist: str | None
    title: str | None
    score: float
    method: str


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, clipped to ``[-1.0, 1.0]``.

    Returns ``0.0`` when either vector has zero norm or the shapes are not
    comparable (scoring must never hard-fail on mixed-model vectors).
    """
    vec_a = np.asarray(a, dtype=np.float64).reshape(-1)
    vec_b = np.asarray(b, dtype=np.float64).reshape(-1)
    if vec_a.size == 0 or vec_b.size == 0 or vec_a.shape != vec_b.shape:
        return 0.0
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, score))


def compute_centroid(chunk_vecs: list[np.ndarray]) -> np.ndarray:
    """Mean of the chunk vectors, L2-normalized (float32).

    Returns a zero-length array for an empty list and a zero vector when the
    mean collapses to zero norm.
    """
    if not chunk_vecs:
        return np.zeros(0, dtype=np.float32)
    matrix = np.asarray(
        [np.asarray(vec, dtype=np.float64).reshape(-1) for vec in chunk_vecs],
        dtype=np.float64,
    )
    centroid = matrix.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros(centroid.shape[0], dtype=np.float32)
    return (centroid / norm).astype(np.float32)


def _chunk_vectors_by_model(conn: sqlite3.Connection, track_id: int,
                            noise: set[int] | None = None,
                            ) -> dict[str, list[np.ndarray]]:
    """Chunk embedding vectors of one track, grouped by model name.

    ``noise`` is an optional set of chunk ids to skip (noise-filtered
    chunks are never compared).
    """
    by_model: dict[str, list[np.ndarray]] = {}
    for chunk in repo.get_chunks(conn, track_id):
        if noise and int(chunk["id"]) in noise:
            continue
        for row in repo.get_chunk_embeddings(conn, chunk["id"]):
            by_model.setdefault(row["model"], []).append(row["vec"])
    return by_model


def ensure_track_embeddings(
    conn: sqlite3.Connection,
    track_ids: list[int],
    ollama: OllamaClient | None = None,
    emb_model: str | None = None,
) -> None:
    """(Re)compute track-level vectors for the given tracks.

    Always stores chunk centroids for every model found on the track's chunks
    (via :func:`app.db.repo.set_track_embedding`, overwriting stale values);
    additionally stores an Ollama text embedding of the track description
    under the model name ``f'ollama:{emb_model}'`` when client+model are
    provided and the track has a description. Ollama failures are logged and
    skipped so one unreachable server never aborts a batch.
    """
    for track_id in track_ids:
        track = repo.get_track(conn, track_id)
        if track is None:
            continue
        by_model = _chunk_vectors_by_model(conn, track_id)
        for model, vectors in by_model.items():
            if not vectors:
                continue
            centroid = compute_centroid(vectors)
            if centroid.size == 0:
                continue
            repo.set_track_embedding(conn, track_id, model, centroid)
        if ollama is not None and emb_model and track["description"]:
            try:
                vectors = ollama.embed(track["description"], emb_model)
                if vectors:
                    repo.set_track_embedding(
                        conn, track_id, f"ollama:{emb_model}",
                        np.asarray(vectors[0], dtype=np.float32))
            except OllamaError as exc:
                log.warning("Ollama text embedding failed for track %s: %s",
                            track_id, exc)
            except Exception as exc:  # defensive: never abort the batch
                log.warning("Unexpected error embedding description of track %s: %s",
                            track_id, exc)


def _seed_ollama_methods(conn: sqlite3.Connection, seed_track_id: int) -> list[str]:
    """``ollama:<model>`` methods the seed track already has embeddings for."""
    rows = conn.execute(
        "SELECT model FROM track_embeddings WHERE track_id = ? AND model LIKE 'ollama:%' "
        "ORDER BY model",
        (seed_track_id,),
    ).fetchall()
    return [str(row["model"]) for row in rows]


def _configured_ollama_model() -> str | None:
    """Ollama embedding model configured in ``AppConfig``, if any."""
    try:
        from app.config import AppConfig  # local import keeps module import cheap

        model = AppConfig.load().ollama_embedding_model
    except Exception:  # config problems must never break similarity search
        return None
    return model or None


def _ollama_candidates(conn: sqlite3.Connection, seed_track_id: int | None) -> list[str]:
    """Usable ``ollama:<model>`` methods (each with >= 2 tracks holding vectors)."""
    methods: list[str] = []
    if seed_track_id is not None:
        methods.extend(_seed_ollama_methods(conn, seed_track_id))
    else:
        configured = _configured_ollama_model()
        if configured:
            methods.append(f"ollama:{configured}")
        try:
            methods.extend(
                f"ollama:{name}" for name in repo.get_ollama_models(conn)
                if name and f"ollama:{name}" not in methods
            )
        except Exception as exc:  # defensive: settings table may be minimal
            log.debug("get_ollama_models unavailable: %s", exc)
    seen: set[str] = set()
    usable: list[str] = []
    for method in methods:
        if method in seen:
            continue
        seen.add(method)
        if len(repo.tracks_with_embeddings(conn, method)) >= 2:
            usable.append(method)
    return usable


def resolve_method(conn: sqlite3.Connection, requested: str,
                   ollama: OllamaClient | None,
                   seed_track_id: int | None = None) -> str | None:
    """Pick a concrete similarity method; ``None`` when nothing is usable.

    Explicit (non-``auto``) requests pass through unchanged. ``auto`` prefers
    an ``ollama:<model>`` text-embedding method while the Ollama server is
    alive — one the seed already has when ``seed_track_id`` is given, else the
    configured/detected embedding model — then falls back to the first model
    among ``clap``/``mert``/``openl3``/``fft`` for which at least two tracks hold
    embeddings (the seed plus at least one comparable track, when known).
    """
    requested = (requested or "auto").strip() or "auto"
    if requested != "auto":
        return requested

    try:
        alive = ollama.is_running() if ollama is not None else False
    except Exception:  # defensive: server probing must not break resolution
        alive = False
    if alive:
        for method in _ollama_candidates(conn, seed_track_id):
            return method

    for model in _MODEL_ORDER:
        ids = set(repo.tracks_with_embeddings(conn, model))
        if seed_track_id is not None:
            if seed_track_id in ids and len(ids) >= 2:
                return model
        elif len(ids) >= 2:
            return model
    return None


def _seed_result(conn: sqlite3.Connection, seed_track_id: int,
                 method: str) -> SimilarResult | None:
    """The seed track itself as a 100% result row (or None if gone)."""
    track = repo.get_track(conn, seed_track_id)
    if track is None:
        return None
    return SimilarResult(
        track_id=seed_track_id,
        path=track["path"],
        filename=track["filename"],
        artist=track["artist"],
        title=track["title"],
        score=1.0,
        method=method,
    )


def _prepend_seed(conn: sqlite3.Connection, seed_track_id: int, method: str,
                  results: list[SimilarResult]) -> list[SimilarResult]:
    """Put the seed itself first with a 100% score.

    The user searched *from* this track — it is always shown as row one at
    100%, followed by the best matches.  Duplicates are impossible because
    every search excludes the seed from its candidate set.
    """
    seed_row = _seed_result(conn, seed_track_id, method)
    if seed_row is None:
        return results
    return [seed_row] + [row for row in results
                         if row.track_id != seed_track_id]


def similar_tracks(conn: sqlite3.Connection, seed_track_id: int,
                   dataset: str = "auto", algorithm: str = "centroid",
                   limit: int = 20, ollama: OllamaClient | None = None,
                   method: str | None = None,
                   discard_noise: "Iterable[str]" = ()) -> list[SimilarResult]:
    """Tracks most similar to the seed, best first — the seed itself is
    always returned as the first row with a 100% score.

    *dataset* selects the analysis dataset: a model name (``"clap"``,
    ``"mert"``, ``"mert330"``, ``"openl3"``, ``"fft"`` — centroid comparison
    over that
    model's per-track chunk centroids), a stored dimensionality reduction
    (``"red:<id>"``), an Ollama description dataset (``"ollama:<model>"``)
    or ``"auto"`` (legacy: best available dataset).  *algorithm* selects how
    two tracks are compared: ``"centroid"``, ``"pareto"``/``"psvi"`` (Pareto
    surface volume intersection), ``"emd"`` or ``"chamfer"``.  Ollama
    description datasets always compare by centroid.

    *discard_noise* optionally names stored noise filters
    (``"hdbscan"``/``"optics"`` — see :mod:`app.similarity.noise_filter`);
    chunks those runs flagged as noise are dropped before any comparison.
    With several methods named, a chunk is discarded when ANY of them
    flags it.  Missing runs contribute nothing.

    ``SimilarResult.method`` carries the concrete dataset for centroid rows
    (e.g. ``"clap"``, ``"red:3"``, ``"ollama:nomic-embed-text"``) or the
    algorithm name for chunk-level rows (``"pareto"`` / ``"emd"`` /
    ``"chamfer"``).  Raises ``RuntimeError`` with a friendly, actionable
    message when nothing can be compared.

    ``method`` (deprecated) keeps old callers working: ``method="clap"`` is
    ``dataset="clap", algorithm="centroid"``, ``method="pareto"`` is
    ``algorithm="psvi"`` with dataset auto, etc.
    """
    seed = repo.get_track(conn, seed_track_id)
    if seed is None:
        raise RuntimeError(f"Seed track {seed_track_id} not found in database")

    # ---- legacy single-string methods --------------------------------------
    if method is not None:
        dataset, algorithm = _split_legacy_method(method)

    algorithm = (algorithm or "centroid").strip().lower()
    if algorithm in ("psvi", "pareto"):
        algorithm = "pareto"
    if algorithm not in ("centroid", "pareto", "emd", "chamfer"):
        raise RuntimeError(f"Unknown similarity algorithm: {algorithm!r}")

    noise_methods = _validated_noise_methods(discard_noise)

    dataset = (dataset or "auto").strip() or "auto"
    if dataset.startswith("ollama:"):
        # Text-description datasets have no chunks to filter.
        algorithm = "centroid"
        noise_methods = ()

    if algorithm == "pareto":
        # Dispatched before dataset resolution so a model-name dataset is
        # never mistaken for a track_embeddings lookup.
        from app.similarity.pareto import pareto_similar_tracks

        results = pareto_similar_tracks(conn, seed_track_id, dataset=dataset,
                                        limit=limit,
                                        discard_noise=noise_methods)
        return _prepend_seed(conn, seed_track_id, "pareto", results)
    if algorithm in ("emd", "chamfer"):
        from app.similarity.chunk_distances import similar_tracks_chunk_distance

        results = similar_tracks_chunk_distance(
            conn, seed_track_id, method=algorithm, dataset=dataset,
            limit=limit, discard_noise=noise_methods)
        return _prepend_seed(conn, seed_track_id, algorithm, results)

    chosen = _resolve_dataset(conn, dataset, ollama, seed_track_id)
    if chosen is None:
        raise RuntimeError(
            "No similarity dataset available: analyze at least two tracks with "
            "an embedding model (CLAP/MERT/OpenL3/FFT) or set up Ollama with "
            "an embedding model and generate track descriptions.")

    if noise_methods:
        scored = _noise_aware_centroid_scores(conn, seed_track_id, chosen,
                                              dataset, noise_methods)
        if scored is None:
            # No stored noise flags for this dataset — fall through to the
            # plain centroid comparison.
            pass
        else:
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
            scored = scored[: max(1, int(limit))]
            results: list[SimilarResult] = []
            for track_id, score in scored:
                track = repo.get_track(conn, track_id)
                if track is None:
                    continue
                results.append(SimilarResult(
                    track_id=track_id,
                    path=track["path"],
                    filename=track["filename"],
                    artist=track["artist"],
                    title=track["title"],
                    score=score,
                    method=chosen,
                ))
            if not results:
                raise RuntimeError(
                    f"No other tracks keep comparable '{chosen}' chunks "
                    "after the noise filter — analyze more tracks or use "
                    "fewer noise filters.")
            return _prepend_seed(conn, seed_track_id, chosen, results)

    seed_vec = repo.get_track_embedding(conn, seed_track_id, chosen)
    if seed_vec is None:
        raise RuntimeError(
            f"Seed track has no '{chosen}' embedding yet — analyze it first "
            "(right-click the track in the tree and choose Analyze).")

    # Learned component weights (Learning tab) scale every comparison:
    # vectors enter the weighted space BEFORE the similarity algorithm.
    from app.learning.weights import apply_weight_vector, weight_vector_for_dataset

    model_weights = weight_vector_for_dataset(conn, chosen)
    seed_vec = apply_weight_vector(seed_vec, model_weights)

    candidates = repo.get_track_embeddings(conn, chosen)
    candidates.pop(seed_track_id, None)
    scored: list[tuple[int, float]] = []
    for track_id, vec in candidates.items():
        if vec.shape != seed_vec.shape:
            continue
        scored.append((track_id,
                       cosine(seed_vec, apply_weight_vector(vec, model_weights))))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    scored = scored[: max(1, int(limit))]

    results: list[SimilarResult] = []
    for track_id, score in scored:
        track = repo.get_track(conn, track_id)
        if track is None:
            continue
        results.append(SimilarResult(
            track_id=track_id,
            path=track["path"],
            filename=track["filename"],
            artist=track["artist"],
            title=track["title"],
            score=score,
            method=chosen,
        ))
    if not results:
        raise RuntimeError(
            f"No other tracks have a '{chosen}' embedding yet — analyze more "
            "tracks first (Analyze All).")
    return _prepend_seed(conn, seed_track_id, chosen, results)


def geometric_mean(values: Iterable[float]) -> float:
    """Geometric mean of *values*; any value <= 0 yields 0.0.

    The combination rule for multi-reference similarity: a candidate's
    match percentages to every reference are multiplied and the N-th root
    taken, so a track must be similar to ALL references to rank high (an
    arithmetic mean would let a perfect match to one reference paper over
    a total mismatch to another).  Percentages live in [0, 1], so the mean
    stays in [0, 1]; a negative or zero match (e.g. negative cosine) means
    "no similarity" and floors the combined score at 0.
    """
    vals = [float(v) for v in values]
    if not vals or any(v <= 0.0 for v in vals):
        return 0.0
    total = 0.0
    for v in vals:
        total += math.log(v)
    return math.exp(total / len(vals))


def similar_tracks_multi(conn: sqlite3.Connection, seed_track_ids,
                         dataset: str = "auto", algorithm: str = "centroid",
                         limit: int = 20, ollama: OllamaClient | None = None,
                         discard_noise: "Iterable[str]" = (),
                         ) -> list[SimilarResult]:
    """Similarity to ONE OR MULTIPLE reference tracks, best first.

    For a single reference this is exactly :func:`similar_tracks` (the
    reference itself first at 100%).  For several references, each one is
    searched individually with the same dataset/algorithm/noise settings
    over the untruncated candidate set, and every candidate's
    per-reference similarity percentages are combined with their
    :func:`geometric mean` — see there for why.  The reference tracks
    themselves never appear in the results; a candidate must be comparable
    to *every* reference (same dataset vectors present) to be listed.

    With ``dataset="auto"`` and a centroid comparison, the concrete dataset
    is resolved once from the first reference so all per-reference
    percentages share one vector space; chunk-level algorithms (pareto /
    emd / chamfer) keep their per-reference ``"auto"`` model resolution.

    References without vectors in the dataset (e.g. a whole folder added
    as references where some tracks were never analyzed) are skipped, not
    fatal — the search fails only when NO reference is comparable.  The
    plain centroid comparison uses one shared candidate fetch and
    vectorized per-reference scoring, so even hundreds of folder
    references stay fast.
    """
    ids: list[int] = [int(t) for t in seed_track_ids]
    if not ids:
        raise RuntimeError(
            "No reference tracks selected — pick at least one file in the "
            "library tree.")
    if len(ids) == 1:
        return similar_tracks(conn, ids[0], dataset=dataset,
                              algorithm=algorithm, limit=limit,
                              ollama=ollama, discard_noise=discard_noise)

    resolved = (dataset or "auto").strip() or "auto"
    if resolved == "auto" and (algorithm or "centroid") not in (
            "pareto", "emd", "chamfer"):
        chosen = _resolve_dataset(conn, "auto", ollama, ids[0])
        if chosen is None:
            raise RuntimeError(
                "No similarity dataset available: analyze at least two tracks "
                "with an embedding model (CLAP/MERT/OpenL3/FFT) or set up "
                "Ollama with an embedding model and generate track "
                "descriptions.")
        resolved = chosen

    noise_methods = _validated_noise_methods(discard_noise)
    refs = set(ids)
    per_seed: list[dict[int, float]] = []
    method_label: str | None = None

    if (algorithm or "centroid") == "centroid" and not noise_methods:
        # Fast path: one candidate fetch, vectorized per-reference scoring.
        # Identical math to the single-reference centroid search (cosine,
        # 0.0 on zero norms, clipped to [-1, 1]).
        seed_vecs: dict[int, np.ndarray] = {}
        for sid in ids:
            vec = repo.get_track_embedding(conn, sid, resolved)
            if vec is not None and vec.size:
                seed_vecs[sid] = np.asarray(vec, dtype=np.float64)
        if not seed_vecs:
            raise RuntimeError(
                f"None of the reference tracks has a '{resolved}' embedding "
                "yet — analyze them first (right-click a track and choose "
                "Analyze).")
        candidates = repo.get_track_embeddings(conn, resolved)
        cand_ids = [tid for tid in candidates if tid not in refs]
        if cand_ids:
            cand_mat = np.stack([np.asarray(candidates[tid],
                                            dtype=np.float64)
                                 for tid in cand_ids])
            norms = np.linalg.norm(cand_mat, axis=1)
            for sid, svec in seed_vecs.items():
                if svec.shape[0] != cand_mat.shape[1]:
                    continue   # mixed dimensions — this reference scores none
                snorm = float(np.linalg.norm(svec))
                denom = norms * snorm
                with np.errstate(divide="ignore", invalid="ignore"):
                    scores = np.where(denom > 0,
                                      cand_mat @ svec / denom, 0.0)
                scores = np.clip(scores, -1.0, 1.0)
                per_seed.append(dict(zip(cand_ids,
                                         (float(s) for s in scores))))
        method_label = resolved
    else:
        for sid in ids:
            if not _reference_comparable(conn, sid, resolved):
                continue   # folder references may lack vectors — skip them
            # A huge limit keeps the per-reference candidate sets complete
            # so the geometric mean is not biased by top-N cuts.
            rows = similar_tracks(conn, sid, dataset=resolved,
                                  algorithm=algorithm, limit=10 ** 9,
                                  ollama=ollama, discard_noise=noise_methods)
            scores = {r.track_id: r.score for r in rows if r.track_id != sid}
            per_seed.append(scores)
            if method_label is None and rows:
                method_label = rows[0].method

    if not per_seed:
        raise RuntimeError(
            f"None of the {len(ids)} reference tracks is comparable in "
            f"dataset '{resolved}' — analyze them with that model first.")
    if len(per_seed) < len(ids):
        log.info("Multi-reference search: %d of %d references comparable",
                 len(per_seed), len(ids))

    common: set[int] = set(per_seed[0])
    for scores in per_seed[1:]:
        common &= set(scores)
    common -= set(ids)

    combined = [(tid, geometric_mean(scores[tid] for scores in per_seed))
                for tid in common]
    combined.sort(key=lambda pair: (-pair[1], pair[0]))
    combined = combined[: max(1, int(limit))]

    results: list[SimilarResult] = []
    for track_id, score in combined:
        track = repo.get_track(conn, track_id)
        if track is None:
            continue
        results.append(SimilarResult(
            track_id=track_id,
            path=track["path"],
            filename=track["filename"],
            artist=track["artist"],
            title=track["title"],
            score=score,
            method=method_label or str(resolved),
        ))
    if not results:
        raise RuntimeError(
            f"No tracks are comparable to ALL {len(ids)} reference tracks "
            f"(dataset '{resolved}') — analyze them together first or pick "
            "another dataset.")
    return results


def _reference_comparable(conn: sqlite3.Connection, track_id: int,
                          resolved: str) -> bool:
    """Does the track carry any vectors for *resolved*? (skip-decision)."""
    if resolved.startswith("red:"):
        return track_id in set(repo.tracks_with_reduced_chunks(
            conn, int(resolved[4:])))
    if resolved.startswith("ollama:"):
        return track_id in set(repo.tracks_with_embeddings(conn, resolved))
    if resolved == "auto":
        return bool(repo.get_chunks(conn, track_id))
    _n_chunks, coverage = repo.track_model_coverage(conn, track_id)
    return coverage.get(resolved, 0) > 0


def _split_legacy_method(method: str) -> tuple[str, str]:
    """Map the pre-1.4 ``method`` strings onto (dataset, algorithm)."""
    legacy = (method or "auto").strip() or "auto"
    if legacy in ("pareto", "psvi"):
        return "auto", "pareto"
    if legacy in ("emd", "chamfer"):
        return "auto", legacy
    if legacy.startswith("ollama:"):
        return legacy, "centroid"
    if legacy == "auto":
        return "auto", "centroid"
    # A concrete model name: centroid comparison over that model.
    return legacy, "centroid"


def _validated_noise_methods(discard_noise: Iterable[str]) -> tuple[str, ...]:
    """Deduplicated, order-preserving tuple of known noise-filter names."""
    from app.similarity.noise_filter import METHODS as NOISE_METHODS

    seen: list[str] = []
    for name in discard_noise or ():
        normalized = str(name).strip().lower()
        if normalized not in NOISE_METHODS:
            raise RuntimeError(
                f"Unknown noise filter: {name!r} — expected one of "
                f"{', '.join(NOISE_METHODS)}.")
        if normalized not in seen:
            seen.append(normalized)
    return tuple(seen)


def _noise_aware_centroid_scores(
        conn: sqlite3.Connection, seed_track_id: int, chosen: str,
        dataset: str, noise_methods: tuple[str, ...],
) -> list[tuple[int, float]] | None:
    """Centroid scores computed over noise-filtered chunks, or ``None``.

    Returns ``None`` when no stored noise run applies to *dataset* (the
    caller falls back to the precomputed track centroids); ``"auto"``
    pools the filters of every dataset, a concrete dataset uses only its
    own run.  Otherwise per-track centroids are recomputed from the
    surviving chunks — ``(track_id, cosine)`` pairs for every track that
    keeps at least one chunk; the seed is excluded.  Raises a friendly
    ``RuntimeError`` when the noise filter leaves the seed with no
    chunks at all.
    """
    from app.similarity.noise_filter import filtered_centroids, noise_ids_for

    noise = noise_ids_for(conn, dataset, noise_methods)
    if not noise:
        return None
    centroids = filtered_centroids(conn, chosen, noise)
    from app.learning.weights import apply_weight_vector, weight_vector_for_dataset

    model_weights = weight_vector_for_dataset(conn, chosen)
    seed_vec = centroids.pop(int(seed_track_id), None)
    if seed_vec is None:
        raise RuntimeError(
            "The noise filter discarded every chunk of the seed track — "
            "uncheck a noise filter (or refit it) and search again.")
    seed_vec = apply_weight_vector(seed_vec, model_weights)
    scored: list[tuple[int, float]] = []
    for track_id, vec in centroids.items():
        if vec.shape != seed_vec.shape:
            continue
        scored.append((track_id,
                       cosine(seed_vec, apply_weight_vector(vec, model_weights))))
    return scored


def _resolve_dataset(conn: sqlite3.Connection, dataset: str,
                     ollama: OllamaClient | None,
                     seed_track_id: int | None) -> str | None:
    """Concrete track_embeddings model for a centroid comparison.

    ``red:<id>`` datasets store their per-track centroids in
    ``track_embeddings`` under the model name ``red:<id>`` (written when the
    reduction is created), so they resolve to themselves.  ``auto`` keeps the
    legacy behaviour of :func:`resolve_method`.
    """
    if dataset.startswith("red:"):
        red_id = dataset[4:]
        try:
            int(red_id)
        except ValueError:
            raise RuntimeError(f"Unknown reduction dataset: {dataset!r}")
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM track_embeddings WHERE model = ?",
            (dataset,),
        ).fetchone()
        if rows is None or int(rows["n"]) < 2:
            raise RuntimeError(
                f"Reduction dataset {dataset!r} has fewer than two analyzed "
                "tracks — recreate it after analyzing more tracks.")
        return dataset
    if dataset.startswith("ollama:"):
        if len(repo.tracks_with_embeddings(conn, dataset)) < 2:
            raise RuntimeError(
                f"Dataset '{dataset}' needs at least two tracks with "
                "description embeddings — analyze more tracks first.")
        return dataset
    if dataset == "auto":
        return resolve_method(conn, "auto", ollama, seed_track_id)
    # A plain model name: validate that it is usable before searching.
    ids = set(repo.tracks_with_embeddings(conn, dataset))
    if seed_track_id is not None:
        if seed_track_id in ids and len(ids) >= 2:
            return dataset
        raise RuntimeError(
            f"Dataset '{dataset}' is not comparable yet — it needs the seed "
            "track plus at least one other track analyzed with that model.")
    if len(ids) >= 2:
        return dataset
    raise RuntimeError(
        f"Dataset '{dataset}' needs at least two analyzed tracks — analyze "
        "more tracks first (Analyze All).")
