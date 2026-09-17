"""Per-component dataset normalization of stored chunk vectors.

FFT band statistics live on wildly different scales (a fraction vs. tens of
Hz vs. an unbounded ratio), so raw component values are not comparable
across dimensions.  After every FFT analysis run the whole dataset's stored
FFT vectors are re-standardized per component (z-score: mean 0, std 1 over
ALL chunk vectors of the model), and the per-track centroids are rebuilt
from the normalized vectors.  Re-normalizing the WHOLE dataset each time —
not just the newly analyzed tracks — keeps every vector on exactly the same
scale no matter when it was written.

The learned similarity weights (``app.learning.weights``) build on this
normalized space; per-component z-scoring is what makes a learned weight
meaningful as "importance of this component" rather than an artifact of the
component's unit scale.
"""

from __future__ import annotations

import logging

import numpy as np

from app.db import repo
from app.db.database import blob_to_vec, vec_to_blob

log = logging.getLogger(__name__)

#: Components with (near-)zero variance carry no discriminative signal;
#: dividing by ~0 would blow them up, so they are left untouched.
_STD_FLOOR = 1e-9


def component_stats(conn, model: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-component mean/std over ALL stored chunk vectors of *model*.

    Returns ``(mean, std, n)`` — ``n`` is the number of usable vectors
    (all with the modal dimensionality); vectors of a foreign dimension
    (should not happen) are ignored.
    """
    rows = conn.execute(
        "SELECT vector FROM embeddings WHERE model = ?", (model,)).fetchall()
    vecs = [blob_to_vec(r["vector"]) for r in rows]
    if not vecs:
        return np.zeros(0), np.ones(0), 0
    dim = max(v.size for v in vecs)
    mat = np.stack([v for v in vecs if v.size == dim])
    return mat.mean(axis=0), mat.std(axis=0), int(mat.shape[0])


def normalize_model_components(conn, model: str) -> int:
    """Z-score every stored chunk vector of *model* per component.

    Rewrites the vectors in place (and their norms).  Returns the number of
    rewritten vectors; 0 when there is nothing to do (fewer than two
    vectors, or the dataset is already standardized).
    """
    rows = conn.execute(
        "SELECT id, vector FROM embeddings WHERE model = ?", (model,)
    ).fetchall()
    if len(rows) < 2:
        return 0
    vecs = [blob_to_vec(r["vector"]) for r in rows]
    dim = max(v.size for v in vecs)
    usable = [(int(r["id"]), v) for r, v in zip(rows, vecs) if v.size == dim]
    if len(usable) < 2:
        return 0
    mat = np.stack([v for _, v in usable])
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < _STD_FLOOR, 1.0, std)
    scaled = (mat - mean) / std
    if np.allclose(scaled, mat, atol=1e-6):
        return 0   # already normalized — no pointless rewrite
    updates = [(vec_to_blob(v.astype(np.float32)),
                float(np.linalg.norm(v)), cid)
               for (cid, _), v in zip(usable, scaled)]
    conn.executemany(
        "UPDATE embeddings SET vector = ?, norm = ? WHERE id = ?", updates)
    log.info("Normalized %d '%s' chunk vectors per component", len(updates),
             model)
    return len(updates)


def rebuild_model_centroids(conn, model: str) -> int:
    """Recompute every per-track centroid of *model* from its chunk vectors.

    Used after a normalization pass so ``track_embeddings`` matches the
    rewritten chunk vectors.  Returns the number of centroids stored.
    """
    from app.similarity.search import compute_centroid  # lazy: avoid import cycle

    conn.execute("DELETE FROM track_embeddings WHERE model = ?", (model,))
    track_ids = [int(r["track_id"]) for r in conn.execute(
        "SELECT DISTINCT c.track_id AS track_id FROM chunks c "
        "JOIN embeddings e ON e.chunk_id = c.id WHERE e.model = ?",
        (model,))]
    count = 0
    for track_id in track_ids:
        rows = conn.execute(
            "SELECT e.vector FROM embeddings e "
            "JOIN chunks c ON c.id = e.chunk_id "
            "WHERE c.track_id = ? AND e.model = ? ORDER BY c.idx",
            (track_id, model)).fetchall()
        centroid = compute_centroid([blob_to_vec(r["vector"]) for r in rows])
        if centroid.size == 0:
            continue
        repo.set_track_embedding(conn, track_id, model, centroid)
        count += 1
    log.info("Rebuilt %d '%s' track centroids", count, model)
    return count


def normalize_model(db, model: str) -> tuple[int, int]:
    """Normalize *model*'s chunk vectors per component, then rebuild centroids.

    One transaction so a crash never leaves vectors and centroids
    disagreeing.  Returns ``(vectors_normalized, centroids_rebuilt)``.
    """
    with db.transaction() as conn:
        n_vectors = normalize_model_components(conn, model)
        if not n_vectors:
            return 0, 0
        n_centroids = rebuild_model_centroids(conn, model)
    return n_vectors, n_centroids
