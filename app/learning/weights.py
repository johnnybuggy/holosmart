"""Learning from user-supplied song pairs: per-component similarity weights.

The user marks pairs of tracks that sound similar to them.  From those
pairs the app learns WHICH vector components carry the perceived
similarity — and which carry none.  The result is one non-negative weight
per vector component (primarily FFT, but any audio model works) that is
persisted in the ``learning_weights`` table and applied to every vector
comparison in the similarity search: each vector is scaled component-wise
by ``sqrt(weight)`` BEFORE any similarity algorithm runs, which turns plain
cosine / Euclidean chunk distances into their weighted counterparts.

Optimization objective
----------------------
Learning from positive pairs only (no "these sound different" pairs) must
not reward accidental positive correlations, so the objective works on
agreement magnitudes, not products.  With per-component dataset
normalization (see :mod:`app.analysis.normalization`), a component's
squared difference between two random tracks is comparable across
components, and the per-component mean squared difference over the pairs

    c_j = mean over pairs of (a_j − b_j)²

measures how strongly the user's perceived-similarity pairs agree on
component j.  The weights solve the diagonal-metric quadratic program

    minimize  Σ_j c_j w_j + λ Σ_j w_j²    subject to  Σ_j w_j = 1, w ≥ 0

(ridge-regularized: λ keeps the solution from collapsing onto the single
best component).  The KKT conditions give the closed form

    w_j = max(0, (ν − c_j) / (2λ))

with ν found by bisection so the weights sum to 1 — a small but genuine
optimization algorithm, deterministic and dependency-free.  Components
where paired songs consistently AGREE (small c_j) get the largest weights;
components that vary across songs the user calls similar end at zero —
exactly the "most vs. least similarity-carrying" split the Learning tab
reports.  The stored vector is rescaled to mean 1 afterwards so weighted
distances keep the same overall magnitude as unweighted ones (mean-1
scaling leaves cosine unchanged and keeps EMD/Chamfer percentages
comparable).
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "add_pair", "list_pairs", "remove_pair",
    "load_weight_vector", "save_weights", "clear_weights",
    "apply_weight_vector", "weight_vector_for_dataset",
    "scale_model_vectors", "weight_vectors_for_models",
    "learn_weights", "learn_and_store",
]


# ----------------------------------------------------------------- pairs ----
def add_pair(conn: sqlite3.Connection, track_a: int, track_b: int) -> int | None:
    """Store a similarity pair; returns its id or ``None`` when rejected.

    Rejected: same track twice, or an existing pair over the same two
    tracks (either order — the pair is an undirected statement).
    """
    a, b = int(track_a), int(track_b)
    if a == b:
        return None
    row = conn.execute(
        "SELECT id FROM learning_pairs WHERE (track_a = ? AND track_b = ?) "
        "OR (track_a = ? AND track_b = ?)", (a, b, b, a)).fetchone()
    if row is not None:
        return None
    cur = conn.execute(
        "INSERT INTO learning_pairs(track_a, track_b) VALUES (?, ?)", (a, b))
    return int(cur.lastrowid)


def list_pairs(conn: sqlite3.Connection) -> list[dict]:
    """All stored pairs with track metadata for display, oldest first."""
    rows = conn.execute(
        "SELECT p.id AS pair_id, p.track_a AS track_a, p.track_b AS track_b, "
        "ta.filename AS name_a, ta.artist AS artist_a, "
        "tb.filename AS name_b, tb.artist AS artist_b "
        "FROM learning_pairs p "
        "JOIN tracks ta ON ta.id = p.track_a "
        "JOIN tracks tb ON tb.id = p.track_b "
        "ORDER BY p.id").fetchall()
    return [dict(r) for r in rows]


def remove_pair(conn: sqlite3.Connection, pair_id: int) -> None:
    conn.execute("DELETE FROM learning_pairs WHERE id = ?", (int(pair_id),))


# --------------------------------------------------------------- weights ----
def save_weights(conn: sqlite3.Connection, model: str,
                 weights: np.ndarray) -> None:
    """Replace the stored weight vector of *model* (one row per component)."""
    conn.execute("DELETE FROM learning_weights WHERE model = ?", (model,))
    conn.executemany(
        "INSERT INTO learning_weights(model, component_index, weight) "
        "VALUES (?, ?, ?)",
        [(model, int(i), float(w)) for i, w in enumerate(weights)])


def load_weight_vector(conn: sqlite3.Connection, model: str,
                       ) -> np.ndarray | None:
    """The stored per-component weight vector of *model*, or ``None``."""
    rows = conn.execute(
        "SELECT component_index, weight FROM learning_weights "
        "WHERE model = ? ORDER BY component_index", (model,)).fetchall()
    if not rows:
        return None
    weights = np.zeros(max(int(r["component_index"]) for r in rows) + 1,
                       dtype=np.float64)
    for r in rows:
        weights[int(r["component_index"])] = float(r["weight"])
    return weights


def clear_weights(conn: sqlite3.Connection, model: str) -> int:
    """Drop the learned weights of *model*; returns the rows removed."""
    cur = conn.execute("DELETE FROM learning_weights WHERE model = ?", (model,))
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def apply_weight_vector(vec: np.ndarray,
                        weights: np.ndarray | None) -> np.ndarray:
    """Scale *vec* component-wise by ``sqrt(weights)`` (weighted space).

    Scaling by the SQUARE root means a plain cosine/Euclidean distance of
    the scaled vectors equals the weight-weighted distance of the originals
    — the weights are applied "before any similarity algorithm", exactly
    once, wherever vectors are compared.  Returns *vec* unchanged when
    there are no weights or the dimensionality does not match.
    """
    vec = np.asarray(vec, dtype=np.float64)
    if weights is None or weights.size != vec.size:
        return vec
    return vec * np.sqrt(np.asarray(weights, dtype=np.float64))


def weight_vector_for_dataset(conn: sqlite3.Connection,
                              dataset: str) -> np.ndarray | None:
    """Learned weights for a search dataset — ``None`` when not applicable.

    Reduced (``red:<id>``) and text (``ollama:<model>``) datasets live in
    their own component spaces, so raw-model weights never apply to them.
    """
    if dataset.startswith("red:") or dataset.startswith("ollama:"):
        return None
    return load_weight_vector(conn, dataset)


def scale_model_vectors(conn: sqlite3.Connection, by_model: dict) -> dict:
    """Apply learned weights to ``{model: {id: vec} | [vec, ...]}`` containers.

    Chunk-level algorithms (Pareto / EMD / Chamfer) fetch vectors grouped
    per model; this scales every vector by ``sqrt(w_model)`` so the
    downstream cosine / Euclidean distances become their weighted
    counterparts.  Models without learned weights (and reduced / text
    datasets) pass through untouched.
    """
    out: dict = {}
    for model, container in by_model.items():
        weights = weight_vector_for_dataset(conn, model)
        if weights is None:
            out[model] = container
        elif isinstance(container, dict):
            out[model] = {key: apply_weight_vector(vec, weights)
                          for key, vec in container.items()}
        else:
            out[model] = [apply_weight_vector(vec, weights)
                          for vec in container]
    return out


def weight_vectors_for_models(conn: sqlite3.Connection,
                              models: list[str]) -> dict[str, np.ndarray]:
    """``{model: weight vector}`` for the models that have learned weights."""
    out: dict[str, np.ndarray] = {}
    for model in models:
        if model.startswith("red:") or model.startswith("ollama:"):
            continue   # reduced/text spaces have their own component bases
        weights = load_weight_vector(conn, model)
        if weights is not None:
            out[model] = weights
    return out


# ------------------------------------------------------------ optimizer ----
def learn_weights(pairs: list[tuple[np.ndarray, np.ndarray]],
                  l2: float | None = None) -> np.ndarray:
    """Optimize per-component weights from positive similarity pairs.

    *pairs* carries ``(centroid_a, centroid_b)`` vectors (per-component
    normalized by the dataset normalization — see
    :mod:`app.analysis.normalization`).  Returns the weight vector with
    mean 1 and no negative entries; the ranking of its components is the
    "most vs. least similarity-carrying" answer.  ``l2`` overrides the
    ridge strength (default: half the mean per-component disagreement).
    """
    usable = [(np.asarray(a, dtype=np.float64),
               np.asarray(b, dtype=np.float64))
              for a, b in pairs
              if np.asarray(a).size and np.asarray(a).size == np.asarray(b).size
              and np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0]
    if not usable:
        raise RuntimeError(
            "No usable vector pairs for learning — analyze the paired "
            "tracks with the model first.")
    sq_diffs = np.stack([(a - b) ** 2 for a, b in usable])   # (n_pairs, dim)
    c = sq_diffs.mean(axis=0)                                # per-component
    dim = c.size
    if l2 is None:
        l2 = max(float(c.mean()) / 2.0, 1e-12)
    # KKT: w_j = max(0, (ν − c_j)/(2λ)), ν from Σ w = 1 via bisection
    # (Σ w is monotonically increasing in ν).
    lo = float(c.min()) - 2.0 * l2 * 1.0
    hi = float(c.max()) + 2.0 * l2 * dim

    def weights_for(nu: float) -> np.ndarray:
        return np.maximum(0.0, (nu - c) / (2.0 * l2))

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if weights_for(mid).sum() > 1.0:
            hi = mid
        else:
            lo = mid
    w = weights_for(0.5 * (lo + hi))
    total = float(w.sum())
    w = w / total if total > 1e-12 else np.full(dim, 1.0 / dim)
    return w * dim   # mean 1: weighted distances keep a familiar magnitude


def learn_and_store(db, model: str) -> dict:
    """Learn weights for *model* from the stored pairs; persist and report.

    Returns a summary dict (``model``, ``pairs_used``, ``pairs_stored``,
    ``weights``); raises ``RuntimeError`` with a friendly message when
    there are no pairs or none of the paired tracks has vectors for
    *model*.
    """
    conn = db.connect()
    try:
        pairs = list_pairs(conn)
        if not pairs:
            raise RuntimeError(
                "No song pairs to learn from — add at least one pair of "
                "tracks that sound similar to you first.")
        centroids: dict[int, np.ndarray] = {}
        wanted: set[int] = set()
        for p in pairs:
            wanted.add(int(p["track_a"]))
            wanted.add(int(p["track_b"]))
        for track_id in wanted:
            centroids[track_id] = repo.get_track_embedding(
                conn, track_id, model)
        vector_pairs = []
        missing = 0
        for p in pairs:
            a = centroids.get(int(p["track_a"]))
            b = centroids.get(int(p["track_b"]))
            if a is None or b is None:
                missing += 1
                continue
            vector_pairs.append((a, b))
        if not vector_pairs:
            raise RuntimeError(
                f"None of the paired tracks has a '{model}' centroid yet — "
                "analyze them with that model first.")
    finally:
        conn.close()

    weights = learn_weights(vector_pairs)
    with db.transaction() as tconn:
        save_weights(tconn, model, weights)
    log.info("Learned %d component weights for '%s' from %d pair(s) "
             "(%d skipped without vectors)", weights.size, model,
             len(vector_pairs), missing)
    return {"model": model, "pairs_used": len(vector_pairs),
            "pairs_stored": len(pairs), "pairs_missing": missing,
            "weights": weights}


from app.db import repo  # noqa: E402  (bottom import: avoids a cycle noise)
