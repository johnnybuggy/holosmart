"""Dimensionality reduction: PCA (built-in), t-SNE and UMAP (optional libs).

Used by two features:

* the **Visualisation dialog** — project chunk vectors to 2-D for a
  scatter plot (:func:`pca_2d` / :func:`tsne_2d` / :func:`umap_2d`);
* **similarity-search datasets** — store an n-component projection of a
  model's chunk vectors as a searchable dataset (:func:`fit_reduce`).

All methods are deterministic (fixed random seed where the implementation
accepts one) and standardize the input (zero mean, unit variance per
column, zero-variance columns stay zero) so mixed-scale components — e.g.
CLAP's ±0.5 range next to FFT's ±10 — project fairly.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "DimReductionError",
    "METHODS",
    "TSNE_MAX_POINTS",
    "UMAP_MAX_POINTS",
    "availability",
    "fit_reduce",
    "pca_2d",
    "pca_reduce",
    "reduce_2d",
    "standardize",
    "tsne_2d",
    "tsne_reduce",
    "umap_2d",
    "umap_reduce",
]

#: Random seed handed to the stochastic methods so re-running on the same
#: data reproduces the same result.
_SEED = 0

#: Point caps for the slow stochastic methods.  Callers refuse (with a
#: friendly, actionable error) instead of hanging for hours: scikit-learn's
#: t-SNE is quadratic-ish beyond ~15k points, while umap-learn stays
#: linear-ish and handles hundreds of thousands of points in minutes
#: (84k × 40 → 16 dims ≈ 1 min).  PCA needs no cap.
TSNE_MAX_POINTS = 15000
UMAP_MAX_POINTS = 200000

#: Reducers offered to the user.
METHODS = ("pca", "umap", "tsne")


class DimReductionError(RuntimeError):
    """A reduction is unavailable or its input/parameters are unusable."""


def _as_matrix(x) -> np.ndarray:
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise DimReductionError(
            "dimensionality reduction needs a non-empty (n, d) matrix")
    return matrix


def _check_components(n_components: int, max_allowed: int) -> int:
    n = int(n_components)
    if n < 2:
        raise DimReductionError("n_components must be at least 2")
    if max_allowed is not None and n > max_allowed:
        raise DimReductionError(
            f"n_components must be at most {max_allowed}")
    return n


def standardize(x) -> np.ndarray:
    """Zero mean, unit variance per column; zero-variance columns → 0."""
    matrix = _as_matrix(x)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return centered / std


# --------------------------------------------------------------------- PCA --
def pca_reduce(x, n_components: int = 2) -> tuple[np.ndarray, list[float]]:
    """Principal component projection via centered SVD.

    Returns ``(coords (n, n_components), explained_variance_ratios)`` — the
    ratios are 0.0 when the data has no variance at all.  Works for any
    ``n``/``d`` combination; *n_components* is capped to ``min(n, d)``.
    """
    matrix = _as_matrix(x)
    n_components = _check_components(n_components, None)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, u.shape[1], s.shape[0])
    coords = u[:, :k] * s[:k]
    total = float((s ** 2).sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros((matrix.shape[0], k)), [0.0] * k
    ratios = ((s[:k] ** 2) / total).tolist()
    return coords, ratios


def pca_2d(x) -> tuple[np.ndarray, list[str]]:
    """First two principal components + axis labels (``"PC1 (72.3%)"``)."""
    coords, ratios = pca_reduce(x, 2)
    labels = [f"PC{i + 1} ({ratio * 100.0:.1f}%)" for i, ratio in
              enumerate(ratios)]
    return coords, labels


# ------------------------------------------------------------------- t-SNE --
def tsne_reduce(x, n_components: int = 2, perplexity: float = 30.0
                ) -> tuple[np.ndarray, dict]:
    """t-SNE via scikit-learn (optional dependency)."""
    matrix = _as_matrix(x)
    n_components = _check_components(n_components, 3)
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:   # ImportError or a broken install
        raise DimReductionError(
            "t-SNE needs scikit-learn — install it with "
            "`pip install scikit-learn`") from exc
    if matrix.shape[0] > TSNE_MAX_POINTS:
        raise DimReductionError(
            f"t-SNE is limited to {TSNE_MAX_POINTS} points per run")
    embedded = TSNE(
        n_components=n_components, random_state=_SEED, init="pca",
        perplexity=min(float(perplexity),
                       max(5.0, (matrix.shape[0] - 1) / 4.0)),
    ).fit_transform(standardize(matrix))
    return embedded, {}


def tsne_2d(x) -> tuple[np.ndarray, list[str]]:
    """2-D t-SNE + axis labels (wrapper over :func:`tsne_reduce`)."""
    embedded, _info = tsne_reduce(x, 2)
    return embedded, ["t-SNE 1", "t-SNE 2"]


# -------------------------------------------------------------------- UMAP --
def umap_reduce(x, n_components: int = 2, n_neighbors: int = 15,
                min_dist: float = 0.1) -> tuple[np.ndarray, dict]:
    """UMAP via umap-learn (optional dependency)."""
    matrix = _as_matrix(x)
    n_components = _check_components(n_components, 32)
    try:
        import umap
    except Exception as exc:   # ImportError or a broken install
        raise DimReductionError(
            "UMAP needs umap-learn — install it with "
            "`pip install umap-learn`") from exc
    if matrix.shape[0] > UMAP_MAX_POINTS:
        raise DimReductionError(
            f"UMAP is limited to {UMAP_MAX_POINTS} points per run")
    embedded = umap.UMAP(
        n_components=n_components, random_state=_SEED, n_jobs=1,
        n_neighbors=max(2, int(n_neighbors)),
        min_dist=min(max(float(min_dist), 0.0), 1.0),
    ).fit_transform(standardize(matrix).astype(np.float32))
    return embedded, {}


def umap_2d(x) -> tuple[np.ndarray, list[str]]:
    """2-D UMAP + axis labels (wrapper over :func:`umap_reduce`)."""
    embedded, _info = umap_reduce(x, 2)
    return embedded, ["UMAP 1", "UMAP 2"]


# ---------------------------------------------------------------- dispatch --
def fit_reduce(x, method: str, n_components: int = 2, *,
               n_neighbors: int = 15, min_dist: float = 0.1,
               perplexity: float = 30.0) -> tuple[np.ndarray, dict]:
    """Reduce *x* to *n_components* dims with the chosen method.

    Returns ``(coords (n, n_components), info)`` where *info* carries
    ``"explained_variance"`` (ratio list) for PCA.  Raises
    :class:`DimReductionError` for unknown methods, missing optional
    libraries or bad parameters.
    """
    reducers = {
        "pca": lambda m: pca_reduce(m, n_components),
        "tsne": lambda m: tsne_reduce(m, n_components,
                                      perplexity=perplexity),
        "umap": lambda m: umap_reduce(m, n_components,
                                      n_neighbors=n_neighbors,
                                      min_dist=min_dist),
    }
    if method not in reducers:
        raise DimReductionError(f"unknown reduction method: {method!r}")
    coords, info = reducers[method](x)
    if method == "pca":
        info = {"explained_variance": info}
    return coords, info


def availability() -> dict[str, str | None]:
    """``{method: None | missing-dependency message}`` for the dialogs.

    ``pca`` is always available (numpy); ``tsne``/``umap`` entries carry a
    user-presentable install hint when their library is missing.
    """
    table: dict[str, str | None] = {}
    for method in METHODS:
        try:
            fit_reduce(np.zeros((4, 3)), method, 2)
            table[method] = None
        except DimReductionError as exc:
            table[method] = str(exc)
        except Exception:   # present but broken — still offer, fail at use
            table[method] = None
    return table


def reduce_2d(x, method: str) -> tuple[np.ndarray, list[str]]:
    """2-D projection + axis labels (Visualisation dialog helper)."""
    if method == "pca":
        return pca_2d(x)
    if method == "tsne":
        return tsne_2d(x)
    if method == "umap":
        return umap_2d(x)
    raise DimReductionError(f"unknown reduction method: {method!r}")
