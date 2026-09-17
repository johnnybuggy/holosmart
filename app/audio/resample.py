"""Linear-interpolation resampling (dependency-free)."""
from __future__ import annotations

import numpy as np


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio along axis 0 with linear interpolation.

    Supports 1-D (n,) and 2-D (n, channels) float arrays. Returns float32.
    """
    if orig_sr == target_sr:
        return np.asarray(samples, dtype=np.float32)
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Invalid sample rates: {orig_sr} -> {target_sr}")
    data = np.asarray(samples, dtype=np.float64)
    n_in = data.shape[0]
    if n_in == 0:
        return np.zeros(0, dtype=np.float32)
    n_out = max(1, int(round(n_in * target_sr / orig_sr)))
    x_in = np.linspace(0.0, max(n_in - 1, 1), num=n_in, endpoint=True)
    x_out = np.linspace(0.0, max(n_in - 1, 1), num=n_out, endpoint=True)
    if data.ndim == 1:
        out = np.interp(x_out, x_in, data)
    else:
        out = np.column_stack([np.interp(x_out, x_in, data[:, ch])
                               for ch in range(data.shape[1])])
    return out.astype(np.float32)
