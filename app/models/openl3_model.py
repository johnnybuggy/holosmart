"""OpenL3 audio embedding plugin (requires openl3 + tensorflow; optional)."""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from app.models.base import ModelPlugin

log = logging.getLogger(__name__)


class OpenL3Plugin(ModelPlugin):
    """OpenL3 mel256/music 512-d embeddings.

    TensorFlow and openl3 are optional and not installed in this venv, so this
    plugin reports itself unavailable with a helpful message, while the code
    itself remains correct for environments where they are present.
    """

    name: ClassVar[str] = "openl3"
    display_name: ClassVar[str] = "OpenL3"
    embedding_dim: ClassVar[int | None] = 512
    provides_text: ClassVar[bool] = False
    preferred_sample_rate: ClassVar[int] = 48000
    requirements: ClassVar[tuple[str, ...]] = ("openl3", "tensorflow", "numpy")

    def __init__(self) -> None:
        super().__init__()
        self._openl3 = None

    def availability_error(self) -> str | None:
        """Explicit, actionable message (OpenL3/TF are intentionally absent)."""
        from app.models.base import module_available

        missing = [m for m in self.requirements if not module_available(m)]
        if not missing:
            return None
        return (
            "OpenL3 is unavailable because optional dependencies are missing: "
            + ", ".join(missing)
            + ". Install with: pip install openl3 tensorflow"
        )

    def _resample(self, samples: np.ndarray, orig_sr: int) -> np.ndarray:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if orig_sr == self.preferred_sample_rate:
            return arr
        try:
            from app.audio.resample import resample  # sibling module, contract-guarded
        except Exception:  # ImportError or broken sibling during parallel dev
            x = np.arange(len(arr), dtype=np.float64)
            n_out = int(round(len(arr) * self.preferred_sample_rate / orig_sr))
            if n_out <= 0:
                return np.zeros(0, dtype=np.float32)
            x_out = np.linspace(0.0, len(arr) - 1.0, num=n_out)
            return np.interp(x_out, x, arr.astype(np.float64)).astype(np.float32)
        return np.asarray(resample(arr, orig_sr, self.preferred_sample_rate), dtype=np.float32)

    def _load(self) -> None:
        import openl3  # lazy: heavy TF import

        self._openl3 = openl3

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        openl3 = self._openl3
        out: list[np.ndarray] = []
        for chunk in chunks:
            audio = self._resample(chunk, sr)
            emb, _ = openl3.get_audio_embedding(
                audio, self.preferred_sample_rate,
                input_repr="mel256", content_type="music",
                embedding_size=512, verbose=False,
            )
            emb = np.asarray(emb, dtype=np.float64)
            if emb.ndim == 3:  # (batch=1, n_frames, dim)
                emb = emb[0]
            vec = emb.mean(axis=0)  # mean over frames
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                norm = 1.0
            out.append((vec / norm).astype(np.float32))
        return out
