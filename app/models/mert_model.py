"""MERT (m-a-p/MERT-v1-*) audio embedding plugins.

Two sizes ship: :class:`MertPlugin` (MERT-v1-95M, 768-d, plugin key
``"mert"``) and :class:`Mert330Plugin` (MERT-v1-330M, 1024-d, plugin key
``"mert330"``).  They are separate plugins — not one configurable id —
so their embeddings live under different model keys in the library and
can coexist (768-d and 1024-d vectors must never share a key).
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from app.models.base import AudioChunks, ModelPlugin

log = logging.getLogger(__name__)

MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MERT330_MODEL_ID = "m-a-p/MERT-v1-330M"

#: Apple MPS (Metal) has a hard limit in the HuBERT conv frontend: inputs longer
#: than ~10 s of audio raise ``NotImplementedError: Output channels > 65536 not
#: supported at the MPS device``. Long chunks are therefore analyzed in windows
#: of at most MAX_WINDOW_SEC and their embeddings are averaged (then
#: re-normalized) into one vector per chunk.
MAX_WINDOW_SEC = 10.0
WINDOW_OVERLAP_SEC = 1.0


def split_windows(samples: np.ndarray, max_samples: int,
                  overlap_samples: int) -> list[np.ndarray]:
    """Split a 1-D array into overlapping windows of ``max_samples``.

    The final window always reaches the end of the input: when the natural
    stepping would leave a remainder shorter than 25% of ``max_samples``, the
    last window start is shifted back so it covers the tail fully instead.
    Inputs that already fit are returned unchanged (single window).
    """
    samples = np.asarray(samples).reshape(-1)
    n = len(samples)
    if n <= max_samples:
        return [samples]
    step = max(max_samples - overlap_samples, 1)
    starts = list(range(0, n - max_samples + 1, step))
    tail_start = n - max_samples
    if starts[-1] < tail_start:
        if tail_start - starts[-1] >= max_samples // 4:
            starts.append(tail_start)
        else:
            starts[-1] = tail_start
    return [samples[s:s + max_samples] for s in starts]


class MertPlugin(ModelPlugin):
    """HuBERT-based MERT music embeddings (768-d, mean-pooled over time)."""

    name: ClassVar[str] = "mert"
    display_name: ClassVar[str] = "MERT"
    embedding_dim: ClassVar[int | None] = 768
    provides_text: ClassVar[bool] = False
    preferred_sample_rate: ClassVar[int] = 24000
    requirements: ClassVar[tuple[str, ...]] = ("torch", "transformers")
    #: Prefix of the AppConfig fields this plugin reads
    #: (``<prefix>_model_id``, ``<prefix>_window_sec``, ...).
    settings_prefix: ClassVar[str] = "mert"
    default_model_id: ClassVar[str] = MERT_MODEL_ID

    BATCH_SIZE = 8

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._extractor = None
        self._device = None
        # Editable settings (class defaults; apply_config() overrides them
        # from the persisted AppConfig).
        self.model_id: str = self.default_model_id
        self.window_sec: float = MAX_WINDOW_SEC
        self.window_overlap_sec: float = WINDOW_OVERLAP_SEC
        self.batch_size: int = self.BATCH_SIZE
        self._loaded_model_id: str | None = None

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt model id / windowing / batch size from an AppConfig.

        Tolerant of partially-built configs and test fakes: every attribute is
        read via ``getattr`` with a fallback. The overlap is clamped to stay
        below the window length, and both window values must be positive. When
        the model id changes while weights are already loaded, the cached
        weights are dropped so the next analysis loads the new model.
        """
        prefix = self.settings_prefix
        old_id = self.model_id
        self.model_id = str(getattr(config, f"{prefix}_model_id", old_id)
                            or old_id)
        try:
            window_sec = float(getattr(config, f"{prefix}_window_sec",
                                       self.window_sec))
            overlap_sec = float(getattr(config, f"{prefix}_window_overlap_sec",
                                        self.window_overlap_sec))
        except (TypeError, ValueError):
            window_sec, overlap_sec = self.window_sec, self.window_overlap_sec
        window_sec = max(0.5, window_sec)
        # Overlap must stay strictly below the window length (clamp to win-0.5).
        overlap_sec = max(0.0, min(window_sec - 0.5, overlap_sec))
        self.window_sec = window_sec
        self.window_overlap_sec = overlap_sec
        try:
            self.batch_size = max(1, int(getattr(
                config, f"{prefix}_batch_size", self.batch_size)))
        except (TypeError, ValueError):
            pass
        if self._loaded:
            # An unknown loaded id (never set) is assumed to match the id the
            # plugin was constructed with, so an unchanged id never unloads.
            loaded_id = self._loaded_model_id if self._loaded_model_id else old_id
            if loaded_id != self.model_id:
                self.unload()

    def unload(self) -> None:
        """Drop cached weights and forget which model id they belonged to."""
        super().unload()
        self._loaded_model_id = None

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
        import torch
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            model = HubertModel.from_pretrained(self.model_id)
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
        except Exception:
            log.info("MERT remote fetch failed; retrying with local files only")
            model = HubertModel.from_pretrained(self.model_id, local_files_only=True)
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_id, local_files_only=True)
        model.to(device)
        model.eval()
        # The real hidden size wins over the declared default: variants of
        # the same family (95M = 768-d, 330M = 1024-d, fine-tunes) differ,
        # and _embed preallocates from this attribute.
        self.embedding_dim = int(getattr(model.config, "hidden_size",
                                         self.embedding_dim))
        self._model = model
        self._extractor = extractor
        self._device = device
        self._loaded_model_id = self.model_id

    # ---- inference -----------------------------------------------------------
    def _forward_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """One forward pass over a batch of equal-length windows.

        Returns an (n, dim) float32 array of time-mean-pooled (unnormalized)
        hidden states. Falls back to CPU once when MPS rejects an input shape
        (some operations are unimplemented on the Metal backend).
        """
        import torch

        extractor, model = self._extractor, self._model
        inputs = extractor(
            batch, sampling_rate=self.preferred_sample_rate,
            return_tensors="pt", padding=True,
        )
        inputs = {
            k: (v.to(self._device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }
        try:
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state  # (n, time, dim)
        except (NotImplementedError, RuntimeError) as exc:
            if self._device == "cpu" or "mps" not in str(getattr(self._device, "type", self._device)):
                raise
            log.warning("MERT: %s on %s — retrying this batch on CPU and "
                        "staying on CPU for the rest of the run", exc, self._device)
            self._model = self._model.to("cpu")
            self._device = "cpu"
            return self._forward_batch(batch)
        return hidden.mean(dim=1).cpu().float().numpy()

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        resampled = [self._resample(c, sr) for c in chunks]
        max_window = int(self.window_sec * self.preferred_sample_rate)
        overlap = int(self.window_overlap_sec * self.preferred_sample_rate)

        # Long chunks are analyzed in overlapping sub-windows (MPS conv limit).
        window_lists = [split_windows(c, max_window, overlap) for c in resampled]
        flat_windows = [w for windows in window_lists for w in windows]
        owner = [i for i, windows in enumerate(window_lists) for _ in windows]

        vectors = np.zeros((len(flat_windows), self.embedding_dim or 0),
                           dtype=np.float32)
        for start in range(0, len(flat_windows), self.batch_size):
            batch = flat_windows[start:start + self.batch_size]
            vectors[start:start + len(batch)] = self._forward_batch(batch)

        # Mean-pool each chunk's window embeddings, then L2-normalize.
        out: list[np.ndarray] = []
        for i in range(len(resampled)):
            member = vectors[[j for j, o in enumerate(owner) if o == i]]
            if not len(member):
                out.append(np.zeros(self.embedding_dim or 0, dtype=np.float32))
                continue
            pooled = member.mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            out.append((pooled / norm if norm > 0 else pooled).astype(np.float32))
        return out


class Mert330Plugin(MertPlugin):
    """MERT-v1-330M: the large MERT variant (1024-d hidden states).

    Stored under its own plugin key ``"mert330"`` so 330M and 95M
    embeddings can coexist in one library.  Roughly 4x the compute of
    the 95M per window, so the default batch size is smaller.
    """

    name: ClassVar[str] = "mert330"
    display_name: ClassVar[str] = "MERT-330M"
    embedding_dim: ClassVar[int | None] = 1024
    settings_prefix: ClassVar[str] = "mert330"
    default_model_id: ClassVar[str] = MERT330_MODEL_ID

    BATCH_SIZE = 4
