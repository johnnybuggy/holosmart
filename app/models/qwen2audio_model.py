"""Qwen2-Audio plugin: Whisper-tower audio embeddings (Apache-2.0).

Model: ``Qwen/Qwen2-Audio-7B-Instruct`` — the audio tower is a
Whisper-large-v3-style encoder (128 mel bins, d_model 1280, 32 layers)
that consumes exactly 30 s @ 16 kHz per forward pass and emits a
(750, 1280) frame grid, post-LayerNorm.  The plugin mean-pools the frame
grid into a fixed 1280-d vector and L2-normalizes it.

transformers v5 loads the checkpoint natively (``Qwen2AudioForConditional
Generation``; the 4.38-era legacy key layout is handled by v5's own
mapping).  Only the audio tower + feature extractor are used — the
16.8 GB bf16 text LLM is discarded right after loading so inference
stays inside the tower's footprint (roughly 1.3 GB of tower weights
plus whatever the tokenizer/processor hold).

Longer chunks are analyzed in exact 30 s windows (no overlap — the tower
raises on any other mel-frame count); the final short window is
zero-padded by the feature extractor (``padding="max_length"``).
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from app.models.base import ModelPlugin
from app.models.mert_model import split_windows
from app.models.muq_model import (
    _l2_normalize_rows, _resample_array)

log = logging.getLogger(__name__)

QWEN2AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

#: The tower's hard constraint: exactly 3000 mel frames = 30 s @ 16 kHz
#: (max_source_positions 1500 × 2 after the AvgPool).  No overlap.
QWEN_WINDOW_SEC = 30.0


class Qwen2AudioPlugin(ModelPlugin):
    """Qwen2-Audio audio-tower embeddings (1280-d, 16 kHz)."""

    name: ClassVar[str] = "qwen2audio"
    display_name: ClassVar[str] = "Qwen2-Audio"
    embedding_dim: ClassVar[int | None] = 1280
    provides_text: ClassVar[bool] = False
    preferred_sample_rate: ClassVar[int] = 16000
    requirements: ClassVar[tuple[str, ...]] = ("torch", "transformers")
    settings_prefix: ClassVar[str] = "qwen2audio"
    default_model_id: ClassVar[str] = QWEN2AUDIO_MODEL_ID

    BATCH_SIZE = 2

    def __init__(self) -> None:
        super().__init__()
        self._model = None            # the audio tower only
        self._processor = None
        self._device = None
        self.model_id: str = self.default_model_id
        self.window_sec: float = QWEN_WINDOW_SEC   # fixed by the tower
        self.batch_size: int = self.BATCH_SIZE
        self._loaded_model_id: str | None = None

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt model id / batch size from an AppConfig."""
        prefix = self.settings_prefix
        old_id = self.model_id
        self.model_id = str(getattr(config, f"{prefix}_model_id", old_id)
                            or old_id)
        try:
            self.batch_size = max(1, int(getattr(
                config, f"{prefix}_batch_size", self.batch_size)))
        except (TypeError, ValueError):
            pass
        if self._loaded and self._loaded_model_id != self.model_id:
            self.unload()

    def unload(self) -> None:
        super().unload()
        self._loaded_model_id = None
        self._processor = None

    # ---- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2AudioEncoder

        device = ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "mps" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            # Load ONLY the audio tower (1280-d Whisper-style encoder):
            # the checkpoint's audio_tower.* weights map onto the encoder;
            # the 16.8 GB text LLM never enters memory.
            tower = Qwen2AudioEncoder.from_pretrained(
                self.model_id, dtype=dtype)
        except Exception:
            log.info("Qwen2-Audio: direct tower load failed — falling "
                     "back to the full checkpoint (more RAM needed)")
            from transformers import Qwen2AudioForConditionalGeneration
            full = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_id, dtype=dtype)
            tower = full.model.audio_tower
            del full
        tower = tower.to(device).eval()
        self._model = tower
        self._device = device
        self._loaded_model_id = self.model_id

    # ---- inference ----------------------------------------------------------
    def _forward_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """Tower forward on exact 30 s windows; (n, 1280) pooled vectors."""
        import torch

        features = self._processor.feature_extractor(
            [np.asarray(b, dtype=np.float32) for b in batch],
            sampling_rate=self.preferred_sample_rate, return_tensors="pt",
            padding="max_length")
        to_device = getattr(features, "to", None)   # BatchFeature or dict
        if to_device is not None:
            features = to_device(self._device)
        input_features = features["input_features"]
        if self._device == "mps" and input_features.dtype == torch.float32:
            input_features = input_features.to(torch.bfloat16)
        try:
            with torch.no_grad():
                out = self._model(input_features)
        except (NotImplementedError, RuntimeError) as exc:
            if self._device == "cpu" or "mps" not in str(
                    getattr(self._device, "type", self._device)):
                raise
            log.warning("%s: %s on %s — retrying this batch on CPU",
                        type(self).__name__, exc, self._device)
            self._model = self._model.to("cpu")
            self._device = "cpu"
            with torch.no_grad():
                out = self._model(input_features.to("cpu", torch.float32))
        frames = out.last_hidden_state               # (n, 750, 1280)
        return frames.mean(dim=1).cpu().float().numpy()

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        resampled = [_resample_array(c, sr, self.preferred_sample_rate)
                     for c in chunks]
        max_window = int(self.window_sec * self.preferred_sample_rate)
        window_lists = [split_windows(c, max_window, 0) or [c]
                        for c in resampled]
        flat_windows = [w for windows in window_lists for w in windows]
        owner = [i for i, windows in enumerate(window_lists)
                 for _ in windows]
        # An empty chunk (decode failure) has no windows — it falls
        # through to the zero vector below instead of feeding an empty
        # waveform into the model.
        flat_windows = [w for w in flat_windows if len(w) > 0]
        if not flat_windows:
            return [np.zeros(self.embedding_dim or 0, dtype=np.float32)
                    for _ in resampled]

        per_owner: dict[int, list[np.ndarray]] = {}
        for start in range(0, len(flat_windows), self.batch_size):
            batch = flat_windows[start:start + self.batch_size]
            batch_vecs = self._forward_batch(batch)
            for j, o in enumerate(owner[start:start + len(batch)]):
                per_owner.setdefault(o, []).append(batch_vecs[j])

        out: list[np.ndarray] = []
        for i in range(len(resampled)):
            parts = per_owner.get(i)
            if not parts:
                out.append(np.zeros(self.embedding_dim or 0,
                                    dtype=np.float32))
                continue
            pooled = np.mean(np.stack(parts), axis=0)
            out.append(_l2_normalize_rows(pooled[None, :])[0]
                       .astype(np.float32))
        return out
