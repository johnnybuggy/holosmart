"""LP-MusicCaps plugin: music captions + mel-CNN audio embeddings.

Model: ``seungheondoh/lp-music-caps`` (ISMIR 2023) — a Whisper-style mel
front-end (16 kHz, 128 mels, 10 s windows) feeding a strided Conv1d stack
whose 768-d tokens condition a ``facebook/bart-base`` caption decoder.
The vendored model code lives in
:mod:`app.models.vendor.lpmc_bart` (the upstream pip/repo tree cannot
install on Python 3.14; only these two model files were needed).

The plugin ships BOTH halves:

* **embedding** — mean-pooled mel-CNN audio tokens per 10 s window
  (768-d, L2-normalized); the caption decoder is bypassed, so embedding
  runs never touch beam search;
* **text** — beam-5 captions per 10 s window (``transfer.pth``
  checkpoint), stored as the chunk's tags with score 1.0.

Checkpoint: the official ``transfer.pth`` (~1.8 GB, HF repo
``seungheondoh/lp-music-caps``), fetched once into the HF cache.
License caveat: the HF *model* repo is tagged MIT; the GitHub README
mentions CC-BY-NC for the caption DATASETS (not redistributed here).
"""
from __future__ import annotations

import logging
import math
from typing import ClassVar

import numpy as np

from app.models.base import ModelPlugin
from app.models.mert_model import split_windows
from app.models.muq_model import (
    _l2_normalize_rows, _resample_array, pad_to_max)

log = logging.getLogger(__name__)

LPMC_MODEL_ID = "seungheondoh/lp-music-caps"
LPMC_CHECKPOINT_FILE = "transfer.pth"

#: The model was trained on 10 s windows (N_SAMPLES = 10 s @ 16 kHz);
#: longer chunks are analyzed in 10 s windows without overlap — the
#: captioner's native granularity.
LPMC_WINDOW_SEC = 10.0


class LpMusicCapsPlugin(ModelPlugin):
    """LP-MusicCaps: 768-d mel-CNN embeddings + caption tags."""

    name: ClassVar[str] = "lpmc"
    display_name: ClassVar[str] = "LP-MusicCaps"
    embedding_dim: ClassVar[int | None] = 768
    provides_text: ClassVar[bool] = True
    preferred_sample_rate: ClassVar[int] = 16000
    requirements: ClassVar[tuple[str, ...]] = (
        "torch", "torchaudio", "transformers")
    settings_prefix: ClassVar[str] = "lpmc"
    default_model_id: ClassVar[str] = LPMC_MODEL_ID

    BATCH_SIZE = 4
    NUM_BEAMS = 5

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None
        self.model_id: str = self.default_model_id
        self.checkpoint_file: str = LPMC_CHECKPOINT_FILE
        self.window_sec: float = LPMC_WINDOW_SEC
        self.window_overlap_sec: float = 0.0
        self.batch_size: int = self.BATCH_SIZE
        self._loaded_model_id: str | None = None

    # ---- availability -------------------------------------------------------
    def is_available(self) -> bool:
        """Dependencies present AND the caption checkpoint reachable.

        The checkpoint itself ships on demand (first analysis run), so
        plain dependency presence is enough for the roster UI; the load
        surfaces download problems clearly.
        """
        return self._deps_available()

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt model id / checkpoint file / batch size from an AppConfig."""
        prefix = self.settings_prefix
        old_id = self.model_id
        self.model_id = str(getattr(config, f"{prefix}_model_id", old_id)
                            or old_id)
        self.checkpoint_file = str(
            getattr(config, f"{prefix}_checkpoint_file",
                    self.checkpoint_file) or self.checkpoint_file)
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

    # ---- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        from app.models.vendor.lpmc_bart import (
            BartCaptionModel, load_lpmc_checkpoint)

        device = ("mps" if torch.backends.mps.is_available() else "cpu")
        path = hf_hub_download(repo_id=self.model_id,
                               filename=self.checkpoint_file)
        model = BartCaptionModel()
        load_lpmc_checkpoint(model, path)
        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._loaded_model_id = self.model_id

    # ---- inference ----------------------------------------------------------
    def _forward_batch(self, batch: list[np.ndarray]) -> tuple[
            np.ndarray, list[str]]:
        """Embed + caption one padded batch of 10 s windows."""
        import torch

        padded = pad_to_max(batch)
        wavs = torch.from_numpy(
            np.stack([np.asarray(b, dtype=np.float32) for b in padded]))
        wavs = wavs.to(self._device)
        captions: list[str] = []
        try:
            with torch.no_grad():
                encoder_out, audio_embs = self._model.forward_encoder(wavs)
                pooled = audio_embs.mean(dim=1)          # (n, 768)
                captions = self._model.generate(
                    wavs, num_beams=self.NUM_BEAMS)
        except (NotImplementedError, RuntimeError) as exc:
            if self._device == "cpu" or "mps" not in str(
                    getattr(self._device, "type", self._device)):
                raise
            log.warning("%s: %s on %s — retrying this batch on CPU",
                        type(self).__name__, exc, self._device)
            import torch as _torch
            self._model = self._model.to("cpu")
            self._device = "cpu"
            with _torch.no_grad():
                _encoder_out, audio_embs = self._model.forward_encoder(wavs)
                pooled = audio_embs.mean(dim=1)
                captions = self._model.generate(wavs,
                                                num_beams=self.NUM_BEAMS)
        return pooled.cpu().float().numpy(), list(captions)

    def _windows(self, chunk: np.ndarray) -> list[np.ndarray]:
        """10 s windows covering one resampled chunk (final short window
        zero-padded by the batcher)."""
        max_window = int(self.window_sec * self.preferred_sample_rate)
        return split_windows(chunk, max_window, 0) or [chunk]

    def _analyze(self, chunks: list[np.ndarray], sr: int, *,
                 with_captions: bool) -> tuple[list[np.ndarray],
                                               list[list[str]]]:
        resampled = [_resample_array(c, sr, self.preferred_sample_rate)
                     for c in chunks]
        window_lists = [self._windows(c) for c in resampled]
        flat_windows = [w for windows in window_lists for w in windows]
        owner = [i for i, windows in enumerate(window_lists)
                 for _ in windows]
        # An empty chunk (decode failure) has no windows at all — it
        # falls through to the zero vector below instead of feeding an
        # empty waveform into the model.
        flat_windows = [w for w in flat_windows if len(w) > 0]
        if not flat_windows:
            return ([np.zeros(self.embedding_dim or 0, dtype=np.float32)
                     for _ in resampled], [[] for _ in resampled])

        per_owner_vecs: dict[int, list[np.ndarray]] = {}
        per_owner_caps: dict[int, list[str]] = {}
        for start in range(0, len(flat_windows), self.batch_size):
            batch = flat_windows[start:start + self.batch_size]
            batch_vecs, batch_caps = self._forward_batch(batch)
            for j, o in enumerate(owner[start:start + len(batch)]):
                per_owner_vecs.setdefault(o, []).append(batch_vecs[j])
                if with_captions and j < len(batch_caps):
                    per_owner_caps.setdefault(o, []).append(batch_caps[j])

        vectors: list[np.ndarray] = []
        tag_lists: list[list[str]] = []
        for i in range(len(resampled)):
            parts = per_owner_vecs.get(i)
            if not parts:
                vectors.append(np.zeros(self.embedding_dim or 0,
                                        dtype=np.float32))
                tag_lists.append([])
                continue
            pooled = np.mean(np.stack(parts), axis=0)
            vectors.append(_l2_normalize_rows(
                pooled[None, :])[0].astype(np.float32))
            captions = per_owner_caps.get(i, [])
            tag_lists.append([c for c in captions if c and c.strip()])
        return vectors, tag_lists

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        vectors, _tags = self._analyze(chunks, sr, with_captions=False)
        return vectors

    def describe(self, chunks, sr: int, top_k: int = 5
                 ) -> list[list[tuple[str, float]]] | None:
        """Beam captions per chunk, stored as tags with score 1.0."""
        if not self.provides_text:
            return None
        self.ensure_loaded()
        _vectors, tag_lists = self._analyze(list(chunks), sr,
                                            with_captions=True)
        return [[(caption, 1.0) for caption in tags]
                for tags in tag_lists]
