"""MuQ and MuQ-MuLan audio embedding plugins (Tencent, arXiv 2501.01108).

Two plugins ship:

* :class:`MuQPlugin` — ``OpenMuQ/MuQ-large-msd-iter``: self-supervised music
  representation (Mel-RVQ masked modeling), ``output.last_hidden_state`` is
  mean-pooled over time into a 1024-d vector (plugin key ``"muq"``).
* :class:`MuQMuLanPlugin` — ``OpenMuQ/MuQ-MuLan-large``: CLIP-like music-text
  joint embedding (512-d) trained with contrastive learning over English and
  Chinese captions (plugin key ``"muqlan"``).  Because the audio and text
  spaces are aligned it can also zero-shot tag chunks (provides_text).

Both run through the official ``muq`` PyPI package and strictly require
24 kHz mono audio (resampling is handled here).  The paper recommends fp32
inference to avoid NaNs — no autocast is used anywhere.
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from app.models.base import ModelPlugin
from app.models.mert_model import split_windows

log = logging.getLogger(__name__)

MUQ_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
MUQ_MULAN_MODEL_ID = "OpenMuQ/MuQ-MuLan-large"

#: MuQ-MuLan was trained on ~10 s clips ("clip_secs": 10 in its config);
#: longer chunks are analyzed in 10 s windows (1 s overlap) and pooled.
MULAN_CLIP_SEC = 10.0
MULAN_CLIP_OVERLAP_SEC = 1.0

#: ~90 music labels for zero-shot tagging (same vocabulary as CLAP, so the
#: two taggers stay comparable in the Chunks tab).
CANDIDATE_TAGS: tuple[str, ...] = None  # filled from clap_model lazily


def candidate_tags() -> tuple[str, ...]:
    """The shared music-tag vocabulary (CLAP's list, imported lazily)."""
    global CANDIDATE_TAGS
    if CANDIDATE_TAGS is None:
        from app.models.clap_model import CANDIDATE_TAGS as _TAGS
        CANDIDATE_TAGS = _TAGS
    return CANDIDATE_TAGS


def _resample_array(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample via app.audio.resample, with a local linear-interp fallback."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr:
        return samples
    try:
        from app.audio.resample import resample  # sibling module, contract-guarded
    except Exception:  # ImportError or broken sibling during parallel dev
        x = np.arange(len(samples), dtype=np.float64)
        n_out = int(round(len(samples) * target_sr / orig_sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_out = np.linspace(0.0, len(samples) - 1.0, num=n_out)
        return np.interp(x_out, x, samples.astype(np.float64)).astype(np.float32)
    return np.asarray(resample(samples, orig_sr, target_sr), dtype=np.float32)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def pad_to_max(batch: list[np.ndarray]) -> "list[np.ndarray]":
    """Zero-pad a batch of 1-D float32 arrays to a common length.

    MuQ / MuQ-MuLan take a plain (batch, samples) waveform tensor with no
    padding mask, so every wave in a forward pass must share its length.
    Zero padding at the tail is the standard approach for batched audio.
    """
    if len(batch) == 1:
        return batch
    max_len = max(len(b) for b in batch)
    if all(len(b) == max_len for b in batch):
        return batch
    return [
        np.concatenate([b, np.zeros(max_len - len(b), dtype=np.float32)])
        if len(b) < max_len else b
        for b in batch
    ]


def _patch_wav2vec2_easydict_configs(model: object) -> None:
    """Make muq 0.1.0 loadable under transformers v5.

    Two incompatibilities bite at the first forward pass:

    * muq builds its Wav2Vec2Conformer encoder with a plain ``EasyDict``
      config, which predates transformers v5's masking API — the encoder
      dies with ``AttributeError: 'EasyDict' object has no attribute
      '_attn_implementation'``.
    * transformers v5 removed per-layer ``hidden_states`` from the
      ``Wav2Vec2ConformerEncoder`` output, while muq reads
      ``out["hidden_states"]`` (both the MuQ feature path and MuQ-MuLan's
      ``use_layer_idx`` depend on the v4 tuple semantics).

    Both are fixed without touching weights: the missing config attribute
    is stamped on, and each conformer encoder's forward is wrapped to
    rebuild the v4 ``hidden_states`` tuple (raw per-layer outputs, then
    the final layer-normed state) via forward hooks.
    """
    try:
        from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
            Wav2Vec2ConformerEncoder,
        )
    except Exception:
        return

    modules = (model.modules() if hasattr(model, "modules")
               else (model,))
    for module in modules:
        try:
            config = getattr(module, "config", None)
        except Exception:
            config = None
        if config is not None and not isinstance(
                config, (str, int, float, bool)):
            try:
                if getattr(config, "_attn_implementation", None) is None:
                    config._attn_implementation = "eager"
            except Exception:
                pass
        if isinstance(module, Wav2Vec2ConformerEncoder):
            _wrap_conformer_hidden_states(module)


def _wrap_conformer_hidden_states(conformer) -> None:
    """Wrap a Wav2Vec2ConformerEncoder so ``output_hidden_states=True``
    yields the v4-style ``hidden_states`` tuple transformers v5 dropped."""
    if getattr(conformer, "_holosmart_v5_hidden_states_patch", False):
        return
    orig_forward = conformer.forward

    def forward(hidden_states, attention_mask=None,
                output_hidden_states=False, **kwargs):
        if not output_hidden_states:
            return orig_forward(hidden_states,
                                attention_mask=attention_mask, **kwargs)
        collected: list = []
        hooks = [layer.register_forward_hook(
            lambda mod, args, out: collected.append(out))
            for layer in conformer.layers]
        try:
            out = orig_forward(hidden_states,
                               attention_mask=attention_mask, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        last = out.last_hidden_state
        # v4 semantics: every raw layer output, then the final
        # layer-normed state (so hidden_states[-1] == last_hidden_state).
        hidden = tuple(collected) + (last,)
        return type(out)(last_hidden_state=last, hidden_states=hidden)

    conformer.forward = forward
    conformer._holosmart_v5_hidden_states_patch = True


class _MpsFallbackMixin:
    """Retry a failed forward pass on CPU once (Metal gaps), like MERT."""

    def _to_cpu_and_retry(self, exc: Exception, batch):
        log.warning("%s: %s on %s — retrying this batch on CPU and staying "
                    "on CPU for the rest of the run",
                    type(self).__name__, exc, self._device)
        import torch
        self._model = self._model.to("cpu")
        self._device = "cpu"
        return self._forward_batch(batch)


class MuQPlugin(_MpsFallbackMixin, ModelPlugin):
    """MuQ SSL music embeddings (1024-d, time-pooled last hidden state)."""

    name: ClassVar[str] = "muq"
    display_name: ClassVar[str] = "MuQ"
    embedding_dim: ClassVar[int | None] = 1024
    provides_text: ClassVar[bool] = False
    preferred_sample_rate: ClassVar[int] = 24000
    # The 'muq' package ships broken metadata (no Requires-Dist for pip), so
    # its import-time dependencies are checked here too: 'from muq import …'
    # pulls easydict/torchaudio/x_clip immediately and crashes without them.
    requirements: ClassVar[tuple[str, ...]] = (
        "torch", "muq", "easydict", "torchaudio", "x_clip")
    settings_prefix: ClassVar[str] = "muq"
    default_model_id: ClassVar[str] = MUQ_MODEL_ID

    BATCH_SIZE = 4

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None
        self.model_id: str = self.default_model_id
        self.window_sec: float = 10.0
        self.window_overlap_sec: float = 1.0
        self.batch_size: int = self.BATCH_SIZE
        self._loaded_model_id: str | None = None

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt model id / windowing / batch size from an AppConfig.

        Same tolerant getattr-with-fallback semantics as the MERT plugins;
        a changed model id drops cached weights.
        """
        prefix = self.settings_prefix
        old_id = self.model_id
        self.model_id = str(getattr(config, f"{prefix}_model_id", old_id)
                            or old_id)
        try:
            window_sec = float(getattr(config, f"{prefix}_window_sec",
                                       self.window_sec))
            overlap_sec = float(getattr(
                config, f"{prefix}_window_overlap_sec",
                self.window_overlap_sec))
        except (TypeError, ValueError):
            window_sec, overlap_sec = self.window_sec, self.window_overlap_sec
        window_sec = max(0.5, window_sec)
        overlap_sec = max(0.0, min(window_sec - 0.5, overlap_sec))
        self.window_sec = window_sec
        self.window_overlap_sec = overlap_sec
        try:
            self.batch_size = max(1, int(getattr(
                config, f"{prefix}_batch_size", self.batch_size)))
        except (TypeError, ValueError):
            pass
        if self._loaded:
            loaded_id = self._loaded_model_id if self._loaded_model_id else old_id
            if loaded_id != self.model_id:
                self.unload()

    def unload(self) -> None:
        super().unload()
        self._loaded_model_id = None

    # ---- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        import torch
        from muq import MuQ

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            model = MuQ.from_pretrained(self.model_id)
        except Exception:
            log.info("MuQ remote fetch failed; retrying with local files only")
            model = MuQ.from_pretrained(self.model_id, local_files_only=True)
        _patch_wav2vec2_easydict_configs(model)
        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._loaded_model_id = self.model_id

    # ---- inference ----------------------------------------------------------
    def _forward_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """One forward pass; returns (n, dim) time-pooled hidden states."""
        import torch

        padded = pad_to_max(batch)
        wavs = torch.from_numpy(
            np.stack([np.asarray(b, dtype=np.float32) for b in padded]))
        wavs = wavs.to(self._device)
        try:
            with torch.no_grad():
                output = self._model(wavs, output_hidden_states=True)
            hidden = output.last_hidden_state           # (n, time, dim)
        except (NotImplementedError, RuntimeError) as exc:
            if self._device == "cpu" or "mps" not in str(
                    getattr(self._device, "type", self._device)):
                raise
            return self._to_cpu_and_retry(exc, batch)
        return hidden.mean(dim=1).cpu().float().numpy()

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        resampled = [_resample_array(c, sr, self.preferred_sample_rate)
                     for c in chunks]
        max_window = int(self.window_sec * self.preferred_sample_rate)
        overlap = int(self.window_overlap_sec * self.preferred_sample_rate)

        window_lists = [split_windows(c, max_window, overlap)
                        for c in resampled]
        flat_windows = [w for windows in window_lists for w in windows]
        owner = [i for i, windows in enumerate(window_lists) for _ in windows]

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
                out.append(np.zeros(self.embedding_dim or 0, dtype=np.float32))
                continue
            pooled = np.mean(np.stack(parts), axis=0)
            out.append(_l2_normalize_rows(pooled[None, :])[0].astype(np.float32))
        return out


class MuQMuLanPlugin(_MpsFallbackMixin, ModelPlugin):
    """MuQ-MuLan joint music-text embeddings (512-d) + zero-shot tagging."""

    name: ClassVar[str] = "muqlan"
    display_name: ClassVar[str] = "MuQ-MuLan"
    embedding_dim: ClassVar[int | None] = 512
    provides_text: ClassVar[bool] = True
    preferred_sample_rate: ClassVar[int] = 24000
    # see MuQPlugin: the muq package's own metadata is incomplete, so the
    # import-time modules are gated here as well
    requirements: ClassVar[tuple[str, ...]] = (
        "torch", "muq", "easydict", "torchaudio", "x_clip")
    settings_prefix: ClassVar[str] = "muqlan"
    default_model_id: ClassVar[str] = MUQ_MULAN_MODEL_ID

    BATCH_SIZE = 4

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None
        self.model_id: str = self.default_model_id
        self.window_sec: float = MULAN_CLIP_SEC
        self.window_overlap_sec: float = MULAN_CLIP_OVERLAP_SEC
        self.batch_size: int = self.BATCH_SIZE
        self.tag_top_k: int = 5
        self.tag_candidates: tuple[str, ...] = candidate_tags()
        self._loaded_model_id: str | None = None
        self._tag_texts: tuple[str, ...] = ()
        self._text_features: np.ndarray | None = None

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt model id / windowing / batch size / tag list, MuLan flavor.

        A changed model id OR tag list drops cached weights: the zero-shot
        text features are baked for exactly one tag list.
        """
        prefix = self.settings_prefix
        old_id = self.model_id
        self.model_id = str(getattr(config, f"{prefix}_model_id", old_id)
                            or old_id)
        try:
            window_sec = float(getattr(config, f"{prefix}_window_sec",
                                       self.window_sec))
            overlap_sec = float(getattr(
                config, f"{prefix}_window_overlap_sec",
                self.window_overlap_sec))
        except (TypeError, ValueError):
            window_sec, overlap_sec = self.window_sec, self.window_overlap_sec
        window_sec = max(0.5, window_sec)
        overlap_sec = max(0.0, min(window_sec - 0.5, overlap_sec))
        self.window_sec = window_sec
        self.window_overlap_sec = overlap_sec
        try:
            self.batch_size = max(1, int(getattr(
                config, f"{prefix}_batch_size", self.batch_size)))
        except (TypeError, ValueError):
            pass
        try:
            self.tag_top_k = max(1, int(getattr(
                config, f"{prefix}_tag_top_k", self.tag_top_k)))
        except (TypeError, ValueError):
            pass
        old_tags = self.tag_candidates
        custom = getattr(config, f"{prefix}_tags", None)
        if isinstance(custom, (list, tuple)) and custom:
            seen: set[str] = set()
            tags: list[str] = []
            for raw in custom:
                tag = str(raw).strip()
                if tag and tag.casefold() not in seen:
                    seen.add(tag.casefold())
                    tags.append(tag)
            self.tag_candidates = tuple(tags)
        else:
            self.tag_candidates = candidate_tags()
        tags_changed = self.tag_candidates != old_tags
        if self._loaded and (tags_changed
                             or self._loaded_model_id
                             and self._loaded_model_id != self.model_id):
            self.unload()

    def unload(self) -> None:
        super().unload()
        self._loaded_model_id = None
        self._text_features = None
        self._tag_texts = ()

    # ---- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        import torch
        from muq import MuQMuLan

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            model = MuQMuLan.from_pretrained(self.model_id)
        except Exception:
            log.info("MuQ-MuLan remote fetch failed; local files only")
            model = MuQMuLan.from_pretrained(self.model_id,
                                             local_files_only=True)
        _patch_wav2vec2_easydict_configs(model)
        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._loaded_model_id = self.model_id

        # Zero-shot text features, computed once per load (CLAP pattern).
        tags = list(self.tag_candidates)
        text_feats = self._encode_texts([f"{t} music." for t in tags])
        self._tag_texts = tuple(tags)
        self._text_features = _l2_normalize_rows(text_feats)

    # ---- inference ----------------------------------------------------------
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            output = self._model(texts=list(texts))
        return self._to_numpy_matrix(output)

    def _forward_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """One audio forward pass; returns (n, 512) MuLan embeddings."""
        import torch

        padded = pad_to_max(batch)
        wavs = torch.from_numpy(
            np.stack([np.asarray(b, dtype=np.float32) for b in padded]))
        wavs = wavs.to(self._device)
        try:
            with torch.no_grad():
                output = self._model(wavs=wavs)
        except (NotImplementedError, RuntimeError) as exc:
            if self._device == "cpu" or "mps" not in str(
                    getattr(self._device, "type", self._device)):
                raise
            return self._to_cpu_and_retry(exc, batch)
        return self._to_numpy_matrix(output)

    @staticmethod
    def _to_numpy_matrix(output: object) -> np.ndarray:
        """MuLan returns a (n, dim) tensor (or a tuple/dict wrapping one)."""
        import torch

        if isinstance(output, torch.Tensor):
            mat = output
        elif isinstance(output, dict):
            mat = next(v for v in output.values()
                       if torch.is_tensor(v) and v.ndim == 2)
        else:
            mat = output
            if not torch.is_tensor(mat):
                for attr in ("audio_embeds", "text_embeds", "embeds"):
                    val = getattr(mat, attr, None)
                    if val is not None:
                        mat = val
                        break
        return mat.detach().cpu().float().numpy()

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        resampled = [_resample_array(c, sr, self.preferred_sample_rate)
                     for c in chunks]
        max_window = int(self.window_sec * self.preferred_sample_rate)
        overlap = int(self.window_overlap_sec * self.preferred_sample_rate)

        window_lists = [split_windows(c, max_window, overlap)
                        for c in resampled]
        flat_windows = [w for windows in window_lists for w in windows]
        owner = [i for i, windows in enumerate(window_lists) for _ in windows]

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
                out.append(np.zeros(self.embedding_dim or 0, dtype=np.float32))
                continue
            pooled = np.mean(np.stack(parts), axis=0)
            out.append(_l2_normalize_rows(pooled[None, :])[0].astype(np.float32))
        return out

    def _describe(
        self, chunks: list[np.ndarray], sr: int, top_k: int
    ) -> list[list[tuple[str, float]]]:
        embeds = self._embed(chunks, sr)          # (n, dim), L2-normalized
        text_feats = self._text_features          # (n_tags, dim), L2-normalized
        results: list[list[tuple[str, float]]] = []
        for vec in embeds:
            sims = text_feats @ vec               # cosine similarities
            scores = np.clip(sims * 100.0, -1e4, 1e4)
            probs = _softmax(scores)
            order = np.argsort(-probs)[: max(1, top_k)]
            results.append([(self._tag_texts[i], float(probs[i]))
                            for i in order])
        return results


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)
