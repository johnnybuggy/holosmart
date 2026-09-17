"""CLAP zero-shot audio tagging + audio embedding plugin."""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from app.models.base import AudioChunks, ModelPlugin

log = logging.getLogger(__name__)

CLAP_MODEL_ID = "laion/clap-htsat-unfused"

#: ~70 music labels (genres, moods, instruments) for zero-shot tagging.
CANDIDATE_TAGS: tuple[str, ...] = (
    # genres
    "rock", "pop", "jazz", "classical", "hip hop", "electronic", "ambient",
    "folk", "metal", "blues", "reggae", "country", "funk", "soul", "disco",
    "techno", "house", "trance", "dubstep", "punk", "grunge", "indie rock",
    "lounge", "gospel", "opera", "soundtrack", "world music", "latin",
    "bluegrass", "r and b", "rap", "synthwave", "lo-fi", "acoustic",
    # moods / character
    "dance", "sad", "happy", "calm", "energetic", "aggressive", "chill",
    "dark", "upbeat", "dreamy", "melancholic", "romantic", "epic",
    "mysterious", "tense", "hopeful", "playful", "nostalgic", "relaxing",
    "meditative", "anthemic", "groovy", "ethereal", "dramatic",
    # instruments / voices
    "piano", "guitar", "acoustic guitar", "electric guitar", "violin",
    "cello", "drums", "percussion", "synthesizer", "bass", "flute",
    "saxophone", "trumpet", "harp", "strings", "orchestral", "vocal",
    "male vocal", "female vocal", "choir", "beatboxing", "whistling",
    "organ", "banjo", "mandolin", "accordion", "theremin",
)


def _resample_array(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample via app.audio.resample, with a local linear-interp fallback."""
    if orig_sr == target_sr:
        return np.asarray(samples, dtype=np.float32)
    try:
        from app.audio.resample import resample  # sibling module, contract-guarded
    except Exception:  # ImportError or broken sibling during parallel dev
        x = np.arange(len(samples), dtype=np.float64)
        n_out = int(round(len(samples) * target_sr / orig_sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_out = np.linspace(0.0, len(samples) - 1.0, num=n_out)
        return np.interp(x_out, x, np.asarray(samples, dtype=np.float64)).astype(np.float32)
    return np.asarray(resample(samples, orig_sr, target_sr), dtype=np.float32)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def _extract_embeds(output: object) -> "np.ndarray | None":
    """Pull the embedding matrix out of whatever transformers v5 returns."""
    import torch

    if isinstance(output, torch.Tensor):
        return output.cpu().float().numpy()
    if isinstance(output, tuple):
        for part in output:
            got = _extract_embeds(part)
            if got is not None:
                return got
        return None
    for attr in ("audio_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(output, attr, None)
        if val is not None:
            return _extract_embeds(val)
    if isinstance(output, dict):
        for key in ("audio_embeds", "text_embeds", "pooler_output"):
            if key in output:
                return _extract_embeds(output[key])
    return None


class ClapPlugin(ModelPlugin):
    """CLAP audio embeddings and zero-shot genre/mood/instrument tagging."""

    name: ClassVar[str] = "clap"
    display_name: ClassVar[str] = "CLAP"
    embedding_dim: ClassVar[int | None] = 512
    provides_text: ClassVar[bool] = True
    preferred_sample_rate: ClassVar[int] = 48000
    requirements: ClassVar[tuple[str, ...]] = ("torch", "transformers")

    BATCH_SIZE = 8

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._device = None
        self._text_features = None
        self._tag_texts: tuple[str, ...] = ()
        # Editable settings (module constants are the defaults; apply_config()
        # overrides them from the persisted AppConfig).
        self.model_id: str = CLAP_MODEL_ID
        self.tag_top_k: int = 5
        self.batch_size: int = self.BATCH_SIZE
        # Zero-shot candidate tags; apply_config() swaps in the user's edited
        # list (AppConfig.clap_tags). _load() computes the text features for
        # exactly this list.
        self.tag_candidates: tuple[str, ...] = CANDIDATE_TAGS
        self._loaded_model_id: str | None = None

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt model id / tag top-k / batch size / tag list from an AppConfig.

        Tolerant of partially-built configs and test fakes: every attribute is
        read via ``getattr`` with a fallback. When the model id or the
        candidate tag list changes while weights are already loaded, the
        cached weights are dropped so the next analysis reloads the model and
        recomputes the zero-shot text features for the new settings.
        """
        old_id = self.model_id
        self.model_id = str(getattr(config, "clap_model_id", old_id) or old_id)
        try:
            self.tag_top_k = max(1, int(getattr(config, "clap_tag_top_k",
                                                self.tag_top_k)))
        except (TypeError, ValueError):
            pass
        try:
            self.batch_size = max(1, int(getattr(config, "clap_batch_size",
                                                 self.batch_size)))
        except (TypeError, ValueError):
            pass
        old_tags = self.tag_candidates
        custom = getattr(config, "clap_tags", None)
        if isinstance(custom, (list, tuple)) and custom:
            self.tag_candidates = tuple(str(t).strip() for t in custom
                                        if str(t).strip())
        else:
            self.tag_candidates = CANDIDATE_TAGS
        tags_changed = self.tag_candidates != old_tags
        if self._loaded and (tags_changed or self._loaded_model_id
                             and self._loaded_model_id != self.model_id):
            # The zero-shot text features are baked for one tag list; a list
            # change needs a reload exactly like a model-id change.
            self.unload()

    def unload(self) -> None:
        """Drop cached weights and forget which model id they belonged to."""
        super().unload()
        self._loaded_model_id = None

    # ---- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            model = ClapModel.from_pretrained(self.model_id)
            processor = ClapProcessor.from_pretrained(self.model_id)
        except Exception:
            log.info("CLAP remote fetch failed; retrying with local files only")
            model = ClapModel.from_pretrained(self.model_id, local_files_only=True)
            processor = ClapProcessor.from_pretrained(self.model_id,
                                                      local_files_only=True)
        model.to(device)
        model.eval()

        # Text features for the zero-shot tags: computed exactly once at load,
        # from the configured candidate list (CANDIDATE_TAGS or the user's
        # edited list from Settings → CLAP).
        tags = list(self.tag_candidates)
        text_inputs = processor(
            text=[f"This is a sound of {t}." for t in tags],
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            text_out = model.get_text_features(
                **{k: v.to(device) for k, v in text_inputs.items()
                   if k in ("input_ids", "attention_mask")}
            )
        text_feats = _extract_embeds(text_out)
        if text_feats is None:
            raise RuntimeError("CLAP get_text_features returned no usable embeddings")
        text_feats = _l2_normalize_rows(text_feats)

        self._model = model
        self._processor = processor
        self._device = device
        self._tag_texts = tuple(tags)
        self._text_features = text_feats
        self._loaded_model_id = self.model_id

    # ---- inference ----------------------------------------------------------
    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        import torch

        model, processor = self._model, self._processor
        resampled = [
            _resample_array(np.asarray(c, dtype=np.float32).reshape(-1), sr, 48000)
            for c in chunks
        ]
        out: list[np.ndarray] = []
        for start in range(0, len(resampled), self.batch_size):
            batch = resampled[start:start + self.batch_size]
            inputs = processor(
                audio=batch, sampling_rate=48000,
                return_tensors="pt", padding=True,
            )
            kwargs = {k: v.to(self._device) if torch.is_tensor(v) else v
                      for k, v in inputs.items()
                      if k in ("input_features", "is_longer", "attention_mask")}
            with torch.no_grad():
                audio_out = model.get_audio_features(**kwargs)
            feats = _extract_embeds(audio_out)
            if feats is None:
                raise RuntimeError("CLAP get_audio_features returned no usable embeddings")
            if feats.ndim == 1:
                feats = feats[None, :]
            out.extend(_l2_normalize_rows(feats).astype(np.float32))
        return out

    def _describe(
        self, chunks: list[np.ndarray], sr: int, top_k: int
    ) -> list[list[tuple[str, float]]]:
        import torch

        embeds = self._embed(chunks, sr)  # (n, dim), L2-normalized
        text_feats = self._text_features  # (n_tags, dim), L2-normalized
        results: list[list[tuple[str, float]]] = []
        for vec in embeds:
            sims = text_feats @ vec  # cosine similarities
            scores = torch.from_numpy(sims * 100.0)
            probs = torch.softmax(scores, dim=-1).numpy()
            order = np.argsort(-probs)[: max(1, top_k)]
            results.append([(self._tag_texts[i], float(probs[i])) for i in order])
        return results
