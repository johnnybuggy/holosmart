"""M2D-CLAP audio-language embedding plugin (NTT, IEEE Access 2025).

``m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025``: a ViT-base audio encoder
pre-trained with Masked Modeling Duo + CLAP objectives (16 kHz, 80-mel log
spectrograms, 10 s units), producing 768-d audio embeddings aligned with a
fine-tuned BERT text-embedding space that ships inside the checkpoint
("BpTI" = BERT text encoder, Text Included) — so it zero-shot tags audio
with the same candidate-tag mechanism as CLAP.

Licensing note: both the runtime (``portable_m2d.py``) and the weights are
NTT code distributed under a NON-COMMERCIAL evaluation license (see
https://github.com/nttcslab/m2d — LICENSE.pdf).  This plugin therefore does
NOT vendor any NTT code: on first load it fetches the official runtime file
and the official release zip from the NTT repository into
``<data dir>/models/m2d/`` and imports them from there.  By triggering the
download you accept NTT's evaluation license terms — for non-commercial,
internal evaluation use.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import ClassVar

import numpy as np

from app.models.base import ModelPlugin
from app.models.muq_model import _l2_normalize_rows, _softmax, _resample_array

log = logging.getLogger(__name__)

#: Pinned official sources (a GitHub release tag, not a moving branch).
M2D_RUNTIME_URL = (
    "https://raw.githubusercontent.com/nttcslab/m2d/v0.5.0/"
    "examples/portable_m2d.py")
M2D_WEIGHTS_URL = (
    "https://github.com/nttcslab/m2d/releases/download/v0.5.0/"
    "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip")
M2D_WEIGHTS_DIR = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
M2D_CHECKPOINT = "checkpoint-30.pth"

#: Official zero-shot text template (examples/Colab_M2D-CLAP_ESC-50_ZS).
TAG_TEMPLATE = "{tag} can be heard"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class M2dClapPlugin(ModelPlugin):
    """M2D-CLAP audio embeddings (768-d) + zero-shot audio tagging."""

    name: ClassVar[str] = "m2dclap"
    display_name: ClassVar[str] = "M2D-CLAP"
    embedding_dim: ClassVar[int | None] = 768
    provides_text: ClassVar[bool] = True
    preferred_sample_rate: ClassVar[int] = 16000
    requirements: ClassVar[tuple[str, ...]] = (
        "torch", "timm", "einops", "nnAudio", "transformers")
    settings_prefix: ClassVar[str] = "m2dclap"

    BATCH_SIZE = 4

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None
        self.batch_size: int = self.BATCH_SIZE
        self.tag_top_k: int = 5
        from app.models.clap_model import CANDIDATE_TAGS
        self.tag_candidates: tuple[str, ...] = tuple(CANDIDATE_TAGS)
        self._tag_texts: tuple[str, ...] = ()
        self._text_features: np.ndarray | None = None

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt batch size / tag top-k / tag list from an AppConfig.

        A changed tag list drops cached weights: the zero-shot text features
        are baked for exactly one tag list (CLAP semantics).
        """
        try:
            self.batch_size = max(1, int(getattr(
                config, f"{self.settings_prefix}_batch_size",
                self.batch_size)))
        except (TypeError, ValueError):
            pass
        try:
            self.tag_top_k = max(1, int(getattr(
                config, f"{self.settings_prefix}_tag_top_k",
                self.tag_top_k)))
        except (TypeError, ValueError):
            pass
        old_tags = self.tag_candidates
        custom = getattr(config, f"{self.settings_prefix}_tags", None)
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
            from app.models.clap_model import CANDIDATE_TAGS
            self.tag_candidates = tuple(CANDIDATE_TAGS)
        if self._loaded and self.tag_candidates != old_tags:
            self.unload()

    def unload(self) -> None:
        super().unload()
        self._text_features = None
        self._tag_texts = ()

    # ---- bootstrap ----------------------------------------------------------
    @staticmethod
    def _model_dir() -> Path:
        from app.config import DATA_DIR

        return Path(DATA_DIR) / "models" / "m2d"

    def _runtime_path(self) -> Path:
        return self._model_dir() / "portable_m2d.py"

    def _checkpoint_path(self) -> Path:
        return self._model_dir() / M2D_WEIGHTS_DIR / M2D_CHECKPOINT

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        """Stream *url* into *dest* (via a .part file), logging progress."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        log.info("Downloading %s → %s", url, dest)
        with urllib.request.urlopen(url, timeout=120) as resp, \
                open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            next_log = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total and done >= next_log:
                    log.info("  %d / %d MB (%d%%)",
                             done >> 20, total >> 20, done * 100 // total)
                    next_log += max(total >> 4, 1 << 20)
        if total and done < total:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download of {url} ended early ({done} of {total} bytes)")
        os.replace(tmp, dest)

    def _ensure_runtime(self) -> None:
        path = self._runtime_path()
        if not path.exists():
            self._download(M2D_RUNTIME_URL, path)

    def _ensure_weights(self) -> None:
        if self._checkpoint_path().exists():
            return
        base = self._model_dir()
        zip_path = base / f"{M2D_WEIGHTS_DIR}.zip"
        if not zip_path.exists():
            self._download(M2D_WEIGHTS_URL, zip_path)
        log.info("Unpacking %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(base)
        if not self._checkpoint_path().exists():
            raise RuntimeError(
                f"{M2D_WEIGHTS_URL} did not contain "
                f"{M2D_WEIGHTS_DIR}/{M2D_CHECKPOINT}")

    def _import_runtime(self):
        """Import the downloaded official portable runtime as a module."""
        path = self._runtime_path()
        spec = importlib.util.spec_from_file_location(
            "m2d_portable_runtime", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)   # noqa: exec_module OK (official file)
        return module

    # ---- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        import torch

        self._ensure_runtime()
        self._ensure_weights()
        runtime = self._import_runtime()

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = runtime.PortableM2D(
            weight_file=str(self._checkpoint_path()), flat_features=True)
        model.to(device)
        model.eval()
        self._model = model
        self._device = device

        # Zero-shot text features, computed once per load (CLAP pattern).
        tags = list(self.tag_candidates)
        text_feats = self._encode_texts([TAG_TEMPLATE.format(tag=t)
                                         for t in tags])
        self._tag_texts = tuple(tags)
        self._text_features = _l2_normalize_rows(text_feats)

    # ---- inference ----------------------------------------------------------
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            emb = self._model.encode_clap_text(list(texts))
        return emb.detach().cpu().float().numpy()

    def _forward_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """One audio forward pass; returns (n, 768) CLAP embeddings."""
        import torch

        max_len = max(len(b) for b in batch)
        wavs = torch.zeros((len(batch), max_len), dtype=torch.float32)
        for i, chunk in enumerate(batch):
            wavs[i, :len(chunk)] = torch.from_numpy(
                np.asarray(chunk, dtype=np.float32))
        wavs = wavs.to(self._device)
        try:
            with torch.no_grad():
                emb = self._model.encode_clap_audio(wavs)
        except (NotImplementedError, RuntimeError) as exc:
            if self._device == "cpu" or "mps" not in str(
                    getattr(self._device, "type", self._device)):
                raise
            log.warning("M2D-CLAP: %s on %s — retrying this batch on CPU and "
                        "staying on CPU for the rest of the run",
                        exc, self._device)
            self._model = self._model.to("cpu")
            self._device = "cpu"
            return self._forward_batch(batch)
        if emb.ndim == 1:
            emb = emb[None, :]
        return emb.detach().cpu().float().numpy()

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        resampled = [_resample_array(c, sr, self.preferred_sample_rate)
                     for c in chunks]
        out: list[np.ndarray] = []
        for start in range(0, len(resampled), self.batch_size):
            batch = resampled[start:start + self.batch_size]
            batch_vecs = self._forward_batch(batch)
            out.extend(_l2_normalize_rows(batch_vecs).astype(np.float32))
        return out

    def _describe(
        self, chunks: list[np.ndarray], sr: int, top_k: int
    ) -> list[list[tuple[str, float]]]:
        embeds = self._embed(chunks, sr)          # (n, dim), L2-normalized
        text_feats = self._text_features          # (n_tags, dim), L2-normalized
        results: list[list[tuple[str, float]]] = []
        for vec in embeds:
            sims = text_feats @ vec               # cosine similarities
            probs = _softmax(np.clip(sims * 100.0, -1e4, 1e4))
            order = np.argsort(-probs)[: max(1, top_k)]
            results.append([(self._tag_texts[i], float(probs[i]))
                            for i in order])
        return results
