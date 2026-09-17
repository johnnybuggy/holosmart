"""Model plugin contract.

Every analysis model (CLAP, MERT, OpenL3, ...) implements :class:`ModelPlugin`.
Heavy imports (torch, transformers, tensorflow, openl3) MUST happen lazily inside
``_load``/``_embed``/``_describe`` - never at module import time - so the GUI
starts instantly even without any model installed.

``is_available`` must be CHEAP: use importlib.util.find_spec probes only, never
import torch or download weights.
"""
from __future__ import annotations

import abc
import importlib.util
import logging
import threading
from typing import ClassVar, Sequence

import numpy as np

log = logging.getLogger(__name__)

AudioChunks = Sequence[np.ndarray]  # each chunk: 1-D float32 mono samples


def module_available(name: str) -> bool:
    """Cheap importability probe that does NOT import the module."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


class ModelPlugin(abc.ABC):
    """Base class for audio-embedding model plugins."""

    name: ClassVar[str] = "base"
    display_name: ClassVar[str] = "Base"
    embedding_dim: ClassVar[int | None] = None  # None = decided at load time
    provides_text: ClassVar[bool] = False       # True -> describe() returns tags
    preferred_sample_rate: ClassVar[int] = 48000
    #: Python modules that must be importable for this plugin to be available.
    requirements: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        self._loaded = False
        # Serializes first-use weight loading: plugins are process-wide
        # singletons, and parallel analysis threads can call ensure_loaded()
        # on the same not-yet-loaded instance simultaneously.
        self._load_lock = threading.Lock()

    # ---- availability -----------------------------------------------------
    @classmethod
    def _deps_available(cls) -> bool:
        return all(module_available(m) for m in cls.requirements)

    def is_available(self) -> bool:
        """True when required packages are importable. Cheap, no side effects."""
        return self._deps_available()

    def availability_error(self) -> str | None:
        """Human-readable reason when is_available() is False, else None."""
        missing = [m for m in self.requirements if not module_available(m)]
        if not missing:
            return None
        return "Missing dependencies: %s. Install with: pip install %s" % (
            ", ".join(missing), " ".join(missing))

    # ---- lifecycle ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load model weights once. Raises RuntimeError with a clear message on failure.

        Thread-safe: a double-checked lock guarantees concurrent first use of
        a singleton plugin cannot load the weights twice. Failure semantics
        are unchanged — the RuntimeError propagates and ``_loaded`` stays
        False so a later call can retry. Inference itself is NOT serialized.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:  # another thread finished loading while we waited
                return
            if not self.is_available():
                raise RuntimeError(f"{self.display_name} unavailable: {self.availability_error()}")
            log.info("Loading model '%s' (weights may download on first use)...", self.name)
            try:
                self._load()
            except Exception as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(f"Failed to load {self.display_name}: {exc}") from exc
            self._loaded = True
        log.info("Model '%s' loaded.", self.name)

    def _load(self) -> None:
        """Override: perform the heavy imports and load weights."""

    # ---- inference ----------------------------------------------------------
    def embed(self, chunks: AudioChunks, sr: int) -> list[np.ndarray]:
        """Return one 1-D float32 vector per chunk. Resampling to the plugin's
        preferred sample rate is the plugin's responsibility."""
        self.ensure_loaded()
        return self._embed(list(chunks), sr)

    def _embed(self, chunks: list[np.ndarray], sr: int) -> list[np.ndarray]:
        raise NotImplementedError

    def describe(
        self, chunks: AudioChunks, sr: int, top_k: int = 5
    ) -> list[list[tuple[str, float]]] | None:
        """Return per-chunk human-readable tags ``[(label, score), ...]``, or
        None when the plugin cannot produce text."""
        if not self.provides_text:
            return None
        self.ensure_loaded()
        return self._describe(list(chunks), sr, top_k)

    def _describe(
        self, chunks: list[np.ndarray], sr: int, top_k: int
    ) -> list[list[tuple[str, float]]] | None:
        return None

    def unload(self) -> None:
        """Drop cached weights (best effort)."""
        self._loaded = False
