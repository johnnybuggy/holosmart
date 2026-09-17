"""Model plugin registry: singleton-per-process instances."""
from __future__ import annotations

import threading

from app.models.base import ModelPlugin
from app.models.clap_model import ClapPlugin
from app.models.fft_model import FftPlugin
from app.models.mert_model import Mert330Plugin, MertPlugin
from app.models.openl3_model import OpenL3Plugin

_lock = threading.Lock()
_instances: dict[str, ModelPlugin] = {}


def _plugin_classes() -> list[type[ModelPlugin]]:
    """All registered plugin classes, in canonical order."""
    return [ClapPlugin, MertPlugin, Mert330Plugin, OpenL3Plugin, FftPlugin]


def get_plugin(name: str) -> ModelPlugin:
    """Return the process-wide singleton plugin instance for ``name``.

    Raises KeyError with a friendly message listing known names.
    """
    with _lock:
        if not _instances:
            for cls in _plugin_classes():
                _instances[cls.name] = cls()
        if name not in _instances:
            known = ", ".join(sorted(_instances))
            raise KeyError(
                f"Unknown model plugin '{name}'. Known plugins: {known}")
        return _instances[name]


def list_plugins() -> list[ModelPlugin]:
    """All plugin singletons in canonical order (clap, mert, mert330,
    openl3, fft)."""
    with _lock:
        if not _instances:
            for cls in _plugin_classes():
                _instances[cls.name] = cls()
        return [_instances[cls.name] for cls in _plugin_classes()]


def plugin_info() -> list[dict]:
    """Summary dicts for the UI: name, display_name, embedding_dim,
    provides_text, available, error, loaded."""
    info: list[dict] = []
    for plugin in list_plugins():
        info.append({
            "name": plugin.name,
            "display_name": plugin.display_name,
            "embedding_dim": plugin.embedding_dim,
            "provides_text": plugin.provides_text,
            "available": plugin.is_available(),
            "error": plugin.availability_error(),
            "loaded": plugin.is_loaded,
        })
    return info
