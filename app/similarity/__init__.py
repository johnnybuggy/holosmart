"""Similarity search for HoloSmart Music Explorer.

Modules:
    ``app.similarity.ollama`` — Ollama HTTP client for text embeddings
        (:class:`~app.similarity.ollama.OllamaClient`,
        :func:`~app.similarity.ollama.detect_ollama`).
    ``app.similarity.search`` — cosine math, per-track centroid/-description
        embeddings and :func:`~app.similarity.search.similar_tracks`.

Import submodules explicitly (this package stays import-light on purpose so
that a missing optional dependency such as ``app.db.repo`` or a down Ollama
server never breaks sibling imports).
"""
from __future__ import annotations

__all__ = ["ollama", "search"]
