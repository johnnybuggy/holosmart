"""Playlist generation and export for HoloSmart Music Explorer.

``app.playlist.generator`` turns :func:`app.similarity.search.similar_tracks`
results into a persisted playlist (SQLite) and exports it as an extended
``.m3u`` file. Imported lazily by callers; this package stays import-light.
"""
from __future__ import annotations

__all__ = ["generator"]
