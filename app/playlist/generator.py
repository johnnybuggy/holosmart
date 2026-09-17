"""Playlist generation from similarity results + .m3u export."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from app.db import repo
from app.similarity.ollama import OllamaClient
from app.similarity.search import similar_tracks

log = logging.getLogger(__name__)


def default_m3u_path(name: str) -> Path:
    """Target path for one-click playlist exports: ``data/playlists``.

    The playlist name is slugified into the file name; when the file already
    exists a ``-2``/``-3``/… suffix is appended so a new export never
    silently overwrites an earlier one.  The directory is created lazily by
    :func:`export_m3u`.
    """
    from app.config import DATA_DIR  # local: keeps module import cheap

    slug = re.sub(r"[^\w-]+", "_", name.strip()).strip("_") or "playlist"
    base = DATA_DIR / "playlists"
    candidate = base / f"{slug}.m3u"
    counter = 2
    while candidate.exists():
        candidate = base / f"{slug}-{counter}.m3u"
        counter += 1
    return candidate


def generate_playlist(conn, name: str, seed_track_id: int, limit: int = 15,
                      method: str = "auto",
                      ollama: OllamaClient | None = None) -> int:
    """Create a playlist from the tracks most similar to ``seed_track_id``.

    Returns the new playlist id. Raises RuntimeError when no similar tracks
    exist (message is user-presentable).
    """
    results = similar_tracks(conn, seed_track_id, method=method, limit=limit,
                             ollama=ollama)
    playlist_id = repo.create_playlist(conn, name=name,
                                       seed_track_id=seed_track_id,
                                       method=results[0].method if results else method)
    repo.add_playlist_items(conn, playlist_id,
                            [(r.track_id, r.score) for r in results])
    log.info("Playlist '%s' created with %d tracks (method=%s)",
             name, len(results), results[0].method if results else method)
    return playlist_id


def export_m3u(conn, playlist_id: int, out_path: str | Path) -> str:
    """Write the playlist as an .m3u file with absolute paths. Returns the path."""
    playlist = repo.get_playlist(conn, playlist_id)
    if playlist is None:
        raise RuntimeError(f"Playlist {playlist_id} not found")
    items = repo.get_playlist_items(conn, playlist_id)
    lines = ["#EXTM3U"]
    for item in items:
        duration = item["duration_sec"]
        dur = int(round(float(duration))) if duration is not None else -1
        artist = item["artist"] or "Unknown artist"
        title = item["title"] or item["filename"]
        lines.append(f"#EXTINF:{dur},{artist} - {title}")
        raw_path = str(item["path"])
        path = Path(raw_path)
        # Library paths are absolute already; absolutize defensively anyway.
        lines.append(raw_path if path.is_absolute() else str(Path.cwd() / path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Exported playlist %s to %s", playlist_id, out)
    return str(out)
