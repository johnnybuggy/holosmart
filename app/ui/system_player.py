"""Open files and playlists in the system-wide default music player.

Kept free of any player-specific logic on purpose: the OS decides which
application handles ``.m3u`` playlists and audio files (Music.app on macOS,
the default media player on Windows/Linux).  Uses
``QDesktopServices.openUrl`` so the behavior matches what double-clicking
the file in Finder/Explorer would do.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

__all__ = ["open_in_system_player"]


def open_in_system_player(path: str | Path) -> bool:
    """Ask the OS to open *path* with its default (music) player.

    Returns ``True`` when the system accepted the request.  A ``False``
    return means no default application is registered for the file type —
    callers should show a hint rather than an error dialog, since this is a
    best-effort convenience.
    """
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path))))
