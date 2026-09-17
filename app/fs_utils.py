"""Filesystem helpers: music discovery, access checks, macOS privacy guidance.

macOS TCC (Transparency, Consent and Control) protects folders such as Desktop,
Documents and Downloads. Reading them without user consent raises
``PermissionError`` (errno 1). The very first access attempt from this process
normally triggers the macOS permission prompt; after a "Don't Allow" answer the
failures are silent, and access can only be restored in System Settings.
The helpers here (a) enumerate music files while surviving partially-denied
directory trees, and (b) give the UI the pieces to explain and fix the situation.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.config import SUPPORTED_EXTENSIONS

#: User-facing explanation shown when a folder cannot be read.
ACCESS_HELP_TEXT = (
    "macOS protects folders such as Desktop, Documents and Downloads. To grant access:\n"
    "  1. Open System Settings \u2192 Privacy & Security \u2192 Files and Folders "
    "(or Full Disk Access).\n"
    "  2. Enable access for the application that launched HoloSmart "
    "(e.g. Terminal, iTerm, VS Code).\n"
    "  3. Retry the scan.\n"
    "The first access attempt normally shows a macOS permission prompt \u2014 if "
    "'Don't Allow' was clicked before, access can only be restored in System Settings."
)


def check_read_access(path: str | Path) -> tuple[bool, str]:
    """Return ``(ok, error_message)`` for reading a directory.

    Calling this from the GUI process is also what *triggers* the macOS
    permission prompt on first access.
    """
    try:
        os.listdir(str(path))
        return True, ""
    except PermissionError as exc:
        return False, f"Permission denied: {exc.strerror or exc}"
    except OSError as exc:
        return False, str(exc)


def walk_music_files(root: str | Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """Recursively collect music files under ``root``.

    Returns ``(files, denied)`` where ``files`` is a sorted list of music-file
    paths and ``denied`` lists ``(directory, error_message)`` for every directory
    that could not be read (e.g. macOS-protected or chmod-000). Accessible
    sub-trees are still scanned even when sibling directories are denied.
    Symlinked directories are not followed (loop safety).
    """
    files: list[Path] = []
    denied: list[tuple[str, str]] = []

    def recurse(directory: str) -> None:
        try:
            entries = list(os.scandir(directory))
        except PermissionError as exc:
            denied.append((directory, f"Permission denied: {exc.strerror or exc}"))
            return
        except OSError as exc:
            denied.append((directory, str(exc)))
            return
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    recurse(entry.path)
                elif entry.is_file() and \
                        Path(entry.name).suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(Path(entry.path))
            except OSError as exc:  # entry vanished or is unreadable
                denied.append((entry.path, str(exc)))

    root_str = str(root)
    if Path(root_str).is_file():
        if Path(root_str).suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(Path(root_str))
        return files, denied
    recurse(root_str)
    files.sort()
    return files, denied


def open_privacy_settings() -> bool:
    """Open the macOS Privacy & Security pane (Full Disk Access tab).

    Returns True when a settings pane was opened, False on other platforms.
    """
    if sys.platform == "darwin":
        subprocess.run(
            ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"],
            check=False,
        )
        return True
    return False


def reveal_in_finder(path: str | Path) -> bool:
    """Reveal a folder/file in Finder (macOS) or the platform file manager."""
    target = Path(path)
    if not target.exists():
        return False
    if sys.platform == "darwin":
        subprocess.run(["open", str(target)], check=False)
        return True
    if sys.platform == "win32":  # pragma: no cover - platform specific
        subprocess.run(["explorer", str(target)], check=False)
        return True
    subprocess.run(["xdg-open", str(target)], check=False)  # pragma: no cover
    return True
