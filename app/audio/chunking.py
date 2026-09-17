"""Chunking of decoded audio into overlapping windows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    """One audio chunk: index, time range in seconds, raw mono samples."""
    idx: int
    start_sec: float
    end_sec: float
    samples: np.ndarray  # float32 1-D


def chunk_audio(samples: np.ndarray, sr: int, chunk_seconds: float = 20.0,
                overlap_percent: float = 50.0) -> list[Chunk]:
    """Split ``samples`` (1-D float32) into overlapping chunks.

    hop = chunk_seconds * (1 - overlap_percent / 100). A final partial chunk is
    kept when it covers at least 25 % of ``chunk_seconds``; otherwise it is
    merged into the previous chunk (extending it to the end of the audio).
    Audio shorter than one chunk yields a single shorter chunk.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    overlap = min(max(float(overlap_percent), 0.0), 95.0)
    hop_len = max(1, int(round(chunk_seconds * (1.0 - overlap / 100.0) * sr)))
    chunk_len = max(1, int(round(chunk_seconds * sr)))
    min_tail = max(1, int(round(chunk_seconds * 0.25 * sr)))

    n = data.shape[0]
    segments: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_len, n)
        segments.append((start, end))
        if end >= n:
            break
        start += hop_len

    if len(segments) >= 2 and (segments[-1][1] - segments[-1][0]) < min_tail:
        prev_start = segments[-2][0]
        segments[-2] = (prev_start, segments[-1][1])
        segments.pop()

    return [
        Chunk(i, s / sr, e / sr, data[s:e])
        for i, (s, e) in enumerate(segments)
    ]


def format_duration(seconds: float | None) -> str:
    """Human duration: 'mm:ss' below an hour, else 'hh:mm:ss'. None -> '?'."""
    if seconds is None:
        return "?"
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_size(nbytes: int | None) -> str:
    """Human size: '1.2 MB' style. None -> '?'."""
    if nbytes is None:
        return "?"
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"
