"""Built-in audio player: compact bar docked under the main splitter.

Plays whole tracks and — the point of the feature — single CHUNKS: the
Chunks tab emits ``chunk_play_requested(path, start_sec, end_sec)`` when a
chunk row's index cell is clicked and the main window forwards it here.
Chunked playback seeks to the chunk's start and stops at its end, so
listening to one chunk is exactly that, not the rest of the song.

The Qt multimedia backend (FFmpeg) is imported lazily; environments
without it degrade to a disabled bar with a notice instead of crashing
the app.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import ( QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                                QSlider, QVBoxLayout, QWidget)

__all__ = ["PlayerBar"]


def _fmt_ms(ms: float) -> str:
    """``83_000`` → ``1:23`` (minutes:seconds, no fractions)."""
    total = max(0, int(round(ms / 1000.0)))
    return f"{total // 60}:{total % 60:02d}"


class PlayerBar(QWidget):
    """One-row player: chunk / play-pause / stop / seek / volume."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._chunk_end_ms: int | None = None   # bounded (chunk) playback
        self._pending_start_ms: int | None = None
        self._user_seeking = False
        self._available = True
        self._player = None
        self._audio = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._now_label = QLabel("")
        self._now_label.setToolTip(
            "Click a chunk's number in the Chunks tab to play exactly "
            "that chunk; use Play/Pause to resume from where you stopped.")
        self._now_label.setMinimumWidth(180)
        layout.addWidget(self._now_label, 0)

        try:   # lazy backend: never break startup when multimedia is absent
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            self._player = QMediaPlayer(self)
            self._audio = QAudioOutput(self)
            self._player.setAudioOutput(self._audio)
            self._audio.setVolume(0.8)
        except Exception as exc:   # pragma: no cover - backend-dependent
            self._available = False
            layout.addWidget(QLabel(f"Audio unavailable: {exc}"))
            return

        self._toggle_button = QPushButton("▶")
        self._toggle_button.setFixedWidth(32)
        self._toggle_button.setToolTip("Play / pause (space between chunks "
                                       "on repeated chunk clicks).")
        self._toggle_button.clicked.connect(self.toggle_play_pause)
        self._toggle_button.setEnabled(False)
        layout.addWidget(self._toggle_button, 0)

        self._stop_button = QPushButton("⏹")
        self._stop_button.setFixedWidth(32)
        self._stop_button.setToolTip("Stop playback and rewind the seek bar.")
        self._stop_button.clicked.connect(self.stop)
        self._stop_button.setEnabled(False)
        layout.addWidget(self._stop_button, 0)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setToolTip("Seek position.")
        self._slider.setEnabled(False)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self._slider, 1)

        self._time_label = QLabel("0:00 / 0:00")
        layout.addWidget(self._time_label, 0)

        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setRange(0, 100)
        self._volume.setValue(80)
        self._volume.setFixedWidth(90)
        self._volume.setToolTip("Player volume.")
        self._volume.valueChanged.connect(self._on_volume_changed)
        layout.addWidget(self._volume, 0)

        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_playback_state)

    # ---- public API ---------------------------------------------------------
    @property
    def available(self) -> bool:
        """False only when the multimedia backend could not be loaded."""
        return self._available

    def play_chunk(self, path: str, start_sec: float, end_sec: float) -> None:
        """Play exactly ``[start_sec, end_sec]`` of ``path`` (a chunk)."""
        if not self._available:
            return
        self._load(path, start_sec * 1000.0, end_sec * 1000.0,
                   f"chunk {_fmt_ms(start_sec * 1000.0)} – "
                   f"{_fmt_ms(end_sec * 1000.0)}")

    def play_file(self, path: str, label: str | None = None) -> None:
        """Play the whole file."""
        if not self._available:
            return
        self._load(path, None, None,
                   label or Path(path).name)

    def toggle_play_pause(self) -> None:
        if not self._available or not self._player.source().isValid():
            return
        if self._player.playbackState() == \
                self._player.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def stop(self) -> None:
        if not self._available:
            return
        self._player.stop()
        self._player.setSource(QUrl())
        self._chunk_end_ms = None
        self._pending_start_ms = None
        self._slider.setRange(0, 0)
        self._time_label.setText("0:00 / 0:00")
        self._now_label.setText("")

    # ---- internals ----------------------------------------------------------
    def _load(self, path: str, start_ms: float | None, end_ms: float | None,
              label: str) -> None:
        filename = Path(path).name
        suffix = f" — {label}" if label != filename else ""
        self._now_label.setText(f"{filename}{suffix}")
        self._pending_start_ms = int(start_ms) if start_ms is not None \
            else None
        self._chunk_end_ms = int(end_ms) if end_ms is not None else None
        # An already-loaded source does NOT re-emit mediaStatusChanged, so
        # the pending seek must be applied immediately in that case.
        already_loaded = (self._player.source() ==
                          QUrl.fromLocalFile(str(path)))
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        if already_loaded:
            self._apply_pending_start()

    def _apply_pending_start(self) -> None:
        if self._pending_start_ms is not None:
            self._player.setPosition(self._pending_start_ms)
            self._pending_start_ms = None
        self._player.play()

    def _on_media_status(self, status) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._apply_pending_start()
        elif status in (QMediaPlayer.MediaStatus.InvalidMedia,
                        QMediaPlayer.MediaStatus.EndOfMedia):
            self._chunk_end_ms = None

    def _on_position(self, ms: int) -> None:
        if not self._user_seeking:
            self._slider.setValue(ms)
        end = self._chunk_end_ms
        if end is not None and ms >= end:
            # reached the chunk's end: stop instead of playing on
            self._chunk_end_ms = None
            self._player.pause()
            self._player.setPosition(int(end))
            return
        self._update_time_label(ms)

    def _on_duration(self, duration_ms: int) -> None:
        self._slider.setRange(0, max(0, int(duration_ms)))
        self._slider.setEnabled(True)
        self._update_time_label(self._player.position())

    def _on_playback_state(self, state) -> None:
        playing = state == self._player.PlaybackState.PlayingState
        self._toggle_button.setText("⏸" if playing else "▶")
        self._toggle_button.setEnabled(True)
        self._stop_button.setEnabled(True)

    def _update_time_label(self, ms: int) -> None:
        self._time_label.setText(f"{_fmt_ms(ms)} / "
                                 f"{_fmt_ms(self._player.duration())}")

    def _on_slider_pressed(self) -> None:
        self._user_seeking = True

    def _on_slider_released(self) -> None:
        self._user_seeking = False
        self._player.setPosition(self._slider.value())

    def _on_slider_moved(self, value: int) -> None:
        if self._user_seeking:
            self._update_time_label(value)

    def _on_volume_changed(self, value: int) -> None:
        if self._audio is not None:
            self._audio.setVolume(max(0.0, min(1.0, value / 100.0)))
