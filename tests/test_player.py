"""Built-in player + chunk click-to-play + results copy button."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from app.config import AppConfig
from app.db import repo
from app.db.database import Database


def _mk_wav(path: Path, seconds: float = 1.0) -> Path:
    sr = 8000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
    return path


class PlayerBarTests(unittest.TestCase):
    """PlayerBar logic: chunk seeks, chunk-end stop, UI state."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.wav = _mk_wav(self.dir / "song.wav")
        from PySide6.QtMultimedia import QMediaPlayer

        from app.ui.player import PlayerBar
        self.QMediaPlayer = QMediaPlayer
        self.bar = PlayerBar()
        self.seeks: list[int] = []
        self.plays: list[str] = []
        self.pauses: list[str] = []
        # record instead of driving the real media backend
        self.bar._player.setPosition = self.seeks.append
        self.bar._player.play = lambda *a, **k: self.plays.append("play")
        self.bar._player.pause = lambda *a, **k: self.pauses.append("pause")

    def tearDown(self):
        self.bar.setParent(None)
        del self.bar
        self._tmp.cleanup()

    def test_play_chunk_sets_bounded_state_and_label(self):
        self.bar.play_chunk(str(self.wav), 1.5, 3.0)
        self.assertEqual(self.bar._pending_start_ms, 1500)
        self.assertEqual(self.bar._chunk_end_ms, 3000)
        self.assertIn("song.wav", self.bar._now_label.text())
        self.assertIn("chunk", self.bar._now_label.text())

    def test_loaded_media_applies_pending_start_and_plays(self):
        self.bar.play_chunk(str(self.wav), 0.5, 1.0)
        self.assertEqual(self.seeks, [])
        self.bar._on_media_status(self.QMediaPlayer.MediaStatus.LoadedMedia)
        self.assertEqual(self.seeks, [500])
        self.assertEqual(self.plays, ["play"])
        self.assertIsNone(self.bar._pending_start_ms)

    def test_reclick_same_file_seeks_immediately(self):
        self.bar.play_chunk(str(self.wav), 0.5, 1.0)
        self.bar._on_media_status(self.QMediaPlayer.MediaStatus.LoadedMedia)
        self.seeks.clear()
        # same source: no fresh mediaStatusChanged will come
        self.bar.play_chunk(str(self.wav), 2.0, 2.5)
        self.assertEqual(self.seeks, [2000])
        self.assertEqual(self.plays, ["play", "play"])

    def test_reaching_chunk_end_pauses_at_the_boundary(self):
        self.bar.play_chunk(str(self.wav), 1.0, 2.0)
        self.bar._on_media_status(self.QMediaPlayer.MediaStatus.LoadedMedia)
        self.bar._on_position(1999)
        self.assertEqual(self.pauses, [])       # inside the chunk: keeps on
        self.bar._on_position(2000)
        self.assertEqual(self.pauses, ["pause"])
        self.assertEqual(self.seeks[-1], 2000)  # clamped to the chunk end
        self.assertIsNone(self.bar._chunk_end_ms)

    def test_whole_file_play_has_no_chunk_end(self):
        self.bar.play_file(str(self.wav))
        self.assertIsNone(self.bar._chunk_end_ms)
        self.assertIn("song.wav", self.bar._now_label.text())

    def test_stop_resets_everything(self):
        self.bar.play_chunk(str(self.wav), 0.5, 1.0)
        self.bar.stop()
        self.assertIsNone(self.bar._chunk_end_ms)
        self.assertIsNone(self.bar._pending_start_ms)
        self.assertEqual(self.bar._time_label.text(), "0:00 / 0:00")
        self.assertEqual(self.bar._now_label.text(), "")

    def test_volume_is_clamped(self):
        self.bar._on_volume_changed(250)
        self.assertEqual(self.bar._audio.volume(), 1.0)
        self.bar._on_volume_changed(-5)
        self.assertEqual(self.bar._audio.volume(), 0.0)

    def test_toggle_without_source_is_a_noop(self):
        self.bar.toggle_play_pause()          # no source yet: must not play
        self.assertEqual(self.plays, [])


class ChunkClickTests(unittest.TestCase):
    """Clicking a chunk row's number plays exactly that chunk."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = Database(self.dir / "lib.db")
        self.wav = _mk_wav(self.dir / "song.wav", seconds=2.0)
        with self.db.transaction() as conn:
            folder_id = repo.add_folder(conn, str(self.dir))
            self.track_id = repo.upsert_track(
                conn, folder_id, str(self.wav),
                {"filename": "song.wav", "extension": ".wav",
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 2.0,
                 "size_bytes": self.wav.stat().st_size})
            self.chunk_ids = repo.replace_chunks(
                conn, self.track_id,
                [(0, 0.0, 0.25), (1, 0.25, 0.5), (2, 0.5, 1.0)])

    def tearDown(self):
        self._tmp.cleanup()

    def _pane(self):
        from app.ui.detail_pane import DetailPane
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        return pane

    def test_click_chunk_number_emits_path_and_range(self):
        pane = self._pane()
        seen: list[tuple] = []
        pane.chunk_play_requested.connect(
            lambda path, start, end: seen.append((path, start, end)))
        pane._on_chunk_cell_clicked(1, 0)      # row 1 = chunk 0.25–0.5 s
        self.assertEqual(seen, [(str(self.wav), 0.25, 0.5)])

    def test_click_other_columns_does_not_play(self):
        pane = self._pane()
        seen: list[tuple] = []
        pane.chunk_play_requested.connect(
            lambda path, start, end: seen.append((path, start, end)))
        pane._on_chunk_cell_clicked(1, 2)      # End column: not the play cell
        self.assertEqual(seen, [])

    def test_play_affordance_tooltip_present(self):
        pane = self._pane()
        tip = pane._chunks_table.item(0, 0).toolTip()
        self.assertIn("Click to play this chunk", tip)


class CopyResultsTests(unittest.TestCase):
    """'Copy files…' copies every result file into the chosen folder."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = Database(self.dir / "lib.db")
        self.a = _mk_wav(self.dir / "a.wav")
        self.b = _mk_wav(self.dir / "b.wav")
        self.dest = self.dir / "copies"
        self.dest.mkdir()
        from app.ui.detail_pane import DetailPane
        self.pane = DetailPane(self.db, AppConfig())
        boxes: list[tuple] = []
        self.pane_boxes = boxes
        results = [
            SimpleNamespace(track_id=1, score=0.9, path=str(self.a),
                            artist="A", title="t1"),
            SimpleNamespace(track_id=2, score=0.8, path=str(self.b),
                            artist="B", title="t2"),
            SimpleNamespace(track_id=3, score=0.7, path=str(self.a),
                            artist="A", title="t1"),   # duplicate row
        ]
        self.pane.show_similar_results(results)

    def tearDown(self):
        self.pane.setParent(None)
        del self.pane
        self._tmp.cleanup()

    def _run_copy(self, chosen: str | None):
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory",
                   staticmethod(lambda *a, **k: chosen)), \
             patch("PySide6.QtWidgets.QMessageBox.information",
                   staticmethod(lambda *a, **k: self.pane_boxes.append(a))):
            self.pane._on_copy_results()

    def test_results_enable_copy_button(self):
        self.assertTrue(self.pane._copy_button.isEnabled())
        self.pane.show_similar_results([])
        self.assertFalse(self.pane._copy_button.isEnabled())

    def test_result_paths_deduplicate_rows(self):
        self.assertEqual(self.pane._result_paths(),
                         [str(self.a), str(self.b)])

    def test_copy_copies_skips_and_counts(self):
        # b.wav already copied identical: must be skipped, not rewritten
        import shutil
        shutil.copy2(self.b, self.dest / "b.wav")
        self._run_copy(str(self.dest))
        names = sorted(p.name for p in self.dest.iterdir())
        self.assertEqual(names, ["a.wav", "b.wav"])
        self.assertIn("1 file(s) copied", self.pane._similar_status.text())
        self.assertIn("1 already present", self.pane._similar_status.text())
        self.assertEqual(len(self.pane_boxes), 1)   # one summary dialog

    def test_colliding_different_file_is_renamed(self):
        import shutil
        # a DIFFERENT recording already occupying b.wav in the folder
        other = self.dir / "c.wav"
        _mk_wav(other, seconds=0.5)
        shutil.copy2(other, self.dest / "b.wav")
        self._run_copy(str(self.dest))
        names = sorted(p.name for p in self.dest.iterdir())
        self.assertEqual(names, ["a.wav", "b (2).wav", "b.wav"])

    def test_missing_source_is_counted_as_failed(self):
        self.pane._similar_table.item(0, 0).setText(str(self.dir / "gone.wav"))
        self._run_copy(str(self.dest))
        self.assertIn("1 failed", self.pane._similar_status.text())

    def test_cancelled_dialog_copies_nothing(self):
        self._run_copy("")     # empty → user cancelled
        self.assertEqual(list(self.dest.iterdir()), [])


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
