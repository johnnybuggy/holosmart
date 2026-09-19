"""Similar results table: the first column shows the FULL path.

Follows the established fixture styles:
- library: a real temp ``app.db.database.Database`` plus the real
  ``app.db.repo`` functions with small synthetic numpy vectors — like
  tests/test_pareto.py's ParetoLibrary, trimmed to two tracks that each
  carry a ``clap`` track embedding;
- widget: an offscreen :class:`app.ui.detail_pane.DetailPane` receiving the
  real ``similar_tracks`` results.

Headless: no torch, no network; Qt offscreen only.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.similarity.search import SimilarResult, similar_tracks  # noqa: E402


class _TwoTrackLibrary:
    """Seed + neighbour track, each with a 'clap' track-level embedding."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.music_dir = self.root / "music"
        self.db = Database(self.root / "library.db")
        self.conn = self.db.connect()
        folder_id = repo.add_folder(self.conn, str(self.music_dir))
        self.seed_id = self._add_track(
            folder_id, "seed_song.wav", "Alpha", "Seed Song", [1.0, 0.0])
        self.near_id = self._add_track(
            folder_id, "near_song.wav", "Bravo", "Near Song", [0.9, 0.1])
        self.conn.commit()

    def _add_track(self, folder_id: int, filename: str, artist: str,
                   title: str, vector: list[float]) -> int:
        path = self.music_dir / filename
        track_id = repo.upsert_track(self.conn, folder_id, str(path), {
            "filename": filename, "artist": artist, "title": title,
        })
        repo.set_track_embedding(
            self.conn, track_id, "clap",
            np.asarray(vector, dtype=np.float32))
        return track_id


class SimilarPathColumnTests(unittest.TestCase):
    """The Similar table's first column carries res.path (with tooltip)."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="holosmart-similar-")
        self.addCleanup(self._tmp.cleanup)
        self.lib = _TwoTrackLibrary(Path(self._tmp.name))
        self.addCleanup(self.lib.conn.close)

    def _pane(self):
        from app.ui.detail_pane import DetailPane
        return DetailPane(self.lib.db, AppConfig())

    def test_first_column_shows_full_path_with_tooltip(self):
        results = similar_tracks(self.lib.conn, self.lib.seed_id,
                                 method="clap")
        # seed row (100 %) first, then the single match
        self.assertEqual([r.track_id for r in results],
                         [self.lib.seed_id, self.lib.near_id])
        self.assertEqual(results[0].score, 1.0)
        expected_path = str(self.lib.music_dir / "near_song.wav")

        pane = self._pane()
        pane.show_similar_results(results)

        item = pane._similar_table.item(1, 0)   # row 0 is the 100 % seed row
        self.assertIsNotNone(item)
        self.assertEqual(item.text(), expected_path)   # FULL path, not filename
        self.assertEqual(item.toolTip(), expected_path)
        seed_item = pane._similar_table.item(0, 0)
        self.assertEqual(seed_item.text(), str(self.lib.music_dir / "seed_song.wav"))

    def test_show_similar_error_still_clears_the_table(self):
        pane = self._pane()
        pane.show_similar_results([
            SimilarResult(track_id=self.lib.near_id,
                          path=str(self.lib.music_dir / "near_song.wav"),
                          filename="near_song.wav", artist="Bravo",
                          title="Near Song", score=0.9, method="clap")])
        self.assertEqual(pane._similar_table.rowCount(), 1)

        pane.show_similar_error("search blew up")

        self.assertEqual(pane._similar_table.rowCount(), 0)
        self.assertIsNone(pane._similar_table.item(0, 0))


if __name__ == "__main__":
    unittest.main()
