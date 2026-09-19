"""Quiet-presentation tests: long diagnostics move to tooltips, not layout.

Companion to the DetailPane warning-banner / chunks-hint / settings-dialog
quieting. Headless (QT_QPA_PLATFORM=offscreen).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402

LONG_REASON = ("MERT unavailable (Missing dependencies: torch, transformers. "
               "Install with: pip install torch transformers)")
LONG_NOTE = f"CLAP: 4 chunk embeddings, 40 tags; {LONG_REASON}"


class _OffscreenBase(unittest.TestCase):
    """Minimal db + wav-track fixture (mirrors test_ui.UiTestBase)."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = Database(self.dir / "lib.db")
        self.wav = self.dir / "song.wav"
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(self.wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            folder_id = repo.add_folder(conn, str(self.dir))
            self.track_id = repo.upsert_track(
                conn, folder_id, str(self.wav),
                {"filename": "song.wav", "extension": ".wav",
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": self.wav.stat().st_size})

    def tearDown(self):
        self._tmp.cleanup()

    def _set_message(self, status: str, message: str) -> None:
        with self.db.transaction() as conn:
            repo.set_track_status(conn, self.track_id, status, message)


class TestQuietWarningBanner(_OffscreenBase):
    """The banner stays quiet; full diagnostics are one hover away."""

    def test_long_unavailable_note_collapses_to_short_line(self):
        from app.ui.detail_pane import DetailPane
        self._set_message("analyzed", LONG_NOTE)
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        self.assertTrue(pane._warning.isVisibleTo(pane))
        # quiet: short wrapped line, height-capped, full text on hover
        self.assertLess(len(pane._warning.text()), 200)
        self.assertTrue(pane._warning.wordWrap())
        self.assertLess(pane._warning.maximumHeight(), 200)
        self.assertIn(LONG_REASON, pane._warning.toolTip())

    def test_multiple_warnings_show_summary_with_full_tooltip(self):
        from app.ui.detail_pane import DetailPane
        self._set_message("analyzed", "CLAP failed: bad audio; " + LONG_NOTE)
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        self.assertTrue(pane._warning.isVisibleTo(pane))
        self.assertIn("2 model warnings", pane._warning.text())
        self.assertNotIn("Missing dependencies", pane._warning.text())
        self.assertIn("CLAP failed: bad audio", pane._warning.toolTip())
        self.assertIn("Missing dependencies: torch", pane._warning.toolTip())

    def test_hard_error_keeps_first_line_and_full_tooltip(self):
        from app.ui.detail_pane import DetailPane
        message = "Decode failed: " + "x" * 300
        self._set_message("error", message)
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        self.assertTrue(pane._warning.isVisibleTo(pane))
        # the user must still see what failed, but truncated and muted
        self.assertIn("Decode failed", pane._warning.text())
        self.assertLess(len(pane._warning.text()), 200)
        self.assertEqual(pane._warning.toolTip(), "⚠ " + message)


class TestQuietChunksTab(_OffscreenBase):
    """Per-model "no results" reasons live in header tooltips, not the hint."""

    def _make_analyzed_track(self):
        with self.db.transaction() as conn:
            chunk_ids = repo.replace_chunks(conn, self.track_id,
                                            [(0, 0.0, 0.5), (1, 0.25, 0.75)])
            repo.add_chunk_embedding(conn, chunk_ids[0], "clap",
                                     np.linspace(0.1, 0.4, 8, dtype=np.float32))
            repo.add_chunk_embedding(conn, chunk_ids[1], "clap",
                                     np.linspace(0.2, 0.5, 8, dtype=np.float32))
        self._set_message("analyzed", LONG_NOTE)

    def test_chunks_hint_stays_short_and_header_carries_reason(self):
        from app.ui.detail_pane import DetailPane
        self._make_analyzed_track()
        config = AppConfig()
        config.models = ["clap", "mert"]
        pane = DetailPane(self.db, config)
        pane.show_track(self.track_id)
        hint = pane._chunks_hint.text()
        self.assertTrue(pane._chunks_hint.wordWrap())
        # the hint stays short: no per-model diagnostics concatenated
        self.assertNotIn("unavailable", hint)
        self.assertNotIn("Missing dependencies", hint)
        self.assertNotIn("no results", hint)
        table = pane._chunks_table
        headers = {table.horizontalHeaderItem(c).text(): c
                   for c in range(table.columnCount())}
        self.assertIn("MERT", headers)
        tip = table.horizontalHeaderItem(headers["MERT"]).toolTip()
        self.assertIn("no results", tip)
        self.assertIn(LONG_REASON, tip)
        # compact marker in every cell of the model-less column
        for r in range(table.rowCount()):
            self.assertEqual(table.item(r, headers["MERT"]).text(), "⚠")


class TestQuietMetaNote(_OffscreenBase):
    """The meta table truncates the analysis note; hover shows everything."""

    def test_analysis_note_cell_truncated_with_full_tooltip(self):
        from app.ui.detail_pane import DetailPane
        self._set_message("analyzed", LONG_NOTE)
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        table = pane._meta_table
        row = next(r for r in range(table.rowCount())
                   if table.item(r, 0).text() == "Analysis note")
        cell = table.item(row, 1)
        self.assertLessEqual(len(cell.text()), 101)
        self.assertTrue(cell.text().endswith("…"))
        self.assertEqual(cell.toolTip(), LONG_NOTE)


class TestQuietSettingsCheckboxes(unittest.TestCase):
    """Model rows show a short "(unavailable)" suffix + tooltip with the reason."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_unavailable_checkbox_is_short_with_tooltip(self):
        from app.ui.settings_dialog import SettingsDialog
        error = ("Missing dependencies: torch, transformers. "
                 "Install with: pip install torch transformers")
        info = {"name": "mert", "display_name": "MERT", "embedding_dim": 768,
                "provides_text": False, "available": False, "error": error,
                "loaded": False}
        with mock.patch("app.ui.settings_dialog.plugin_info",
                        return_value=[dict(info)]):
            dialog = SettingsDialog(AppConfig())
        check = dialog._model_checks["mert"]
        self.assertIn("MERT — dim 768", check.text())
        self.assertIn("(unavailable)", check.text())
        self.assertLess(len(check.text()), 60)
        self.assertNotIn("Install with", check.text())
        self.assertEqual(check.toolTip(), error)
        self.assertFalse(check.isEnabled())

    def test_real_unavailable_plugins_keep_quiet_rows(self):
        from app.models.registry import plugin_info

        from app.ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(AppConfig())
        for info in plugin_info():
            if info["available"]:
                continue
            check = dialog._model_checks[info["name"]]
            self.assertIn("(unavailable)", check.text())
            self.assertLess(len(check.text()), 80)
            self.assertEqual(check.toolTip(), info["error"])


if __name__ == "__main__":
    unittest.main()
