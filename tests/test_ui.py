"""Headless GUI tests (QT_QPA_PLATFORM=offscreen)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402


class UiTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db_path = self.dir / "lib.db"
        self.db = Database(self.db_path)

        self.wav = self.dir / "song.wav"
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(self.wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)

        with self.db.transaction() as conn:
            self.folder_id = repo.add_folder(conn, str(self.dir))
            self.track_id = repo.upsert_track(
                conn, self.folder_id, str(self.wav),
                {"filename": "song.wav", "extension": ".wav", "codec": "pcm_s16le",
                 "sample_rate": 8000, "channels": 1, "duration_sec": 1.0,
                 "size_bytes": self.wav.stat().st_size})

    def tearDown(self):
        self._tmp.cleanup()


class TestFolderTree(UiTestBase):
    def _add_track(self, rel_path: str) -> int:
        """Create a real wav at self.dir/<rel_path> and index it in the db."""
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": path.stat().st_size})

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _dir_item(self, tree, full_path: str):
        from app.ui.folder_tree import DIR_ROLE
        for item in self._walk(tree):
            if item.data(0, DIR_ROLE) == str(full_path):
                return item
        return None

    def _track_item(self, tree, track_id: int):
        from app.ui.folder_tree import TRACK_ROLE
        for item in self._walk(tree):
            if item.data(0, TRACK_ROLE) == track_id:
                return item
        return None

    def test_refresh_and_selection(self):
        from app.ui.folder_tree import FolderTree
        tree = FolderTree(self.db)
        tree.refresh()
        self.assertEqual(tree.topLevelItemCount(), 1)
        folder_item = tree.topLevelItem(0)
        self.assertEqual(folder_item.text(0), str(self.dir))  # top level = folder path
        # a file directly in the root is a direct child (no synthetic dir)
        self.assertEqual(folder_item.childCount(), 1)
        self.assertIn("song.wav", folder_item.child(0).text(0))
        self.assertIsNone(folder_item.child(0).data(0, 0x0100 + 3))  # no DIR_ROLE
        self.assertEqual(folder_item.child(0).data(0, 0x0100 + 2), self.track_id)
        # programmatic selection triggers selection
        tree.select_track(self.track_id)
        self.assertEqual(tree.selected_track_id(), self.track_id)
        self.assertEqual(tree.selected_folder_id(), self.folder_id)
        self.assertEqual(tree.selected_track_ids_in_folder(), [self.track_id])

    def test_status_glyphs(self):
        from app.ui.folder_tree import FolderTree
        with self.db.transaction() as conn:
            repo.set_track_status(conn, self.track_id, "analyzed", "done")
        tree = FolderTree(self.db)
        tree.refresh()
        self.assertEqual(tree.topLevelItem(0).child(0).text(1), "✓")

    def test_nested_hierarchy(self):
        from app.ui.folder_tree import DIR_ROLE, FOLDER_ROLE, FolderTree
        deep_id = self._add_track("artists/Alpha/2024/live_set.wav")
        self._add_track("ambient/drift.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        root = tree.topLevelItem(0)
        # directories first (case-insensitive), then the root-level file;
        # no phantom nodes for empty directories
        self.assertEqual([root.child(i).text(0) for i in range(root.childCount())],
                         ["ambient", "artists", "song.wav"])
        artists = root.child(1)
        self.assertEqual(artists.text(0), "artists")      # name, not full path
        self.assertEqual(artists.text(1), "0%")     # progress percentage
        self.assertEqual(artists.toolTip(0), str(self.dir / "artists"))
        alpha = artists.child(0)
        self.assertEqual(alpha.text(0), "Alpha")
        year = alpha.child(0)
        self.assertEqual(year.text(0), "2024")
        self.assertEqual(year.toolTip(0), str(self.dir / "artists" / "Alpha" / "2024"))
        track_item = year.child(0)
        self.assertEqual(track_item.text(0), "live_set.wav")
        self.assertEqual(track_item.data(0, 0x0100 + 2), deep_id)
        # subfolder nodes carry the full path, never a db folder id
        self.assertIsNone(artists.data(0, FOLDER_ROLE))
        self.assertEqual(artists.data(0, DIR_ROLE), str(self.dir / "artists"))

    def test_selected_folder_id_from_nested_file(self):
        from app.ui.folder_tree import FolderTree
        deep_id = self._add_track("artists/Alpha/2024/live_set.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        tree.select_track(deep_id)
        self.assertEqual(tree.selected_track_id(), deep_id)
        self.assertEqual(tree.selected_folder_id(), self.folder_id)

    def test_select_track_nested_expands_ancestors(self):
        from app.ui.folder_tree import FolderTree
        deep_id = self._add_track("artists/Alpha/2024/live_set.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        root = tree.topLevelItem(0)
        self.assertTrue(root.isExpanded())        # expandToDepth(0) after refresh
        self.assertFalse(root.child(1).isExpanded())  # "artists" collapsed
        tree.select_track(deep_id)
        self.assertEqual(tree.selected_track_id(), deep_id)
        artists = self._dir_item(tree, self.dir / "artists")
        year = self._dir_item(tree, self.dir / "artists" / "Alpha" / "2024")
        self.assertTrue(artists.isExpanded())
        self.assertTrue(year.isExpanded())

    def test_track_ids_in_subfolder_subtree(self):
        from app.ui.folder_tree import FolderTree
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/deep/b.wav")
        c_id = self._add_track("jazz/c.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        rock = self._dir_item(tree, self.dir / "rock")
        tree.setCurrentItem(rock)
        self.assertEqual(tree.selected_track_id(), None)   # folder, not a file
        self.assertEqual(tree.selected_folder_id(), self.folder_id)
        self.assertEqual(sorted(tree.selected_track_ids_in_folder()),
                         sorted([a_id, b_id]))
        # a selected file yields only itself
        tree.select_track(c_id)
        self.assertEqual(tree.selected_track_ids_in_folder(), [c_id])

    def test_search_filters_by_file_partial_match(self):
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/deep/b_all.wav")
        c_id = self._add_track("jazz/c.wav")
        from app.ui.folder_tree import LibraryPane
        pane = LibraryPane(self.db)
        pane.tree.refresh()
        root = pane.tree.topLevelItem(0)
        pane.apply_filter("b_all")
        self.assertFalse(root.isHidden())          # ancestor of the match
        rock = self._dir_item(pane.tree, self.dir / "rock")
        deep = self._dir_item(pane.tree, self.dir / "rock" / "deep")
        self.assertFalse(rock.isHidden())
        self.assertTrue(rock.isExpanded())         # auto-expanded ancestors
        self.assertFalse(deep.isHidden())
        self.assertTrue(deep.isExpanded())
        self.assertFalse(self._track_item(pane.tree, b_id).isHidden())
        self.assertTrue(self._dir_item(pane.tree, self.dir / "jazz").isHidden())
        self.assertTrue(self._track_item(pane.tree, a_id).isHidden())
        self.assertTrue(self._track_item(pane.tree, c_id).isHidden())
        self.assertTrue(self._track_item(pane.tree, self.track_id).isHidden())

    def test_search_folder_match_keeps_subtree(self):
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/deep/b.wav")
        c_id = self._add_track("jazz/c.wav")
        from app.ui.folder_tree import LibraryPane
        pane = LibraryPane(self.db)
        pane.tree.refresh()
        pane.apply_filter("roc")
        rock = self._dir_item(pane.tree, self.dir / "rock")
        self.assertFalse(rock.isHidden())
        self.assertTrue(rock.isExpanded())
        # whole subtree of the matching folder stays reachable
        self.assertFalse(self._track_item(pane.tree, a_id).isHidden())
        self.assertFalse(self._dir_item(pane.tree, self.dir / "rock" / "deep").isHidden())
        self.assertFalse(self._track_item(pane.tree, b_id).isHidden())
        self.assertTrue(self._dir_item(pane.tree, self.dir / "jazz").isHidden())
        self.assertTrue(self._track_item(pane.tree, c_id).isHidden())
        self.assertTrue(self._track_item(pane.tree, self.track_id).isHidden())

    def test_search_no_match_hides_and_empty_restores(self):
        self._add_track("rock/a.wav")
        from app.ui.folder_tree import LibraryPane
        pane = LibraryPane(self.db)
        pane.tree.refresh()
        pane.apply_filter("zzzz_nothing")
        root = pane.tree.topLevelItem(0)
        self.assertTrue(root.isHidden())
        self.assertTrue(self._dir_item(pane.tree, self.dir / "rock").isHidden())
        # empty query restores the refresh state: everything visible,
        # root expanded, deeper levels collapsed
        pane.apply_filter("")
        self.assertFalse(root.isHidden())
        self.assertFalse(self._dir_item(pane.tree, self.dir / "rock").isHidden())
        self.assertTrue(root.isExpanded())
        self.assertFalse(self._dir_item(pane.tree, self.dir / "rock").isExpanded())

    def test_search_line_edit_filters_live(self):
        drift_id = self._add_track("ambient/drift.wav")
        from PySide6.QtTest import QTest

        from app.ui.folder_tree import LibraryPane
        pane = LibraryPane(self.db)
        pane.tree.refresh()
        self.assertEqual(pane._filter_edit.placeholderText(),
                         "Filter files and folders…")
        pane._filter_edit.setText("drift")   # live on textChanged (debounced)
        QTest.qWait(350)                     # > 200 ms debounce interval
        self.assertFalse(self._track_item(pane.tree, drift_id).isHidden())
        self.assertTrue(self._track_item(pane.tree, self.track_id).isHidden())


class TestDetailPane(UiTestBase):
    def test_show_track_populates_overview_and_chunks(self):
        from app.ui.detail_pane import DetailPane
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        # duration row should show human duration
        overview = pane.widget(0)
        table = overview.findChild(type(pane._meta_table))
        self.assertIsNotNone(table)
        pane.show_track(None)

    def test_chunks_tab_groups_results_under_track(self):
        from app.ui.detail_pane import DetailPane
        with self.db.transaction() as conn:
            chunk_ids = repo.replace_chunks(conn, self.track_id,
                                            [(0, 0.0, 0.5), (1, 0.25, 0.75)])
            repo.add_chunk_embedding(conn, chunk_ids[0], "clap",
                                     np.linspace(0.1, 0.4, 8, dtype=np.float32))
            repo.add_chunk_embedding(conn, chunk_ids[1], "clap",
                                     np.linspace(0.2, 0.5, 8, dtype=np.float32))
            repo.add_chunk_tags(conn, chunk_ids[0], "clap",
                                [("rock", 0.9), ("pop", 0.1)])
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        table = pane._chunks_table
        self.assertEqual(table.rowCount(), 2)
        headers = [table.horizontalHeaderItem(c).text()
                   for c in range(table.columnCount())]
        # model columns must show display names ("CLAP"), not plugin ids ("clap")
        self.assertIn("CLAP", headers)
        self.assertNotIn("clap", headers)
        self.assertIn("rock", table.item(0, 3).text())
        clap_cell = table.item(0, headers.index("CLAP")).text()
        self.assertTrue(clap_cell.startswith("dim=8"), clap_cell)

    def test_playlist_tab_roundtrip(self):
        from app.ui.detail_pane import DetailPane
        with self.db.transaction() as conn:
            pid = repo.create_playlist(conn, "Test mix", seed_track_id=self.track_id)
            repo.add_playlist_items(conn, pid, [(self.track_id, 0.87)])
        pane = DetailPane(self.db, AppConfig())
        pane.refresh_playlists(select_id=pid)
        self.assertEqual(pane._playlist_list.count(), 1)
        self.assertEqual(pane._playlist_items.rowCount(), 1)
        self.assertIn("87", pane._playlist_items.item(0, 3).text())


class TestDetailPaneWarnings(UiTestBase):
    """Per-model analysis problems must be visible in the UI."""

    def _set_message(self, status: str, message: str) -> None:
        with self.db.transaction() as conn:
            repo.set_track_status(conn, self.track_id, status, message)

    def test_error_status_shows_warning_banner(self):
        from app.ui.detail_pane import DetailPane
        self._set_message("error", "Decode failed: broken file")
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        self.assertTrue(pane._warning.isVisibleTo(pane))
        # quiet banner: the hard error keeps its first line visible and the
        # full message also lands in the tooltip
        self.assertIn("Decode failed", pane._warning.text())
        self.assertIn("Decode failed: broken file", pane._warning.toolTip())

    def test_model_failure_note_shows_warning_banner(self):
        from app.ui.detail_pane import DetailPane
        self._set_message("analyzed",
                          "CLAP: 4 chunk embeddings, 40 tags; "
                          "MERT failed: Output channels > 65536 not supported")
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        self.assertTrue(pane._warning.isVisibleTo(pane))
        # quiet banner: short visible line, the full reason moved to the tooltip
        self.assertLess(len(pane._warning.text()), 200)
        self.assertIn("MERT failed: Output channels > 65536 not supported",
                      pane._warning.toolTip())

    def test_clean_analysis_has_no_banner(self):
        from app.ui.detail_pane import DetailPane
        self._set_message("analyzed", "CLAP: 4 chunk embeddings, 40 tags")
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        self.assertFalse(pane._warning.isVisibleTo(pane))

    def test_chunks_hint_warns_about_missing_model_results(self):
        from app.ui.detail_pane import DetailPane
        self._set_message("analyzed",
                          "CLAP: 1 chunk embeddings; "
                          "MERT failed: Output channels > 65536")
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        hint = pane._chunks_hint.text()
        # the hint stays short: no per-model diagnostics concatenated here
        self.assertNotIn("MERT", hint)
        self.assertNotIn("no results", hint)
        self.assertNotIn("Output channels", hint)
        self.assertTrue(pane._chunks_hint.wordWrap())
        # the reason stays discoverable via the MERT column header tooltip
        table = pane._chunks_table
        headers = {table.horizontalHeaderItem(c).text(): c
                   for c in range(table.columnCount())}
        self.assertIn("MERT", headers)
        tip = table.horizontalHeaderItem(headers["MERT"]).toolTip()
        self.assertIn("no results", tip)
        self.assertIn("Output channels > 65536", tip)

    def test_similar_error_shown_on_similar_tab(self):
        from app.ui.detail_pane import DetailPane
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        pane.show_similar_error("No comparable embeddings for method 'mert'")
        # the label itself must be shown (its tab page is not the active tab
        # while the window is never shown, so check the widget's own flag)
        self.assertFalse(pane._similar_status.isHidden())
        self.assertIn("mert", pane._similar_status.text())

    def test_available_methods_note(self):
        from app.ui.detail_pane import DetailPane
        with self.db.transaction() as conn:
            repo.set_track_embedding(conn, self.track_id, "clap",
                                     np.ones(8, dtype=np.float32))
            repo.set_track_embedding(conn, self.track_id, "mert",
                                     np.ones(8, dtype=np.float32))
        pane = DetailPane(self.db, AppConfig())
        pane.show_track(self.track_id)
        text = pane._similar_status.text()
        self.assertIn("CLAP", text)
        self.assertIn("MERT", text)


class TestMainWindow(UiTestBase):
    def test_main_window_constructs_and_shows_track(self):
        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        window = MainWindow(config, db_path=self.db_path)
        self.assertEqual(window._tree.topLevelItemCount(), 1)
        window._tree.select_track(self.track_id)
        self.assertEqual(window._details.current_track_id(), self.track_id)
        window.close()

    def test_settings_dialog_roundtrip(self):
        from app.ui.settings_dialog import SettingsDialog
        config = AppConfig()
        config.chunk_seconds = 30.0
        dialog = SettingsDialog(config)
        dialog._chunk_seconds.setValue(45.0)
        dialog._overlap.setValue(25)
        dialog._playlist_length.setValue(20)
        dialog.apply()
        self.assertEqual(config.chunk_seconds, 45.0)
        self.assertEqual(config.overlap_percent, 25)
        self.assertEqual(config.playlist_length, 20)

    def test_scan_worker_indexes_folder(self):
        from app.ui.workers import ScanWorker
        worker = ScanWorker(self.db_path, [str(self.dir)])
        worker.run()  # run synchronously, no QThread event loop needed
        with self.db.transaction() as conn:
            tracks = repo.list_tracks(conn)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["filename"], "song.wav")
        self.assertEqual(tracks[0]["codec"], "pcm_s16le")

    def test_scan_worker_reports_progress_and_permissions(self):
        import stat as stat_mod

        from app.ui.workers import ScanWorker

        # make one subdirectory unreadable to simulate a protected folder
        secret = self.dir / "secret"
        secret.mkdir()
        (secret / "hidden.mp3").write_bytes(b"x")
        secret.chmod(0)
        events = {"started": None, "progress": [], "denied": [], "errors": []}
        worker = ScanWorker(self.db_path, [str(self.dir)])
        worker.scan_started.connect(lambda t: events.__setitem__("started", t))
        worker.progress.connect(
            lambda cur, tot, name: events["progress"].append((cur, tot, name)))
        worker.permission_required.connect(
            lambda d, m: events["denied"].append((d, m)))
        worker.file_error.connect(lambda p, m: events["errors"].append((p, m)))
        try:
            if os.access(secret, os.R_OK):
                self.skipTest("chmod-based denial not effective for this user")
            worker.run()
        finally:
            secret.chmod(stat_mod.S_IRWXU)

        self.assertEqual(events["started"], 1)          # only song.wav visible
        self.assertEqual([p[0] for p in events["progress"]], [1])
        self.assertEqual(events["progress"][0][1], 1)
        self.assertEqual(events["progress"][0][2], "song.wav")
        self.assertEqual(len(events["denied"]), 1)
        self.assertTrue(events["denied"][0][0].endswith("secret"))
        self.assertEqual(events["errors"], [])


class _PickActionMenu:
    """QMenu stand-in whose ``exec`` clicks the action named *label*."""

    def __init__(self, label: str, *args, **kwargs) -> None:
        self._label = label
        self._actions: list = []

    def addAction(self, action) -> None:
        self._actions.append(action)

    def addSeparator(self) -> None:
        self._actions.append(None)

    def exec(self, pos):
        for action in self._actions:
            if action is not None and action.text() == self._label:
                return action
        return None


class TestFolderAnalyzeMenu(UiTestBase):
    """Folder context-menu Analyze runs the subtree recursively (and
    incrementally: skip_analyzed=False)."""

    def _add_track(self, rel_path: str) -> int:
        import numpy as np
        import soundfile as sf

        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "duration_sec": 1.0})

    def test_folder_analyze_passes_skip_analyzed_false(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        import app.ui.folder_tree as folder_tree
        from PySide6.QtCore import QPoint

        from app.ui.folder_tree import FolderTree

        self._add_track("rock/a.wav")
        self._add_track("rock/deep/b.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        win = SimpleNamespace(_tree=tree,
                              analyze_track_ids=self._record)
        with patch.object(tree, "window", return_value=win), \
                patch.object(folder_tree, "QMenu",
                             lambda *a, **k: _PickActionMenu("Analyze", *a, **k)):
            tree.setCurrentItem(self._dir_item(tree, self.dir / "rock"))
            tree._show_context_menu(QPoint(4, 4))
        self.assertEqual(self._calls, {"skip_analyzed": False})
        self.assertEqual(len(self._ids), 2)   # whole subtree

    _calls: dict = {}
    _ids: list = []

    def _record(self, ids, force=False, skip_analyzed=True):
        self._ids = list(ids)
        self._calls = {"skip_analyzed": skip_analyzed}

    def _dir_item(self, tree: FolderTree, path):
        stack = [tree.topLevelItem(i)
                 for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.data(0, 0) is not None and \
                    str(item.data(0, folder_tree_dir_role())) == str(path):
                return item
            stack.extend(item.child(i) for i in range(item.childCount()))
        raise AssertionError(f"dir item {path} not found")


def folder_tree_dir_role() -> int:
    from app.ui.folder_tree import DIR_ROLE
    return DIR_ROLE


class TestExcludedFromProgress(UiTestBase):
    """Tracks not subject to analysis (WAV by default) are greyed out and
    never count toward the analysis-progress percentages."""

    def _walk(self, item):
        yield item
        for i in range(item.childCount()):
            yield from self._walk(item.child(i))

    def _track_item(self, tree, track_id: int):
        from app.ui.folder_tree import TRACK_ROLE
        for item in self._walk(tree.invisibleRootItem()):
            if item.data(0, TRACK_ROLE) == track_id:
                return item
        return None

    def _add_flac(self) -> int:
        import numpy as np
        import soundfile as sf

        path = self.dir / "song.flac"
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": "song.flac", "extension": ".flac",
                 "duration_sec": 1.0})

    @staticmethod
    def _status_texts(tree, item) -> list[str]:
        from app.ui.folder_tree import MODEL_COLUMN_OFFSET
        from app.models.registry import list_plugins
        return ([item.text(1)]
                + [item.text(MODEL_COLUMN_OFFSET + i)
                   for i in range(len(list_plugins()))])

    def test_excluded_track_greyed_and_not_counted(self) -> None:
        from PySide6.QtGui import QBrush

        from app.ui.folder_tree import EXCLUDED_ROLE, FolderTree

        flac_id = self._add_flac()
        tree = FolderTree(self.db, excluded_extensions=(".wav",))
        tree.refresh()
        wav_item = self._track_item(tree, self.track_id)
        flac_item = self._track_item(tree, flac_id)
        # the wav is flagged, greyed, dash-marked
        self.assertTrue(wav_item.data(0, EXCLUDED_ROLE))
        self.assertEqual(wav_item.foreground(0).color().name(), "#9e9e9e")
        self.assertTrue(all(brush.color().name() == "#9e9e9e"
                            for brush in (wav_item.foreground(1),)))
        self.assertTrue(all(t == "–" for t in self._status_texts(tree, wav_item)))
        self.assertIn("excluded from analysis", wav_item.toolTip(1))
        # the flac is a regular row
        self.assertFalse(flac_item.data(0, EXCLUDED_ROLE))
        self.assertNotEqual(flac_item.foreground(0).color().name(), "#9e9e9e")
        # folder percentage counts ONLY the flac
        folder = tree.topLevelItem(0)
        self.assertEqual(folder.text(1), "0%")
        self.assertEqual(folder.toolTip(1), "0 of 1 files analyzed")
        # once the flac is analyzed the folder is 100% (wav ignored)
        with self.db.transaction() as conn:
            repo.set_track_status(conn, flac_id, "analyzed", "done")
        tree.update_track_status(flac_id, "analyzed", "done")
        self.assertEqual(folder.text(1), "100%")
        self.assertEqual(folder.toolTip(1), "1 of 1 files analyzed")
        del QBrush

    def test_analyze_wav_true_counts_everything(self) -> None:
        from app.ui.folder_tree import FolderTree

        flac_id = self._add_flac()
        tree = FolderTree(self.db)   # no exclusions passed
        tree.refresh()
        folder = tree.topLevelItem(0)
        self.assertEqual(folder.toolTip(1), "0 of 2 files analyzed")

    def test_main_window_wires_wav_exclusion(self) -> None:
        from app.config import AppConfig
        from app.ui.main_window import MainWindow

        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        from app.ui.folder_tree import EXCLUDED_ROLE
        item = self._track_item(win._tree, self.track_id)
        self.assertTrue(item.data(0, EXCLUDED_ROLE))
        self.assertEqual(item.foreground(0).color().name(), "#9e9e9e")
        config2 = AppConfig()
        config2.use_ollama = False
        config2.analyze_wav = True
        win2 = MainWindow(config2, db_path=self.db_path)
        item2 = self._track_item(win2._tree, self.track_id)
        self.assertFalse(item2.data(0, EXCLUDED_ROLE))
        self.assertNotEqual(item2.foreground(0).color().name(), "#9e9e9e")


class TestWavExclusion(UiTestBase):
    """WAV files are excluded from batch analysis by default (analyze_wav)."""

    def _add_flac(self) -> int:
        path = self.dir / "song.flac"
        import soundfile as sf
        import numpy as np
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": "song.flac", "extension": ".flac",
                 "duration_sec": 1.0})

    def _window(self, config):
        from app.ui.main_window import MainWindow

        config.use_ollama = False
        return MainWindow(config, db_path=self.db_path)

    def test_wav_filtered_by_default(self) -> None:
        win = self._window(AppConfig())
        flac = self._add_flac()
        kept, wav_skipped = win._split_wav_tracks([self.track_id, flac])
        self.assertEqual(kept, [flac])
        self.assertEqual(wav_skipped, 1)

    def test_wav_kept_when_enabled(self) -> None:
        config = AppConfig()
        config.analyze_wav = True
        win = self._window(config)
        flac = self._add_flac()
        kept, wav_skipped = win._split_wav_tracks([self.track_id, flac])
        self.assertEqual(kept, [self.track_id, flac])
        self.assertEqual(wav_skipped, 0)

    def test_wav_only_batch_starts_nothing(self) -> None:
        from unittest.mock import patch

        win = self._window(AppConfig())
        with patch.object(win, "_status") as status:
            win.analyze_track_ids([self.track_id])
        self.assertIsNone(win._analysis_worker)
        joined = " ".join(str(c.args[0]) for c in status.call_args_list)
        self.assertIn("WAV", joined)


class TestAnalysisStop(UiTestBase):
    """Stop button for the analysis process: stop now, resume later."""

    def _add_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": path.stat().st_size})

    def _make_worker(self, track_ids):
        from app.ui.workers import AnalysisWorker
        config = AppConfig()
        # These tests assert strict sequential ordering and synchronous signal
        # delivery — pin the parallel degree to 1 (AnalysisWorker's sequential
        # fast path reproduces the pre-parallel semantics exactly).
        config.analysis_parallelism = 1
        return AnalysisWorker(self.db_path, config, list(track_ids))

    @staticmethod
    def _wire(worker, events):
        worker.track_started.connect(
            lambda tid, path: events["started"].append(tid))
        worker.track_progress.connect(
            lambda tid, msg: events["progress"].append((tid, msg)))
        worker.track_finished.connect(
            lambda tid, ok, msg: events["finished"].append((tid, ok, msg)))
        worker.stopped.connect(lambda count: events["stopped"].append(count))
        worker.all_finished.connect(lambda: events["all"].append(True))
        worker.failed.connect(lambda msg: events["failed"].append(msg))

    @staticmethod
    def _fake_analyze(worker, state):
        """Fake pipeline shaped like the real one: notify, maybe abort, persist.

        ``state["stop_before_step"]``  — user clicks Stop mid-track before the
        next pipeline step: the worker's own progress callback raises.
        ``state["stop_after_notify"]`` — user clicks Stop mid-track between two
        steps: the fake itself raises the workers-module stop exception.
        ``state["stop_after"]``        — user clicks Stop right after the Nth
        track finished: the stop surfaces between tracks.
        """
        from app.ui import workers as workers_mod

        def fake(db, track_id, config, progress_cb=None,
                force=False):
            notify = progress_cb or (lambda msg, cur=None, tot=None: None)
            if state.get("stop_before_step"):
                worker.request_stop()
            notify("step")            # worker's progress_cb raises when stopped
            if state.get("stop_after_notify"):
                worker.request_stop()
            if worker._stop_requested:
                raise workers_mod._AnalysisStopped(track_id)
            with db.transaction() as conn:
                repo.set_track_status(conn, track_id, "analyzed", "fake done")
            state["completed"].append(track_id)
            if state.get("stop_after") == len(state["completed"]):
                worker.request_stop()
        return fake

    @staticmethod
    def _run(worker, fake):
        from unittest import mock
        with mock.patch("app.analysis.pipeline.analyze_track", fake):
            worker.run()  # synchronous, like the ScanWorker tests above

    def _row(self, track_id):
        conn = self.db.connect()
        try:
            return repo.get_track(conn, track_id)
        finally:
            conn.close()

    def _events(self):
        return {"started": [], "progress": [], "finished": [],
                "stopped": [], "all": [], "failed": []}

    def test_stop_between_tracks_leaves_remaining_untouched(self):
        id_b = self._add_track("second/b.wav")
        worker = self._make_worker([self.track_id, id_b])
        events = self._events()
        self._wire(worker, events)
        state = {"completed": [], "stop_after": 1}

        self._run(worker, self._fake_analyze(worker, state))

        self.assertEqual(events["failed"], [])
        self.assertEqual(events["started"], [self.track_id])  # 2nd never started
        self.assertEqual([ok for _, ok, _ in events["finished"]], [True])
        self.assertEqual(events["stopped"], [1])       # 1 track done before stop
        self.assertEqual(len(events["all"]), 1)
        row_a, row_b = self._row(self.track_id), self._row(id_b)
        self.assertEqual(row_a["status"], "analyzed")
        self.assertEqual(row_b["status"], "new")       # untouched, resume later
        self.assertIsNone(row_b["status_message"])

    def test_stop_mid_track_via_progress_callback(self):
        id_b = self._add_track("second/b.wav")
        worker = self._make_worker([self.track_id, id_b])
        events = self._events()
        self._wire(worker, events)
        state = {"completed": [], "stop_before_step": True}

        self._run(worker, self._fake_analyze(worker, state))

        self.assertEqual(events["failed"], [])
        self.assertEqual(events["progress"], [])       # abort surfaced in the cb
        self.assertEqual(events["started"], [self.track_id])
        self.assertEqual(len(events["finished"]), 1)
        tid, ok, msg = events["finished"][0]
        self.assertEqual(tid, self.track_id)
        self.assertFalse(ok)
        self.assertIn("Stopped by user", msg)
        self.assertEqual(events["stopped"], [0])
        self.assertEqual(len(events["all"]), 1)
        row_a, row_b = self._row(self.track_id), self._row(id_b)
        self.assertEqual(row_a["status"], "new")       # retryable again
        self.assertIn("Stopped by user", row_a["status_message"])
        self.assertEqual(row_b["status"], "new")       # untouched
        self.assertIsNone(row_b["status_message"])

    def test_stop_mid_track_via_fake_raise(self):
        id_b = self._add_track("second/b.wav")
        worker = self._make_worker([self.track_id, id_b])
        events = self._events()
        self._wire(worker, events)
        state = {"completed": [], "stop_after_notify": True}

        self._run(worker, self._fake_analyze(worker, state))

        self.assertEqual(events["failed"], [])
        self.assertEqual(events["progress"], [(self.track_id, "step")])
        self.assertEqual(len(events["finished"]), 1)
        _, ok, msg = events["finished"][0]
        self.assertFalse(ok)
        self.assertIn("Stopped by user", msg)
        self.assertEqual(events["stopped"], [0])
        row_a = self._row(self.track_id)
        self.assertEqual(row_a["status"], "new")
        self.assertIn("Stopped by user", row_a["status_message"])

    def test_no_stop_completes_normally_without_stopped_signal(self):
        id_b = self._add_track("second/b.wav")
        worker = self._make_worker([self.track_id, id_b])
        events = self._events()
        self._wire(worker, events)
        state = {"completed": []}

        self._run(worker, self._fake_analyze(worker, state))

        self.assertEqual(events["failed"], [])
        self.assertEqual(events["started"], [self.track_id, id_b])
        self.assertEqual([ok for _, ok, _ in events["finished"]], [True, True])
        self.assertEqual(events["stopped"], [])        # NOT emitted
        self.assertEqual(len(events["all"]), 1)
        self.assertEqual(self._row(self.track_id)["status"], "analyzed")
        self.assertEqual(self._row(id_b)["status"], "analyzed")

    def test_set_busy_enables_stop_only_while_analysis_running(self):
        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            class _StubWorker:
                def __init__(self, running):
                    self._running = running

                def isRunning(self):
                    return self._running

                def wait(self, msecs=0):   # closeEvent() may call this
                    return True

            others = (win._act_add, win._act_remove, win._act_rescan,
                      win._act_analyze, win._act_analyze_all)
            self.assertFalse(win._act_stop.isEnabled())    # disabled at rest
            win._analysis_worker = _StubWorker(running=False)
            win._set_busy(True)          # scan-style busy: no analysis worker
            for action in others:
                self.assertFalse(action.isEnabled())
            self.assertFalse(win._act_stop.isEnabled())    # Stop stays off
            win._analysis_worker = _StubWorker(running=True)
            win._set_busy(True)          # analysis busy: Stop available
            self.assertTrue(win._act_stop.isEnabled())
            win._set_busy(False)         # idle again
            for action in others:
                self.assertTrue(action.isEnabled())
            self.assertFalse(win._act_stop.isEnabled())
        finally:
            win.close()

    def test_stop_analysis_noops_when_no_analysis_running(self):
        from app.ui.main_window import MainWindow
        from app.ui.workers import AnalysisWorker
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            self.assertFalse(win._act_stop.isEnabled())    # disabled at rest
            win.stop_analysis()                            # no worker at all
            self.assertIn("no analysis",
                          win.statusBar().currentMessage().lower())
            # a created-but-not-running worker also no-ops
            win._analysis_worker = AnalysisWorker(self.db_path, config, [])
            win.stop_analysis()
            self.assertIn("no analysis",
                          win.statusBar().currentMessage().lower())
        finally:
            win.close()

    def test_main_window_stop_analysis_flow(self):
        import threading
        import time
        from unittest import mock

        from PySide6.QtTest import QTest

        from app.ui import workers as workers_mod
        from app.ui.main_window import MainWindow
        id_b = self._add_track("second/b.wav")
        config = AppConfig()
        config.use_ollama = False
        config.analyze_wav = True   # this fixture deliberately uses .wav
        config.analysis_parallelism = 1   # strict ordering asserted below
        win = MainWindow(config, db_path=self.db_path)
        try:
            state = {"completed": []}
            in_track = threading.Event()

            def fake_analyze(db, track_id, cfg, progress_cb=None,
                             force=False):
                notify = progress_cb or (lambda msg, cur=None, tot=None: None)
                in_track.set()       # worker is now mid-track on track 1
                time.sleep(0.05)     # keep the run alive for the stop click
                notify("step")       # worker's progress_cb raises once stopped
                if win._analysis_worker._stop_requested:
                    raise workers_mod._AnalysisStopped(track_id)
                with db.transaction() as conn:
                    repo.set_track_status(conn, track_id, "analyzed",
                                          "fake done")
                state["completed"].append(track_id)

            with mock.patch("app.analysis.pipeline.analyze_track", fake_analyze):
                win.analyze_track_ids([self.track_id, id_b])
                worker = win._analysis_worker
                self.assertIsNotNone(worker)
                self.assertTrue(win._act_stop.isEnabled())  # Stop available
                seen = {"started": [], "finished": [], "stopped": [],
                        "all": []}
                worker.track_started.connect(
                    lambda tid, path: seen["started"].append(tid))
                worker.track_finished.connect(
                    lambda tid, ok, msg: seen["finished"].append((tid, ok, msg)))
                worker.stopped.connect(
                    lambda count: seen["stopped"].append(count))
                worker.all_finished.connect(lambda: seen["all"].append(True))
                # wait until the worker is genuinely mid-track, then stop —
                # stopping earlier would legitimately abort between tracks
                self.assertTrue(in_track.wait(5.0))
                win.stop_analysis()     # the Stop Analysis toolbar action
                self.assertIn("Stopping analysis",
                              win.statusBar().currentMessage())
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    QTest.qWait(20)
                    # all_finished may lag worker shutdown slightly (the
                    # run's post-pass runs first) — wait for the signal.
                    if seen["all"] and not worker.isRunning():
                        break

            self.assertFalse(worker.isRunning())
            self.assertFalse(win._act_stop.isEnabled())     # disabled again
            self.assertEqual(seen["started"], [self.track_id])
            self.assertEqual(seen["stopped"], [0])
            self.assertEqual(len(seen["all"]), 1)
            self.assertEqual(len(seen["finished"]), 1)
            _, ok, msg = seen["finished"][0]
            self.assertFalse(ok)
            self.assertIn("Stopped by user", msg)
            self.assertEqual(state["completed"], [])        # nothing completed
            message = win.statusBar().currentMessage()
            self.assertIn("stopped", message.lower())
            self.assertIn("0 track(s) done", message)
            # first track retryable, second untouched — resume later
            row_a, row_b = self._row(self.track_id), self._row(id_b)
            self.assertEqual(row_a["status"], "new")
            self.assertIn("Stopped by user", row_a["status_message"])
            self.assertEqual(row_b["status"], "new")
        finally:
            win.close()


class TestFoldButtons(UiTestBase):
    """Fold all / Unfold all buttons on the LibraryPane filter row."""

    def _add_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": path.stat().st_size})

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _dir_item(self, tree, full_path: str):
        from app.ui.folder_tree import DIR_ROLE
        for item in self._walk(tree):
            if item.data(0, DIR_ROLE) == str(full_path):
                return item
        return None

    def _dir_items(self, tree):
        """Every subfolder node (DIR_ROLE), e.g. rock, rock/deep, jazz."""
        from app.ui.folder_tree import DIR_ROLE
        return [item for item in self._walk(tree)
                if item.data(0, DIR_ROLE) is not None]

    def _pane(self):
        """Pane with rock/, rock/deep/, jazz/ and a root-level song.wav."""
        from app.ui.folder_tree import LibraryPane
        self._add_track("rock/a.wav")
        self._add_track("rock/deep/b.wav")
        self._add_track("jazz/c.wav")
        pane = LibraryPane(self.db)
        pane.tree.refresh()
        return pane

    def test_initial_state_root_expanded_subfolders_collapsed(self):
        pane = self._pane()
        self.assertTrue(pane.tree.topLevelItem(0).isExpanded())
        for item in self._dir_items(pane.tree):
            self.assertFalse(item.isExpanded())

    def test_unfold_all_expands_every_folder(self):
        pane = self._pane()
        pane.unfold_all()
        self.assertTrue(pane.tree.topLevelItem(0).isExpanded())
        for item in self._dir_items(pane.tree):
            self.assertTrue(item.isExpanded())

    def test_fold_all_collapses_every_folder(self):
        pane = self._pane()
        pane.unfold_all()
        pane.fold_all()
        self.assertFalse(pane.tree.topLevelItem(0).isExpanded())
        for item in self._dir_items(pane.tree):
            self.assertFalse(item.isExpanded())

    def test_buttons_click_toggle_expansion(self):
        from PySide6.QtCore import Qt
        pane = self._pane()
        # compact flat buttons that never steal the tree's keyboard focus
        for button, text, tooltip in (
                (pane._fold_button, "Fold all", "Collapse all folders"),
                (pane._unfold_button, "Unfold all", "Expand all folders")):
            self.assertEqual(button.text(), text)
            self.assertEqual(button.toolTip(), tooltip)
            self.assertTrue(button.autoRaise())
            self.assertEqual(button.toolButtonStyle(),
                             Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus)
        pane._unfold_button.click()
        self.assertTrue(pane.tree.topLevelItem(0).isExpanded())
        for item in self._dir_items(pane.tree):
            self.assertTrue(item.isExpanded())
        pane._fold_button.click()
        self.assertFalse(pane.tree.topLevelItem(0).isExpanded())
        for item in self._dir_items(pane.tree):
            self.assertFalse(item.isExpanded())

    def test_refresh_after_fold_resets_to_root_expanded(self):
        pane = self._pane()
        pane.fold_all()
        self.assertFalse(pane.tree.topLevelItem(0).isExpanded())
        pane.tree.refresh()          # documented reset: root expanded only
        self.assertTrue(pane.tree.topLevelItem(0).isExpanded())
        for item in self._dir_items(pane.tree):
            self.assertFalse(item.isExpanded())

    def test_filter_after_fold_expands_matching_branch(self):
        pane = self._pane()
        pane.fold_all()
        pane.apply_filter("deep")    # matching branch becomes visible again
        deep = self._dir_item(pane.tree, self.dir / "rock" / "deep")
        rock = self._dir_item(pane.tree, self.dir / "rock")
        jazz = self._dir_item(pane.tree, self.dir / "jazz")
        self.assertFalse(deep.isHidden())
        self.assertTrue(deep.isExpanded())
        self.assertFalse(rock.isHidden())
        self.assertTrue(rock.isExpanded())   # ancestor of the match
        self.assertTrue(jazz.isHidden())


class TestTreeStatusUpdates(UiTestBase):
    """In-place track status updates and per-folder ready percentages."""

    def _add_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": path.stat().st_size})

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _dir_item(self, tree, full_path: str):
        from app.ui.folder_tree import DIR_ROLE
        for item in self._walk(tree):
            if item.data(0, DIR_ROLE) == str(full_path):
                return item
        return None

    def _track_item(self, tree, track_id: int):
        from app.ui.folder_tree import TRACK_ROLE
        for item in self._walk(tree):
            if item.data(0, TRACK_ROLE) == track_id:
                return item
        return None

    def _tree(self):
        from app.ui.folder_tree import FolderTree
        tree = FolderTree(self.db)
        tree.refresh()
        return tree

    def test_update_track_status_preserves_selection_and_expansion(self):
        deep_id = self._add_track("artists/Alpha/2024/live_set.wav")
        self._add_track("jazz/c.wav")
        tree = self._tree()
        tree.select_track(deep_id)
        current = tree.currentItem()
        artists = self._dir_item(tree, self.dir / "artists")
        year = self._dir_item(tree, self.dir / "artists" / "Alpha" / "2024")
        jazz = self._dir_item(tree, self.dir / "jazz")
        self.assertTrue(artists.isExpanded())   # select_track expanded them
        self.assertTrue(year.isExpanded())
        self.assertFalse(jazz.isExpanded())     # unrelated branch collapsed
        item = self._track_item(tree, deep_id)
        self.assertEqual(item.text(1), "•")

        tree.update_track_status(deep_id, "analyzed", "done")

        self.assertEqual(item.text(1), "✓")
        self.assertEqual(item.toolTip(1), "analyzed — done")
        self.assertIs(tree.currentItem(), current)   # selection untouched
        self.assertEqual(tree.selected_track_id(), deep_id)
        self.assertTrue(artists.isExpanded())        # expansion untouched
        self.assertTrue(year.isExpanded())
        self.assertFalse(jazz.isExpanded())

    def test_update_track_status_glyph_and_tooltip_formats(self):
        tree = self._tree()
        item = self._track_item(tree, self.track_id)
        tree.update_track_status(self.track_id, "analyzing", None)
        self.assertEqual(item.text(1), "…")
        self.assertEqual(item.toolTip(1), "analyzing")
        self.assertEqual(item.toolTip(0), str(self.wav))   # path tooltip kept
        tree.update_track_status(self.track_id, "error",
                                 "Decode failed: broken file")
        self.assertEqual(item.text(1), "✗")
        self.assertEqual(item.toolTip(1), "error — Decode failed: broken file")

    def test_update_track_status_unknown_track_is_noop(self):
        tree = self._tree()
        root = tree.topLevelItem(0)
        before = (root.text(1), root.toolTip(1))
        tree.update_track_status(999999, "analyzed", "done")
        self.assertEqual((root.text(1), root.toolTip(1)), before)

    def test_folder_percentage_counts_analyzed_files(self):
        a_id = self._add_track("rock/a.wav")
        self._add_track("rock/b.wav")
        c_id = self._add_track("rock/deep/c.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
            repo.set_track_status(conn, c_id, "analyzed", "done")
        tree = self._tree()
        deep = self._dir_item(tree, self.dir / "rock" / "deep")
        rock = self._dir_item(tree, self.dir / "rock")
        root = tree.topLevelItem(0)
        self.assertEqual(deep.text(1), "100%")
        self.assertEqual(rock.text(1), "67%")   # 66.67% rounds to 67
        self.assertEqual(rock.toolTip(1), "2 of 3 files analyzed")
        # root aggregate includes the root-level song.wav (still "new")
        self.assertEqual(root.text(1), "50%")

    def test_update_track_status_recomputes_ancestor_percentages_only(self):
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/deep/b.wav")
        self._add_track("jazz/c.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, b_id, "analyzed", "done")
        tree = self._tree()
        rock = self._dir_item(tree, self.dir / "rock")
        deep = self._dir_item(tree, self.dir / "rock" / "deep")
        jazz = self._dir_item(tree, self.dir / "jazz")
        root = tree.topLevelItem(0)
        self.assertEqual(rock.text(1), "50%")
        self.assertEqual(deep.text(1), "100%")
        self.assertEqual(jazz.text(1), "0%")
        self.assertEqual(root.text(1), "25%")

        tree.update_track_status(a_id, "analyzed", "done")

        self.assertEqual(rock.text(1), "100%")   # ancestor updated
        self.assertEqual(root.text(1), "50%")    # ancestor updated
        self.assertEqual(deep.text(1), "100%")   # unchanged subtree
        self.assertEqual(jazz.text(1), "0%")     # untouched branch

    def test_folder_percentages_survive_full_refresh(self):
        a_id = self._add_track("rock/a.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
        self._add_track("rock/deep/b.wav")
        tree = self._tree()
        tree.refresh()          # full rebuild — percentages recomputed
        rock = self._dir_item(tree, self.dir / "rock")
        deep = self._dir_item(tree, self.dir / "rock" / "deep")
        root = tree.topLevelItem(0)
        self.assertEqual(rock.text(1), "50%")
        self.assertEqual(deep.text(1), "0%")
        self.assertEqual(root.text(1), "33%")
        self.assertEqual(rock.toolTip(1), "1 of 2 files analyzed")

    def test_complete_folder_status_turns_green(self):
        from PySide6.QtGui import QColor

        from app.ui.folder_tree import _COMPLETE_GREEN
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/b.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
            repo.set_track_status(conn, b_id, "analyzed", "done")
        tree = self._tree()
        rock = self._dir_item(tree, self.dir / "rock")
        self.assertEqual(rock.text(1), "100%")   # text format unchanged
        self.assertEqual(QColor(rock.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())

    def test_incomplete_folder_status_is_not_green(self):
        from PySide6.QtGui import QBrush, QColor

        from app.ui.folder_tree import _COMPLETE_GREEN
        a_id = self._add_track("rock/a.wav")
        self._add_track("rock/b.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
        tree = self._tree()
        rock = self._dir_item(tree, self.dir / "rock")
        self.assertEqual(rock.text(1), "50%")
        self.assertNotEqual(QColor(rock.foreground(1).color()).name(),
                            QColor(_COMPLETE_GREEN).name())
        self.assertEqual(rock.foreground(1), QBrush())   # default foreground

    def test_update_track_status_turns_complete_folder_green_in_place(self):
        from PySide6.QtGui import QColor

        from app.ui.folder_tree import _COMPLETE_GREEN
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/deep/b.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
        tree = self._tree()
        tree.select_track(a_id)
        current = tree.currentItem()
        rock = self._dir_item(tree, self.dir / "rock")
        deep = self._dir_item(tree, self.dir / "rock" / "deep")
        self.assertEqual(deep.text(1), "0%")
        self.assertNotEqual(QColor(deep.foreground(1).color()).name(),
                            QColor(_COMPLETE_GREEN).name())

        tree.update_track_status(b_id, "analyzed", None)

        # deep folder and its ancestor chain flip green immediately
        self.assertEqual(deep.text(1), "100%")
        self.assertEqual(QColor(deep.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())
        self.assertEqual(rock.text(1), "100%")
        self.assertEqual(QColor(rock.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())
        self.assertIs(tree.currentItem(), current)   # selection untouched
        self.assertEqual(tree.selected_track_id(), a_id)

    def test_root_turns_green_only_when_whole_subtree_analyzed(self):
        from PySide6.QtGui import QColor

        from app.ui.folder_tree import _COMPLETE_GREEN
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/deep/b.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
            repo.set_track_status(conn, b_id, "analyzed", "done")
            second_folder_id = repo.add_folder(conn,
                                               str(self.dir / "elsewhere"))
        path = self.dir / "elsewhere" / "x.wav"
        path.parent.mkdir()
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            repo.upsert_track(
                conn, second_folder_id, str(path),
                {"filename": "x.wav", "extension": ".wav", "codec": "pcm_s16le",
                 "sample_rate": sr, "channels": 1, "duration_sec": 1.0,
                 "size_bytes": path.stat().st_size})
        tree = self._tree()
        roots = {tree.topLevelItem(i).text(0): tree.topLevelItem(i)
                 for i in range(tree.topLevelItemCount())}
        first = roots[str(self.dir)]
        second = roots[str(self.dir / "elsewhere")]
        rock = self._dir_item(tree, self.dir / "rock")
        # complete subfolder is green, but the root is not: song.wav is "new"
        self.assertEqual(QColor(rock.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())
        self.assertEqual(first.text(1), "67%")
        self.assertNotEqual(QColor(first.foreground(1).color()).name(),
                            QColor(_COMPLETE_GREEN).name())
        self.assertEqual(second.text(1), "0%")
        self.assertNotEqual(QColor(second.foreground(1).color()).name(),
                            QColor(_COMPLETE_GREEN).name())

        with self.db.transaction() as conn:
            repo.set_track_status(conn, self.track_id, "analyzed", "done")
        tree.update_track_status(self.track_id, "analyzed", None)

        # last file analysed: the root (whole subtree incl. root-level files
        # and every subfolder) turns green; the other root stays non-green
        self.assertEqual(first.text(1), "100%")
        self.assertEqual(QColor(first.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())
        self.assertEqual(second.text(1), "0%")
        self.assertNotEqual(QColor(second.foreground(1).color()).name(),
                            QColor(_COMPLETE_GREEN).name())

    def test_complete_folder_green_survives_refresh(self):
        from PySide6.QtGui import QColor

        from app.ui.folder_tree import _COMPLETE_GREEN
        a_id = self._add_track("rock/a.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
            repo.set_track_status(conn, self.track_id, "analyzed", "done")
        tree = self._tree()
        self.assertEqual(QColor(tree.topLevelItem(0).foreground(1).color())
                         .name(), QColor(_COMPLETE_GREEN).name())
        tree.refresh()          # full rebuild — green recomputed, not kept
        rock = self._dir_item(tree, self.dir / "rock")
        root = tree.topLevelItem(0)
        self.assertEqual(rock.text(1), "100%")
        self.assertEqual(QColor(rock.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())
        self.assertEqual(root.text(1), "100%")
        self.assertEqual(QColor(root.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())

    def test_reverting_track_status_clears_folder_green(self):
        from PySide6.QtGui import QBrush, QColor

        from app.ui.folder_tree import _COMPLETE_GREEN
        a_id = self._add_track("rock/a.wav")
        b_id = self._add_track("rock/b.wav")
        with self.db.transaction() as conn:
            repo.set_track_status(conn, a_id, "analyzed", "done")
            repo.set_track_status(conn, b_id, "analyzed", "done")
        tree = self._tree()
        rock = self._dir_item(tree, self.dir / "rock")
        self.assertEqual(QColor(rock.foreground(1).color()).name(),
                         QColor(_COMPLETE_GREEN).name())

        with self.db.transaction() as conn:
            repo.set_track_status(conn, b_id, "new")
        tree.update_track_status(b_id, "new", None)

        self.assertEqual(rock.text(1), "50%")
        self.assertNotEqual(QColor(rock.foreground(1).color()).name(),
                            QColor(_COMPLETE_GREEN).name())
        self.assertEqual(rock.foreground(1), QBrush())   # green cleared


class TestScanHighlight(UiTestBase):
    """Yellow highlight of folders being scanned + speed status label."""

    def _add_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": path.stat().st_size})

    def _second_root(self) -> str:
        with self.db.transaction() as conn:
            repo.add_folder(conn, str(self.dir / "elsewhere"))
        return str(self.dir / "elsewhere")

    def test_set_scanning_paths_marks_only_matching_roots(self):
        from PySide6.QtGui import QBrush, QColor

        from app.ui.folder_tree import _ACTIVITY_BRUSH, FolderTree
        self._second_root()
        tree = FolderTree(self.db)
        tree.refresh()
        items = {tree.topLevelItem(i).text(0): tree.topLevelItem(i)
                 for i in range(tree.topLevelItemCount())}
        self.assertEqual(len(items), 2)
        main_item = items[str(self.dir)]
        other_item = items[str(self.dir / "elsewhere")]
        # clean start: no highlight anywhere
        self.assertEqual(main_item.background(0), QBrush())
        self.assertEqual(other_item.background(0), QBrush())

        tree.set_scanning_paths([str(self.dir)])

        self.assertEqual(main_item.background(0).color(),
                         _ACTIVITY_BRUSH.color())
        self.assertEqual(other_item.background(0), QBrush())

        tree.set_scanning_paths([])

        self.assertEqual(main_item.background(0), QBrush())
        self.assertEqual(other_item.background(0), QBrush())

    def test_scan_highlight_survives_refresh(self):
        from PySide6.QtGui import QBrush, QColor

        from app.ui.folder_tree import _ACTIVITY_BRUSH, FolderTree
        tree = FolderTree(self.db)
        tree.refresh()
        tree.set_scanning_paths([str(self.dir)])
        tree.refresh()          # rebuild — highlight re-applied from state
        root = tree.topLevelItem(0)
        self.assertEqual(root.background(0).color(),
                         _ACTIVITY_BRUSH.color())
        tree.set_scanning_paths([])
        tree.refresh()
        self.assertEqual(tree.topLevelItem(0).background(0), QBrush())

    def test_on_folder_scan_started_accumulates_and_clears(self):
        from PySide6.QtGui import QBrush, QColor

        from app.ui.folder_tree import _ACTIVITY_BRUSH
        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            win._on_folder_scan_started(str(self.dir))
            win._on_folder_scan_started(str(self.dir / "second"))
            self.assertEqual(win._scanning_paths,
                             [str(self.dir), str(self.dir / "second")])
            root = win._tree.topLevelItem(0)
            self.assertEqual(root.background(0).color(),
                             _ACTIVITY_BRUSH.color())

            win._on_scan_finished(0, 0, 0)

            self.assertEqual(win._scanning_paths, [])
            # refresh() rebuilt the tree: re-query the new root item
            self.assertEqual(win._tree.topLevelItem(0).background(0), QBrush())
        finally:
            win.close()

    def test_speed_label_updates_over_analysis_run(self):
        import time as time_mod

        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        with self.db.transaction() as conn:
            repo.upsert_track(conn, self.folder_id, str(self.wav),
                              {"duration_sec": 600.0})   # 10 audio minutes
        win = MainWindow(config, db_path=self.db_path)
        try:
            self.assertEqual(win._speed_label.text(), "")
            self.assertIsNone(win._speed_started_at)

            win._on_track_analysis_started(self.track_id, str(self.wav))
            self.assertIsNotNone(win._speed_started_at)   # lazy clock init
            self.assertEqual(win._speed_label.text(), "")
            # pretend exactly 10 s of wall time have passed since the run
            # began (patching the clock keeps the rate assertion deterministic
            # even when the real call takes a few extra milliseconds)
            import app.ui.main_window as main_window_module
            from unittest import mock

            with mock.patch.object(
                    main_window_module.time, "monotonic",
                    return_value=win._speed_started_at + 10.0):
                win._on_track_analysis_finished(self.track_id, True, "Analyzed")

            self.assertRegex(win._speed_label.text(), r"Speed: .*min/h")
            # 10 audio minutes / (10 s wall) = 3600 min/h
            self.assertIn("3600", win._speed_label.text())

            win._on_all_analyzed()

            self.assertEqual(win._speed_label.text(), "")
            self.assertIsNone(win._speed_started_at)
        finally:
            win.close()

    def test_scan_failed_resets_speed_only_for_analysis_failures(self):
        import time as time_mod
        from unittest import mock

        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            win._speed_started_at = time_mod.monotonic()
            win._speed_label.setText("Speed: 42 min/h")

            class _StubWorker:
                def isRunning(self):
                    return False

                def wait(self, msecs=0):   # closeEvent() may call this
                    return True

            # scan failure (no analysis worker): speed readout untouched
            win._analysis_worker = None
            with mock.patch("app.ui.main_window.QMessageBox.warning"):
                win._on_scan_failed("Scan failed: boom")
            self.assertEqual(win._speed_label.text(), "Speed: 42 min/h")

            # analysis failure (worker exists, no longer running): reset
            win._analysis_worker = _StubWorker()
            with mock.patch("app.ui.main_window.QMessageBox.warning"):
                win._on_scan_failed("Analysis worker failed: boom")
            self.assertEqual(win._speed_label.text(), "")
            self.assertIsNone(win._speed_started_at)
        finally:
            win.close()


class ModelStatusColumnTests(UiTestBase):
    """Per-model status columns in the file tree (CLAP/MERT/OpenL3/FFT)."""

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _give_embeddings(self, track_id: int, models: tuple[str, ...]) -> None:
        vec = np.ones(8, dtype=np.float32) / 2.8
        with self.db.transaction() as conn:
            ids = repo.replace_chunks(conn, track_id, [(0, 0.0, 1.0)])
            for model in models:
                repo.add_chunk_embedding(conn, ids[0], model, vec)

    def _track_item(self, tree, track_id: int):
        from app.ui.folder_tree import TRACK_ROLE
        for item in self._walk(tree):
            if item.data(0, TRACK_ROLE) == track_id:
                return item
        return None

    def test_header_has_one_column_per_registered_plugin(self):
        from app.models.registry import list_plugins
        from app.ui.folder_tree import FolderTree
        tree = FolderTree(self.db)
        self.assertEqual(tree.columnCount(), 2 + len(list_plugins()))
        labels = [tree.headerItem().text(c) for c in range(tree.columnCount())]
        self.assertEqual(labels[:2], ["Library", "Status"])
        self.assertEqual(labels[2:],
                         [p.display_name for p in list_plugins()])

    @staticmethod
    def _model_columns(tree) -> dict[str, int]:
        """Plugin name -> tree column, derived from the registry."""
        from app.models.registry import list_plugins
        from app.ui.folder_tree import MODEL_COLUMN_OFFSET
        return {p.name: MODEL_COLUMN_OFFSET + i
                for i, p in enumerate(list_plugins())}

    def test_track_rows_show_per_model_glyphs(self):
        from app.ui.folder_tree import FolderTree
        self._give_embeddings(self.track_id, ("clap", "mert"))
        tree = FolderTree(self.db)
        tree.refresh()
        item = self._track_item(tree, self.track_id)
        col = self._model_columns(tree)
        # unanalyzed track: only models with stored results get a check mark
        self.assertEqual(item.text(1), "•")
        self.assertEqual([item.text(col[m]) for m in ("clap", "mert")],
                         ["✓", "✓"])
        self.assertEqual(
            [item.text(col[m]) for m in ("mert330", "openl3", "fft")],
            ["", "", ""])
        self.assertIn("chunk embeddings", item.toolTip(col["clap"]))
        # once the track is analyzed, missing models show as failed
        with self.db.transaction() as conn:
            repo.set_track_status(conn, self.track_id, "analyzed", None)
        tree.update_track_status(self.track_id, "analyzed", None)
        self.assertEqual(item.text(1), "✓")
        self.assertEqual([item.text(col[m]) for m in ("clap", "mert")],
                         ["✓", "✓"])
        self.assertEqual(
            [item.text(col[m]) for m in ("mert330", "openl3", "fft")],
            ["✗", "✗", "✗"])
        self.assertIn("no results", item.toolTip(col["mert330"]))

    def test_folder_rows_show_per_model_ready_counts(self):
        from app.ui.folder_tree import FolderTree, MODEL_COLUMN_OFFSET
        self._give_embeddings(self.track_id, ("clap",))
        with self.db.transaction() as conn:
            repo.set_track_status(conn, self.track_id, "analyzed", None)
        tree = FolderTree(self.db)
        tree.refresh()
        root = tree.topLevelItem(0)
        self.assertEqual(root.text(1), "100%")
        # CLAP complete; the other models have no results on this track
        col = self._model_columns(tree)
        self.assertEqual(root.text(col["clap"]), "100%")
        for name in ("mert", "mert330", "openl3", "fft"):
            self.assertEqual(root.text(col[name]), "0%", name)
        self.assertIn("CLAP", root.toolTip(col["clap"]))

    def test_update_track_status_refreshes_model_columns_in_place(self):
        from app.ui.folder_tree import FolderTree, MODEL_COLUMN_OFFSET
        self._give_embeddings(self.track_id, ("fft",))
        tree = FolderTree(self.db)
        tree.refresh()
        item = self._track_item(tree, self.track_id)
        fft_col = self._model_columns(tree)["fft"]
        self.assertEqual(item.text(fft_col), "✓")
        # clear the analysis: the check marks must disappear without a rebuild
        with self.db.transaction() as conn:
            repo.clear_track_analysis(conn, self.track_id)
        tree.update_track_status(self.track_id, "new", None)
        self.assertEqual(item.text(1), "•")
        for column in range(MODEL_COLUMN_OFFSET, tree.columnCount()):
            self.assertEqual(item.text(column), "")


class FilenameColumnWidthTests(UiTestBase):
    """Column 0 is auto-sized to fit filenames plus their path nesting."""

    def _make_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * t), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": path.suffix,
                 "codec": "pcm_s16le", "sample_rate": 8000, "channels": 1,
                 "duration_sec": 1.0, "size_bytes": path.stat().st_size})

    def test_column_fits_longest_name_and_nesting(self):
        from PySide6.QtGui import QFontMetrics

        from app.ui.folder_tree import FolderTree
        self._make_track(
            "artists/Alpha/2024/live_set_with_a_rather_long_name.wav")
        tree = FolderTree(self.db)
        tree.resize(1200, 500)     # a wide pane: the cap must not bite
        tree.refresh()
        # width >= indentation of the deepest level + its text width
        metrics = QFontMetrics(tree.font())
        expected = (metrics.horizontalAdvance(
            "live_set_with_a_rather_long_name.wav")
            + 3 * tree.indentation())
        width = tree.header().sectionSize(0)
        self.assertGreaterEqual(width, expected)
        # deeper nesting than the widest text also widens the column
        self._make_track("a/b/c/d/e/f/track.wav")
        tree.refresh()
        self.assertGreater(tree.header().sectionSize(0), width - 1)

    def test_name_column_never_pushes_status_columns_out_of_view(self):
        from app.ui.folder_tree import FolderTree
        self._make_track(
            "artists/Alpha/2024/live_set_with_a_rather_long_name.wav")
        tree = FolderTree(self.db)
        tree.resize(620, 400)      # the default-ish pane width
        tree.refresh()
        header = tree.header()
        total = sum(header.sectionSize(c) for c in range(tree.columnCount()))
        # The whole status area (Status + one column per model) stays inside
        # the visible pane instead of starting behind an 800px name column.
        self.assertLessEqual(total, tree.width() + 4)
        # ... and every status column stays at least wide enough for its
        # own header label (no truncated headers).
        from PySide6.QtGui import QFontMetrics
        metrics = QFontMetrics(tree.font())
        for column in range(1, tree.columnCount()):
            label = tree.headerItem().text(column)
            self.assertGreaterEqual(
                header.sectionSize(column),
                metrics.horizontalAdvance(label), label)
        # the name column gave up its excess but stayed readable
        self.assertGreaterEqual(header.sectionSize(0), 120)

    def test_narrow_pane_rebalances_on_resize(self):
        from app.ui.folder_tree import FolderTree
        tree = FolderTree(self.db)
        tree.resize(1200, 400)
        tree.show()     # resize events are only delivered to shown widgets
        try:
            tree.refresh()
            wide = tree.header().sectionSize(0)
            tree.resize(560, 400)      # user drags the splitter narrower
            self.assertLess(tree.header().sectionSize(0), wide)
            total = sum(tree.header().sectionSize(c)
                        for c in range(tree.columnCount()))
            self.assertLessEqual(total, 560 + 4)
        finally:
            tree.close()

    def test_column_width_survives_status_updates(self):
        from app.ui.folder_tree import FolderTree
        tree = FolderTree(self.db)
        tree.refresh()
        before = tree.header().sectionSize(0)
        self.assertGreater(before, 0)
        tree.update_track_status(self.track_id, "analyzed", None)
        self.assertEqual(tree.header().sectionSize(0), before)


class PlaybackShortcutTests(UiTestBase):
    """Double-click playback + one-click .m3u create-and-play."""

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def test_tree_double_click_emits_play_request(self):
        from app.ui.folder_tree import TRACK_ROLE, FolderTree
        tree = FolderTree(self.db)
        tree.refresh()
        item = None
        for it in self._walk(tree):
            if it.data(0, TRACK_ROLE) == self.track_id:
                item = it
                break
        self.assertIsNotNone(item)
        seen = []
        tree.track_play_requested.connect(seen.append)
        tree.itemDoubleClicked.emit(item, 0)
        self.assertEqual(seen, [self.track_id])

    def test_tree_double_click_on_folder_does_not_play(self):
        from PySide6.QtCore import Qt

        from app.ui.folder_tree import FolderTree
        tree = FolderTree(self.db)
        tree.refresh()
        seen = []
        tree.track_play_requested.connect(seen.append)
        folder_item = tree.topLevelItem(0)
        # A folder node carries the library root path, not TRACK_ROLE
        self.assertIsNone(folder_item.data(0, Qt.ItemDataRole.UserRole + 2))
        tree.itemDoubleClicked.emit(folder_item, 0)
        self.assertEqual(seen, [])

    def test_similar_table_double_click_emits_play_request(self):
        from app.similarity.search import SimilarResult
        from app.ui.detail_pane import DetailPane
        pane = DetailPane(self.db, AppConfig())
        results = [SimilarResult(
            track_id=self.track_id + offset, path=f"/m/{offset}.wav",
            filename=f"{offset}.wav", artist=None, title=None,
            score=0.9 - offset, method="clap") for offset in range(3)]
        pane.show_similar_results(results)
        seen = []
        pane.play_track_requested.connect(seen.append)
        pane._similar_table.cellDoubleClicked.emit(1, 0)
        self.assertEqual(seen, [self.track_id + 1])

    def test_m3u_play_button_wiring(self):
        from app.ui.detail_pane import DetailPane
        pane = DetailPane(self.db, AppConfig())
        button = pane._playlist_play_button
        self.assertEqual(button.text(), "Create .m3u & play")
        self.assertFalse(button.isEnabled())      # nothing searched yet
        pane.show_similar_results([])              # still disabled when empty
        self.assertFalse(button.isEnabled())
        pairs = [(self.track_id, 0.9)]
        pane._similar_cache = list(pairs)
        button.setEnabled(True)
        seen = []
        pane.playlist_play_requested.connect(
            lambda got_pairs, method: seen.append((got_pairs, method)))
        button.click()
        # Default picker state (dataset=clap, algorithm=centroid) labels
        # the playlist metadata "centroid:clap".
        self.assertEqual(seen, [(pairs, "centroid:clap")])

    def test_main_window_plays_track_file(self):
        from unittest import mock

        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            with mock.patch("app.ui.main_window.open_in_system_player",
                            return_value=True) as opener:
                win._on_play_track(self.track_id)
            opener.assert_called_once_with(str(self.wav))
            self.assertIn("Playing song.wav", win.statusBar().currentMessage())
        finally:
            win.close()

    def test_main_window_play_failure_reports_hint(self):
        from unittest import mock

        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            with mock.patch("app.ui.main_window.open_in_system_player",
                            return_value=False):
                win._on_play_track(self.track_id)
            self.assertIn("no default player",
                          win.statusBar().currentMessage())
        finally:
            win.close()

    def test_main_window_create_m3u_and_play(self):
        from unittest import mock

        from app.ui.main_window import MainWindow
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            win._similar_seed = self.track_id
            pairs = [(self.track_id, 0.9)]
            m3u_target = self.dir / "mix.m3u"
            with mock.patch("app.ui.main_window.default_m3u_path",
                            return_value=m3u_target), \
                 mock.patch("app.ui.main_window.open_in_system_player",
                            return_value=True) as opener:
                win._on_create_playlist_and_play(pairs, "auto")
            opener.assert_called_once_with(str(m3u_target))
            # The .m3u exists on disk and the playlist row persisted.
            self.assertTrue(m3u_target.exists())
            self.assertIn("#EXTM3U", m3u_target.read_text(encoding="utf-8"))
            self.assertIn(str(self.wav),
                          m3u_target.read_text(encoding="utf-8"))
            with self.db.transaction() as conn:
                playlists = repo.list_playlists(conn)
            self.assertEqual(len(playlists), 1)
            self.assertIn("Playing playlist", win.statusBar().currentMessage())
        finally:
            win.close()


class SkipLongFilesTests(UiTestBase):
    """Settings checkbox + analysis pre-pass skip for >20-minute files."""

    LONG_SEC = 25.0 * 60.0     # 25 minutes
    SHORT_SEC = 5.0 * 60.0     # 5 minutes

    def _add_track_with_duration(self, name: str, duration) -> int:
        path = self.dir / name
        sr = 8000
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 *
                                    np.linspace(0, 0.5, sr // 2,
                                                endpoint=False)), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": name, "extension": ".wav", "codec": "pcm_s16le",
                 "sample_rate": sr, "channels": 1,
                 "duration_sec": duration,
                 "size_bytes": path.stat().st_size})

    def _worker(self, ids, skip: bool, force: bool = False):
        from app.ui.workers import AnalysisWorker
        config = AppConfig()
        config.use_ollama = False
        config.analysis_skip_long_files = skip
        return AnalysisWorker(self.db_path, config, ids,
                              force_reanalyze=force)

    def test_long_files_are_skipped_by_the_pre_pass(self):
        long_id = self._add_track_with_duration("long.wav", self.LONG_SEC)
        short_id = self._add_track_with_duration("short.wav", self.SHORT_SEC)
        unknown_id = self._add_track_with_duration("unknown.wav", None)
        worker = self._worker([long_id, short_id, unknown_id], skip=True)
        startable, skipped, skipped_long = worker._partition_requested()
        # The long file drops out; unknown duration never skips.
        self.assertEqual(startable, [short_id, unknown_id])
        self.assertEqual((skipped, skipped_long), (0, 1))

    def test_setting_off_and_force_reanalyze_keep_long_files(self):
        long_id = self._add_track_with_duration("long.wav", self.LONG_SEC)
        # Baseline: the skip is on, so the long file drops out of the batch.
        startable, _, skipped_long = self._worker([long_id], skip=True)._partition_requested()
        self.assertEqual((startable, skipped_long), ([], 1))
        # Explicit single-file re-analysis overrides the skip...
        startable, _, skipped_long = self._worker(
            [long_id], skip=True, force=True)._partition_requested()
        self.assertEqual((startable, skipped_long), ([long_id], 0))
        # ...and so does leaving the checkbox off.
        startable, _, skipped_long = self._worker([long_id], skip=False)._partition_requested()
        self.assertEqual((startable, skipped_long), ([long_id], 0))

    def test_run_announces_skipped_long_count(self):
        from unittest import mock
        from PySide6.QtCore import Qt

        long_id = self._add_track_with_duration("long.wav", self.LONG_SEC)
        short_id = self._add_track_with_duration("short.wav", self.SHORT_SEC)
        worker = self._worker([long_id, short_id], skip=True)
        seen_long = []
        worker.skipped_long.connect(seen_long.append,
                                    Qt.ConnectionType.DirectConnection)
        with mock.patch("app.analysis.pipeline.analyze_track",
                        lambda *args, **kwargs: None):
            worker.run()
        self.assertEqual(seen_long, [1])

    def test_settings_checkbox_roundtrip(self):
        from app.ui.settings_dialog import SettingsDialog
        config = AppConfig()
        config.analysis_skip_long_files = True
        dialog = SettingsDialog(config)
        self.assertTrue(dialog._skip_long.isChecked())
        dialog._skip_long.setChecked(False)
        self.assertFalse(dialog.apply().analysis_skip_long_files)


class AnalysisHighlightTests(UiTestBase):
    """Yellow highlight for tracks under analysis + their folder chain."""

    def _add_nested_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 *
                                    np.linspace(0, 0.5, sr // 2,
                                                endpoint=False)), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": ".wav",
                 "codec": "pcm_s16le", "sample_rate": sr, "channels": 1,
                 "duration_sec": 0.5, "size_bytes": path.stat().st_size})

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _item_by_path(self, tree, path: str):
        from app.ui.folder_tree import TRACK_ROLE
        for item in self._walk(tree):
            if item.data(0, TRACK_ROLE) is not None \
                    and str(item.toolTip(0)) == path:
                return item
        return None

    def _item_by_name(self, tree, name: str):
        for item in self._walk(tree):
            if item.text(0) == name:
                return item
        return None

    def test_analyzing_track_and_folder_chain_turn_yellow(self):
        from app.ui.folder_tree import _ACTIVITY_BRUSH, FolderTree
        deep_id = self._add_nested_track("sub/dir/deep/deep_song.wav")
        self._add_nested_track("sub/other.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        track_path = str(self.dir / "sub/dir/deep/deep_song.wav")
        tree.set_analyzing_paths([track_path])
        yellow = _ACTIVITY_BRUSH.color()
        # the file itself, every ancestor folder, and the root are yellow…
        for name in ("deep_song.wav", "deep", "dir", "sub"):
            item = self._item_by_name(tree, name)
            self.assertIsNotNone(item, name)
            self.assertEqual(item.background(0).color(), yellow, name)
        # …while everything unrelated stays unpainted
        other = self._item_by_path(tree, str(self.dir / "sub/other.wav"))
        self.assertNotEqual(other.background(0).color(), yellow)
        # clearing the set clears every highlight
        tree.set_analyzing_paths([])
        for name in ("deep_song.wav", "deep", "dir", "sub"):
            self.assertNotEqual(
                self._item_by_name(tree, name).background(0).color(), yellow)
        del deep_id

    def test_highlight_survives_refresh_and_status_updates(self):
        from app.ui.folder_tree import _ACTIVITY_BRUSH, FolderTree
        track_id = self._add_nested_track("sub/dir/deep/deep_song.wav")
        tree = FolderTree(self.db)
        tree.set_analyzing_paths([str(self.dir / "sub/dir/deep/deep_song.wav")])
        tree.refresh()
        yellow = _ACTIVITY_BRUSH.color()
        item = self._item_by_name(tree, "deep_song.wav")
        self.assertEqual(item.background(0).color(), yellow)   # rebuilt item
        # in-place status rewrites must not erase the highlight
        tree.update_track_status(track_id, "analyzing", None)
        self.assertEqual(item.background(0).color(), yellow)

    def test_main_window_lights_and_clears_highlight(self):
        from app.ui.folder_tree import _ACTIVITY_BRUSH, FolderTree
        from app.ui.main_window import MainWindow
        track_id = self._add_nested_track("sub/dir/deep/deep_song.wav")
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            path = str(self.dir / "sub/dir/deep/deep_song.wav")
            win._on_track_analysis_started(track_id, path)
            yellow = _ACTIVITY_BRUSH.color()
            self.assertEqual(
                self._item_by_name(win._tree,
                                   "deep_song.wav").background(0).color(),
                yellow)
            self.assertEqual(
                self._item_by_name(win._tree, "dir").background(0).color(),
                yellow)
            win._on_track_analysis_finished(track_id, True, "Analyzed")
            self.assertNotEqual(
                self._item_by_name(win._tree,
                                   "deep_song.wav").background(0).color(),
                yellow)
            self.assertNotEqual(
                self._item_by_name(win._tree, "dir").background(0).color(),
                yellow)
        finally:
            win.close()


class FolderReferencesTests(UiTestBase):
    """'Add folder…' adds the selected tree folder's tracks as references."""

    def _add_nested_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 *
                                    np.linspace(0, 0.5, sr // 2,
                                                endpoint=False)), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": ".wav",
                 "codec": "pcm_s16le", "sample_rate": sr, "channels": 1,
                 "duration_sec": 0.5, "size_bytes": path.stat().st_size})

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _item_by_name(self, tree, name: str):
        for item in self._walk(tree):
            if item.text(0) == name:
                return item
        return None

    def _window(self):
        from app.ui.main_window import MainWindow

        config = AppConfig()
        config.use_ollama = False
        return MainWindow(config, db_path=self.db_path)

    def test_folder_selection_adds_whole_subtree(self) -> None:
        outside = self._add_nested_track("top.wav")
        in_folder = self._add_nested_track("sub/b.wav")
        also_folder = self._add_nested_track("sub/c.wav")
        win = self._window()
        try:
            tree = win._tree
            tree.setCurrentItem(self._item_by_name(tree, "sub"))
            win._on_add_folder_references()
            seeds = win._details._seed_ids()
            self.assertIn(in_folder, seeds)
            self.assertIn(also_folder, seeds)
            self.assertNotIn(outside, seeds)
        finally:
            win.close()

    def test_file_selection_uses_parent_folder(self) -> None:
        self._add_nested_track("top.wav")
        in_folder = self._add_nested_track("sub/b.wav")
        also_folder = self._add_nested_track("sub/c.wav")
        win = self._window()
        try:
            tree = win._tree
            tree.setCurrentItem(self._item_by_name(tree, "b.wav"))
            win._on_add_folder_references()
            seeds = win._details._seed_ids()
            self.assertEqual(sorted(seeds), sorted([in_folder, also_folder]))
        finally:
            win.close()

    def test_cap_refuses_oversized_folder(self) -> None:
        self._add_nested_track("sub/b.wav")
        self._add_nested_track("sub/c.wav")
        win = self._window()
        try:
            win.MAX_FOLDER_REFERENCES = 1
            tree = win._tree
            tree.setCurrentItem(self._item_by_name(tree, "sub"))
            win._on_add_folder_references()
            self.assertEqual(win._details._seed_ids(), [])
            self.assertIn("capped", win.statusBar().currentMessage())
        finally:
            win.close()

    def test_no_folder_selection_status(self) -> None:
        win = self._window()
        try:
            win._on_add_folder_references()
            self.assertIn("Select a folder", win.statusBar().currentMessage())
        finally:
            win.close()


class RefreshFocusTests(UiTestBase):
    """refresh() keeps the user's place: selection, expansion, focus."""

    def _add_nested_track(self, rel_path: str) -> int:
        path = self.dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sr = 8000
        sf.write(path, 0.3 * np.sin(2 * np.pi * 440 *
                                    np.linspace(0, 0.5, sr // 2,
                                                endpoint=False)), sr)
        with self.db.transaction() as conn:
            return repo.upsert_track(
                conn, self.folder_id, str(path),
                {"filename": path.name, "extension": ".wav",
                 "codec": "pcm_s16le", "sample_rate": sr, "channels": 1,
                 "duration_sec": 0.5, "size_bytes": path.stat().st_size})

    @staticmethod
    def _walk(tree):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _item_by_name(self, tree, name: str):
        for item in self._walk(tree):
            if item.text(0) == name:
                return item
        return None

    def test_refresh_keeps_selected_track_and_expansion(self):
        from app.ui.folder_tree import FolderTree
        deep_id = self._add_nested_track("sub/dir/deep/deep_song.wav")
        self._add_nested_track("sub/other.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        deep = self._item_by_name(tree, "deep")
        deep.setExpanded(True)
        tree.select_track(deep_id)
        self.assertEqual(tree.selected_track_id(), deep_id)

        tree.refresh()      # e.g. _on_all_analyzed after a run

        self.assertEqual(tree.selected_track_id(), deep_id)
        self.assertIs(tree.currentItem(),
                      self._item_by_name(tree, "deep_song.wav"))
        self.assertTrue(self._item_by_name(tree, "deep").isExpanded())
        self.assertTrue(self._item_by_name(tree, "dir").isExpanded())

    def test_refresh_keeps_folder_selection(self):
        from app.ui.folder_tree import DIR_ROLE, FolderTree
        self._add_nested_track("sub/dir/deep/deep_song.wav")
        tree = FolderTree(self.db)
        tree.refresh()
        folder = self._item_by_name(tree, "dir")
        tree.setCurrentItem(folder)
        folder_path = str(folder.data(0, DIR_ROLE))

        tree.refresh()

        current = tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(str(current.data(0, DIR_ROLE)), folder_path)

    def test_first_refresh_still_expands_roots_only(self):
        from app.ui.folder_tree import FolderTree
        self._add_nested_track("sub/dir/deep/deep_song.wav")
        tree = FolderTree(self.db)
        tree.refresh()      # no previous state: default root-level expansion
        self.assertTrue(tree.topLevelItem(0).isExpanded())
        self.assertFalse(self._item_by_name(tree, "sub").isExpanded())


class TagEditorRemovalTests(UiTestBase):
    """The per-file tag editor was removed; chunk tags are view-only."""

    def test_detail_pane_has_no_tag_editor(self):
        from app.ui.detail_pane import DetailPane
        pane = DetailPane(self.db, AppConfig())
        self.assertFalse(hasattr(pane, "_edit_tags_button"))
        self.assertFalse(hasattr(pane, "tags_edited"))
        self.assertFalse(hasattr(pane, "_on_edit_tags"))

    def test_tag_editor_module_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            import app.ui.tag_editor  # noqa: F401


class VisualisationWindowTests(UiTestBase):
    """The toolbar Visualisation action opens the scatter-plot dialog."""

    def _add_fft_chunks(self, track_id: int) -> None:
        with self.db.transaction() as conn:
            chunk_ids = repo.replace_chunks(
                conn, track_id, [(0, 0.0, 10.0), (1, 10.0, 20.0)])
            for idx, chunk_id in enumerate(chunk_ids):
                repo.add_chunk_embedding(
                    conn, chunk_id, "fft",
                    np.arange(4, dtype=np.float32) + idx)

    def test_open_visualisation_plots_and_reuses_dialog(self):
        from app.ui.main_window import MainWindow
        self._add_fft_chunks(self.track_id)
        config = AppConfig()
        config.use_ollama = False
        win = MainWindow(config, db_path=self.db_path)
        try:
            win.open_visualisation()
            dialog = win._viz_dialog
            self.assertIsNotNone(dialog)
            self.assertTrue(dialog.isVisible())
            self.assertEqual(dialog._canvas.point_count, 2)
            self.assertIn("FFT[0]", dialog._status_label.text())
            first = dialog
            win.open_visualisation()      # same window, fresh data
            self.assertIs(win._viz_dialog, first)
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
