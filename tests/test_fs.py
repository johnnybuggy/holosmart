"""Tests for filesystem helpers (music walk, access checks)."""
from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from app.fs_utils import check_read_access, walk_music_files


class TestWalkMusicFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # nested structure: root/a/one.wav, root/b/c/two.mp3, root/ignore.txt
        (self.root / "a").mkdir()
        (self.root / "b" / "c").mkdir(parents=True)
        (self.root / "a" / "one.wav").write_bytes(b"x")
        (self.root / "b" / "c" / "two.mp3").write_bytes(b"x")
        (self.root / "b" / "flac_three.flac").write_bytes(b"x")
        (self.root / "ignore.txt").write_bytes(b"x")
        (self.root / "no_ext").write_bytes(b"x")

    def tearDown(self):
        self._tmp.cleanup()

    def test_finds_supported_files_recursively_sorted(self):
        files, denied = walk_music_files(self.root)
        # sorted by full path: a/one.wav, b/c/two.mp3, b/flac_three.flac
        self.assertEqual([f.name for f in files],
                         ["one.wav", "two.mp3", "flac_three.flac"])
        self.assertEqual(denied, [])

    def test_single_file_root(self):
        files, denied = walk_music_files(self.root / "a" / "one.wav")
        self.assertEqual([f.name for f in files], ["one.wav"])
        self.assertEqual(denied, [])

    def test_denied_subdir_reported_but_rest_still_scanned(self):
        secret = self.root / "b" / "c"
        secret.chmod(0)
        try:
            if os.access(secret, os.R_OK):  # running as root: chmod is bypassed
                self.skipTest("chmod-based denial not effective for this user")
            files, denied = walk_music_files(self.root)
            # 'c' is denied (hidden.mp3 not found); readable b/ still yields its file
            self.assertEqual(sorted(f.name for f in files),
                             ["flac_three.flac", "one.wav"])
            self.assertEqual(len(denied), 1)
            self.assertEqual(Path(denied[0][0]).name, "c")
            self.assertIn("Permission denied", denied[0][1])
        finally:
            secret.chmod(stat.S_IRWXU)


class TestCheckReadAccess(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_readable_directory(self):
        ok, error = check_read_access(self.dir)
        self.assertTrue(ok)
        self.assertEqual(error, "")

    def test_missing_path(self):
        ok, error = check_read_access(self.dir / "does_not_exist")
        self.assertFalse(ok)
        self.assertTrue(error)

    def test_denied_directory(self):
        self.dir.chmod(0)
        try:
            if os.access(self.dir, os.R_OK):
                self.skipTest("chmod-based denial not effective for this user")
            ok, error = check_read_access(self.dir)
            self.assertFalse(ok)
            self.assertIn("Permission denied", error)
        finally:
            self.dir.chmod(stat.S_IRWXU)


if __name__ == "__main__":
    unittest.main()
