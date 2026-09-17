"""Tests for the MERT windowed-inference splitter (pure logic, no torch)."""
from __future__ import annotations

import unittest

import numpy as np

from app.models.mert_model import split_windows


class TestSplitWindows(unittest.TestCase):
    def test_short_input_single_window(self):
        x = np.arange(10, dtype=np.float32)
        windows = split_windows(x, max_samples=20, overlap_samples=2)
        self.assertEqual(len(windows), 1)
        np.testing.assert_array_equal(windows[0], x)

    def test_exact_fit_single_window(self):
        x = np.arange(20, dtype=np.float32)
        windows = split_windows(x, max_samples=20, overlap_samples=2)
        self.assertEqual(len(windows), 1)

    def test_long_input_all_windows_full_and_cover_tail(self):
        x = np.arange(1000, dtype=np.float32)
        windows = split_windows(x, max_samples=100, overlap_samples=10)
        self.assertGreater(len(windows), 1)
        for w in windows:
            self.assertEqual(len(w), 100)
        self.assertEqual(int(windows[0][0]), 0)
        self.assertEqual(int(windows[-1][-1]), 999)  # tail fully covered

    def test_small_tail_merged_into_previous_window(self):
        x = np.arange(210, dtype=np.float32)
        windows = split_windows(x, max_samples=100, overlap_samples=50)
        # step=50: starts 0,50,100; tail_start=110 is only 10 away -> shifted
        self.assertEqual(len(windows), 3)
        self.assertEqual(int(windows[-1][0]), 110)
        self.assertEqual(int(windows[-1][-1]), 209)

    def test_realistic_mps_case(self):
        # 20 s @ 24 kHz must split into windows of <= 10 s
        x = np.arange(480_000, dtype=np.float32)
        windows = split_windows(x, max_samples=240_000, overlap_samples=24_000)
        self.assertGreaterEqual(len(windows), 2)
        self.assertTrue(all(len(w) <= 240_000 for w in windows))
        self.assertEqual(int(windows[-1][-1]), 479_999)


if __name__ == "__main__":
    unittest.main()
