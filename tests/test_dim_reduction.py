"""Tests for app.analysis.dim_reduction (PCA built-in; t-SNE/UMAP optional)."""
from __future__ import annotations

import unittest

import numpy as np

from app.analysis import dim_reduction as dr


class PcaTests(unittest.TestCase):
    def test_variance_ordering_and_labels(self) -> None:
        rng = np.random.default_rng(1)
        # Columns deliberately scaled: variances 25, 9, then noise.
        x = rng.normal(size=(400, 6)) * np.array([5.0, 3.0, 1.0, .5, .2, .1])
        coords, labels = dr.pca_2d(x)
        self.assertEqual(coords.shape, (400, 2))
        self.assertGreater(float(np.var(coords[:, 0])),
                           float(np.var(coords[:, 1])))
        self.assertEqual(len(labels), 2)
        self.assertTrue(labels[0].startswith("PC1 ("))
        self.assertTrue(labels[1].startswith("PC2 ("))
        # Both variance ratios parse as percentages summing to <= 100.
        ratios = [float(label[label.index("(") + 1: -2].rstrip())
                  for label in labels]
        self.assertGreaterEqual(ratios[0], ratios[1])
        self.assertLessEqual(sum(ratios), 100.0 + 1e-9)

    def test_principal_axis_aligns_with_dominant_direction(self) -> None:
        # Points on a noisy line along (1, 1): PC1 should carry ~all variance.
        t = np.linspace(-1, 1, 200)
        x = np.stack([t, t], axis=1) + 1e-9
        coords, labels = dr.pca_2d(x)
        ratio = float(labels[0][labels[0].index("(") + 1: -2])
        self.assertGreater(ratio, 99.0)
        # After centering, PC1 coordinates lie on a line: y ≈ ±x.
        slope = float(np.polyfit(coords[:, 0], coords[:, 1], 1)[0])
        self.assertLess(abs(slope), 1e-6)

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(50, 4))
        c1, _ = dr.pca_2d(x)
        c2, _ = dr.pca_2d(x)
        np.testing.assert_allclose(c1, c2)

    def test_handles_wide_and_flat_input(self) -> None:
        # n < d (wide matrix) still yields an (n, 2) result.
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                      [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]])
        coords, _ = dr.pca_2d(x)
        self.assertEqual(coords.shape, (2, 2))
        # All-identical points: no variance, zero coordinates, "0.0%".
        coords, labels = dr.pca_2d(np.ones((5, 3)))
        np.testing.assert_allclose(coords, 0.0)
        self.assertEqual(labels, ["PC1 (0.0%)", "PC2 (0.0%)"])

    def test_rejects_bad_input(self) -> None:
        with self.assertRaises(dr.DimReductionError):
            dr.pca_2d(np.zeros(0))
        with self.assertRaises(dr.DimReductionError):
            dr.pca_2d(np.array([1.0, 2.0]))      # 1-D, not a matrix


class StandardizeTests(unittest.TestCase):
    def test_zero_mean_unit_variance_and_flat_columns(self) -> None:
        x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
        out = dr.standardize(x)
        self.assertAlmostEqual(float(out[:, 0].mean()), 0.0, places=12)
        self.assertAlmostEqual(float(out[:, 0].std()), 1.0, places=12)
        np.testing.assert_allclose(out[:, 1], 0.0)   # zero-variance column


class OptionalMethodTests(unittest.TestCase):
    """t-SNE / UMAP raise a friendly DimReductionError without their libs."""

    def test_missing_dependency_message(self) -> None:
        # Simulate absence regardless of the local environment.  Submodules
        # must be purged too: with scikit-learn really installed, a cached
        # ``sklearn.manifold`` would satisfy the import even with
        # ``sys.modules["sklearn"] = None``.
        import sys
        x = np.random.default_rng(0).normal(size=(40, 4))
        for name, fn in (("sklearn", dr.tsne_2d), ("umap", dr.umap_2d)):
            with self.subTest(name=name):
                saved = {k: v for k, v in sys.modules.items()
                         if k == name or k.startswith(name + ".")}
                try:
                    for key in saved:
                        sys.modules.pop(key, None)
                    sys.modules[name] = None   # forces the import to fail
                    with self.assertRaises(dr.DimReductionError) as ctx:
                        fn(x)
                    self.assertIn("pip install", str(ctx.exception))
                finally:
                    sys.modules.pop(name, None)
                    for key, value in saved.items():
                        sys.modules[key] = value

    def test_availability_table(self) -> None:
        table = dr.availability()
        for method in ("pca", "tsne", "umap"):
            self.assertIn(method, table)
        self.assertIsNone(table["pca"])
        for method in ("tsne", "umap"):
            # Either available (None) or a user-presentable install hint.
            if table[method] is not None:
                self.assertIn("pip install", table[method])

    def test_reduce_2d_dispatch_and_unknown(self) -> None:
        x = np.random.default_rng(0).normal(size=(30, 3))
        coords, _ = dr.reduce_2d(x, "pca")
        self.assertEqual(coords.shape, (30, 2))
        with self.assertRaises(dr.DimReductionError):
            dr.reduce_2d(x, "wav2vec")


if __name__ == "__main__":
    unittest.main()
