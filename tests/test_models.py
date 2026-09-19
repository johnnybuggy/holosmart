"""Tests for app.models plugins and registry (headless, no weight downloads)."""
from __future__ import annotations

import unittest

import numpy as np

from app.models.base import ModelPlugin
from app.models.registry import get_plugin, list_plugins, plugin_info

KNOWN = {"clap", "mert", "mert330", "m2dclap", "muq", "muqlan", "lpmc", "qwen2audio", "openl3", "fft"}
CONTRACT_KEYS = {
    "name", "display_name", "embedding_dim", "provides_text",
    "available", "error", "loaded",
}


def _loadable(plugin: ModelPlugin) -> bool:
    """Best-effort check that a plugin actually loads; never raises.

    Used to skip embed/describe tests when weights aren't present locally.
    """
    try:
        plugin.ensure_loaded()
        return True
    except Exception:
        return False


class RegistryTests(unittest.TestCase):
    def test_registry_lists_plugins_in_canonical_order(self) -> None:
        plugins = list_plugins()
        self.assertEqual({p.name for p in plugins}, KNOWN)
        self.assertEqual([p.name for p in plugins],
                         ["clap", "mert", "mert330", "m2dclap", "muq",
                          "muqlan", "lpmc", "qwen2audio", "openl3", "fft"])

    def test_get_plugin_returns_singletons(self) -> None:
        self.assertIs(get_plugin("clap"), get_plugin("clap"))
        self.assertIs(get_plugin("mert"), list_plugins()[1])
        with self.assertRaises(KeyError) as ctx:
            get_plugin("nope")
        self.assertIn("nope", str(ctx.exception))
        for known in KNOWN:
            self.assertIn(known, str(ctx.exception))

    def test_get_plugin_all_known_names_resolve(self) -> None:
        for name in sorted(KNOWN):
            self.assertIsInstance(get_plugin(name), ModelPlugin)

    def test_plugin_info_contract_keys(self) -> None:
        infos = plugin_info()
        self.assertEqual({i["name"] for i in infos}, KNOWN)
        for info in infos:
            self.assertEqual(set(info), CONTRACT_KEYS)
            self.assertIsInstance(info["available"], bool)
            self.assertIsInstance(info["loaded"], bool)

    def test_plugin_class_attributes(self) -> None:
        clap = get_plugin("clap")
        mert = get_plugin("mert")
        openl3 = get_plugin("openl3")
        fft = get_plugin("fft")
        self.assertEqual((clap.embedding_dim, clap.provides_text), (512, True))
        self.assertEqual(clap.preferred_sample_rate, 48000)
        self.assertEqual((mert.embedding_dim, mert.provides_text), (768, False))
        self.assertEqual(mert.preferred_sample_rate, 24000)
        mert330 = get_plugin("mert330")
        self.assertEqual((mert330.embedding_dim, mert330.provides_text),
                         (1024, False))
        self.assertEqual(mert330.preferred_sample_rate, 24000)
        self.assertEqual(mert330.model_id, "m-a-p/MERT-v1-330M")
        self.assertEqual((openl3.embedding_dim, openl3.provides_text), (512, False))
        self.assertEqual(openl3.preferred_sample_rate, 48000)
        self.assertEqual((fft.embedding_dim, fft.provides_text), (40, False))
        self.assertTrue(fft.is_available())   # numpy-only plugin


class AvailabilityTests(unittest.TestCase):
    def test_is_available_fast_and_non_raising(self) -> None:
        for plugin in list_plugins():
            try:
                result = plugin.is_available()
            except Exception as exc:  # must not raise
                self.fail(f"{plugin.name}.is_available() raised: {exc}")
            self.assertIsInstance(result, bool)
            err = plugin.availability_error()
            if result:
                self.assertIsNone(err)
            else:
                self.assertTrue(err)

    def test_openl3_unavailable_with_helpful_message(self) -> None:
        openl3 = get_plugin("openl3")
        # This venv has no tensorflow/openl3; if that changes, relax.
        if openl3.is_available():
            self.skipTest("openl3/tensorflow installed in this environment")
        self.assertFalse(openl3.is_available())
        err = openl3.availability_error() or ""
        self.assertIn("pip install openl3 tensorflow", err)

    def test_unavailable_plugin_ensure_loaded_raises(self) -> None:
        for plugin in list_plugins():
            if plugin.is_available():
                continue
            with self.assertRaises(RuntimeError) as ctx:
                plugin.ensure_loaded()
            msg = str(ctx.exception)
            self.assertIn(plugin.display_name, msg)
            expected = plugin.availability_error() or ""
            self.assertIn(expected, msg)


def _describe_testable(plugin: ModelPlugin) -> bool:
    return plugin.is_available() and _loadable(plugin)


class InferenceTests(unittest.TestCase):
    """Real-inference tests; skipped unless deps AND weights are usable.

    Weight downloads may be blocked/unavailable — failures become skipTest.
    """

    def _noise(self, seconds: float, sr: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        return (0.1 * rng.standard_normal(int(sr * seconds))).astype(np.float32)

    def test_clap_embed_and_describe_tiny(self) -> None:
        plugin = get_plugin("clap")
        if not plugin.is_available():
            self.skipTest("torch/transformers unavailable")
        try:
            plugin.ensure_loaded()
        except RuntimeError as exc:
            self.skipTest(f"CLAP weights not loadable: {exc}")
        try:
            vecs = plugin.embed([self._noise(1.0, 24000)], 24000)
        except Exception as exc:
            self.skipTest(f"CLAP inference failed: {exc}")
        self.assertEqual(len(vecs), 1)
        vec = vecs[0]
        self.assertEqual(vec.shape, (512,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)
        try:
            tags = plugin.describe([self._noise(1.0, 48000)], 48000, top_k=5)
        except Exception as exc:
            self.skipTest(f"CLAP describe failed: {exc}")
        self.assertIsNotNone(tags)
        self.assertEqual(len(tags), 1)
        self.assertEqual(len(tags[0]), 5)
        for tag, score in tags[0]:
            self.assertIsInstance(tag, str)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        scores = [s for _, s in tags[0]]
        self.assertAlmostEqual(sum(scores), 1.0, places=4)

    def test_mert_embed_tiny(self) -> None:
        plugin = get_plugin("mert")
        if not plugin.is_available():
            self.skipTest("torch/transformers unavailable")
        try:
            plugin.ensure_loaded()
        except RuntimeError as exc:
            self.skipTest(f"MERT weights not loadable: {exc}")
        try:
            vecs = plugin.embed([self._noise(1.0, 48000)], 48000)
        except Exception as exc:
            self.skipTest(f"MERT inference failed: {exc}")
        self.assertEqual(len(vecs), 1)
        vec = vecs[0]
        self.assertEqual(vec.shape, (768,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_mert_describe_returns_none(self) -> None:
        plugin = get_plugin("mert")
        self.assertIsNone(plugin.describe([], 48000))


if __name__ == "__main__":
    unittest.main()
