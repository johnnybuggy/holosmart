"""Tests for the FFT spectral-statistics plugin (numpy-only, fully headless).

Covers the band/spectral math helpers, the plugin's embed contract, windowing,
configuration, the one-time config migration that enables the plugin in older
installations, and the similarity-search integration over stored "fft" vectors.

Feature-layout reminder (see app/models/fft_model.py): for each band b in
``BANDS`` order, dims ``b*6 .. b*6+5`` hold (mean amplitude, std, skew,
excess kurtosis, RMS, crest factor) of the band's bin amplitudes relative to
the window's overall RMS; the last four dims hold (dominant frequency,
spectral centroid — both relative to Nyquist —, relative Shannon entropy,
Hurst exponent).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.config import AppConfig
from app.db import repo
from app.db.database import Database
from app.models.base import ModelPlugin
from app.models.fft_model import (
    BANDS,
    COMPRESS_CLIP,
    FEATURE_DIM,
    FftPlugin,
    band_indices,
    compress_and_normalize,
    hurst_exponent,
    moment_stats,
    shannon_entropy,
    spectral_features,
)
from app.models.registry import get_plugin, list_plugins, plugin_info


def _sine(freq: float, seconds: float, sr: int = 48000,
          amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seconds: float, sr: int = 48000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(int(sr * seconds))).astype(np.float32)


class RegistryTests(unittest.TestCase):
    def test_registry_contains_fft_in_canonical_order(self) -> None:
        plugins = list_plugins()
        self.assertEqual([p.name for p in plugins],
                         ["clap", "mert", "mert330", "m2dclap", "muq",
                          "muqlan", "lpmc", "qwen2audio", "openl3", "fft"])

    def test_get_plugin_fft_is_singleton(self) -> None:
        self.assertIs(get_plugin("fft"), get_plugin("fft"))
        self.assertIsInstance(get_plugin("fft"), ModelPlugin)

    def test_plugin_info_reports_fft_available(self) -> None:
        info = {entry["name"]: entry for entry in plugin_info()}
        self.assertIn("fft", info)
        self.assertTrue(info["fft"]["available"])
        self.assertEqual(info["fft"]["embedding_dim"], FEATURE_DIM)
        self.assertFalse(info["fft"]["provides_text"])


class HelperMathTests(unittest.TestCase):
    def test_band_indices_masks_half_open_ranges(self) -> None:
        freqs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        self.assertEqual(
            list(np.nonzero(band_indices(freqs, 10.0, 30.0))[0]), [1, 2])
        self.assertEqual(
            list(np.nonzero(band_indices(freqs, 30.0, None))[0]), [3, 4])

    def test_moment_stats_known_sample(self) -> None:
        sample = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
        mean, std, skew, kurt = moment_stats(sample)
        d = sample - sample.mean()
        self.assertAlmostEqual(mean, 1.5)
        self.assertAlmostEqual(std, float(np.sqrt(np.mean(d ** 2))))
        self.assertAlmostEqual(skew, 0.0)
        self.assertAlmostEqual(
            kurt, float(np.mean(d ** 4)) / float(np.mean(d ** 2)) ** 2 - 3.0)

    def test_moment_stats_degenerate_inputs(self) -> None:
        self.assertEqual(moment_stats([]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(moment_stats([7.0, 7.0, 7.0]), (7.0, 0.0, 0.0, 0.0))

    def test_shannon_entropy_uniform_and_silent(self) -> None:
        p = np.full(8, 1.0 / 8)
        self.assertAlmostEqual(shannon_entropy(p), 3.0, places=6)
        self.assertEqual(shannon_entropy(np.zeros(10)), 0.0)

    def test_hurst_exponent_ranges(self) -> None:
        # White noise sits near the random-walk baseline; a linear ramp
        # (perfectly smooth) approaches the persistent maximum.
        noise_h = hurst_exponent(_noise(1.0))
        self.assertGreater(noise_h, 0.25)
        self.assertLess(noise_h, 0.75)
        ramp = np.linspace(0.0, 1.0, 512)
        self.assertGreater(hurst_exponent(ramp), 0.85)
        self.assertEqual(hurst_exponent(np.zeros(64)), 0.5)   # zero variance
        self.assertEqual(hurst_exponent(np.arange(4)), 0.5)   # too short


class SpectralFeatureTests(unittest.TestCase):
    SR = 48000

    def test_tone_features_are_exact(self) -> None:
        feats = spectral_features(_sine(440.0, 10.0, self.SR), self.SR)
        self.assertEqual(len(feats), FEATURE_DIM)
        nyquist = self.SR / 2.0
        # dominant frequency + centroid land on the tone frequency
        self.assertAlmostEqual(feats[36] * nyquist, 440.0, delta=2.0)
        self.assertAlmostEqual(feats[37] * nyquist, 440.0, delta=5.0)
        # tonal spectrum -> low relative entropy
        self.assertLess(feats[38], 0.25)
        # smooth spectral envelope -> positive Hurst
        self.assertGreaterEqual(feats[39], 0.5)
        # band statistics: energy lives in the 300-1000 Hz band
        band_rms = [feats[b * 6 + 4] for b in range(6)]
        self.assertEqual(band_rms.index(max(band_rms)), 2)
        # ... and the tone band's per-bin RMS dwarfs the (leakage-only) rest
        self.assertGreater(band_rms[2], 5.0 * max(
            band_rms[i] for i in range(6) if i != 2))
        # bands without signal stay near zero
        self.assertAlmostEqual(band_rms[0], 0.0, places=4)
        self.assertAlmostEqual(band_rms[5], 0.0, places=4)
        # crest factor is peak/RMS and > 1 wherever energy exists
        self.assertGreater(feats[2 * 6 + 5], 1.0)
        # every value finite
        self.assertTrue(all(np.isfinite(v) for v in feats))

    def test_band_stats_match_manual_computation(self) -> None:
        x = _sine(440.0, 10.0, self.SR)
        feats = spectral_features(x, self.SR)
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) * 2.0 / len(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.SR)
        band = spectrum[band_indices(freqs, 300.0, 1000.0)]
        rms_all = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        self.assertAlmostEqual(feats[2 * 6 + 0],
                               float(band.mean()) / rms_all, places=9)
        self.assertAlmostEqual(feats[2 * 6 + 4],
                               float(np.sqrt(np.mean(band ** 2))) / rms_all,
                               places=9)

    def test_silent_window_is_all_zero(self) -> None:
        feats = spectral_features(np.zeros(480000, dtype=np.float32), 48000)
        self.assertEqual(feats, (0.0,) * FEATURE_DIM)

    def test_noise_features(self) -> None:
        x = _noise(10.0, self.SR)
        feats = spectral_features(x, self.SR)
        n_bins = len(np.fft.rfftfreq(len(x), d=1.0 / self.SR))
        # broad spectrum -> entropy close to the bin-count maximum
        self.assertGreater(feats[38], 0.75)
        # Hurst near the random-walk baseline for noise
        self.assertLess(feats[39], 0.8)
        # crest factor modest (no single dominant bin)
        self.assertLess(feats[0 * 6 + 5], 20.0)
        # white noise has a flat power spectral density: the per-bin RMS is
        # (approximately) the same in every band
        band_rms = [feats[b * 6 + 4] for b in range(6)]
        self.assertGreater(min(band_rms), 0.0005)
        self.assertLess(max(band_rms) / min(band_rms), 1.5)

    def test_all_values_finite_for_edge_inputs(self) -> None:
        for samples, sr in ((np.zeros(3, dtype=np.float32), 48000),
                            (_sine(440.0, 0.001), 48000),
                            (_sine(440.0, 1.0), 8000),
                            (_noise(2.0), 44100)):
            feats = spectral_features(samples, sr)
            self.assertEqual(len(feats), FEATURE_DIM)
            self.assertTrue(all(np.isfinite(v) for v in feats))


class CompressTests(unittest.TestCase):
    def test_compress_normalizes_to_unit_norm(self) -> None:
        raw = np.zeros(FEATURE_DIM)
        raw[2 * 6 + 4] = 1.0    # band RMS ratio
        raw[36] = 0.018         # dominant frequency / Nyquist
        raw[39] = 0.7           # Hurst
        vec = compress_and_normalize(raw)
        self.assertEqual(vec.shape, (FEATURE_DIM,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)
        self.assertGreater(float(vec[36]), 0.0)

    def test_compress_of_zeros_stays_zero(self) -> None:
        vec = compress_and_normalize(np.zeros(FEATURE_DIM))
        self.assertEqual(float(np.linalg.norm(vec)), 0.0)

    def test_compress_never_emits_nan_and_clips(self) -> None:
        raw = np.full(FEATURE_DIM, np.nan)
        self.assertTrue(np.all(np.isfinite(compress_and_normalize(raw))))
        raw = np.zeros(FEATURE_DIM)
        raw[5] = 1e6            # pathological crest factor
        vec = compress_and_normalize(raw)
        self.assertLessEqual(float(vec[5]), 1.0)   # clip + unit norm
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)


class PluginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plugin = FftPlugin()
        self.sr = 48000

    def test_window_sec_default_and_config_override(self) -> None:
        self.assertEqual(self.plugin.window_sec, 10.0)
        cfg = AppConfig()
        cfg.fft_window_sec = 5.0
        self.plugin.apply_config(cfg)
        self.assertEqual(self.plugin.window_sec, 5.0)
        cfg.fft_window_sec = 0.01          # below the minimum -> clamped
        self.plugin.apply_config(cfg)
        self.assertEqual(self.plugin.window_sec, 0.5)

    def test_split_windows_contiguous_and_tail_kept(self) -> None:
        samples = np.arange(int(self.sr * 25), dtype=np.float32)
        windows = self.plugin._split_chunk_windows(samples, self.sr)
        self.assertEqual([len(w) for w in windows],
                         [480000, 480000, 240000])
        # every sample counted exactly once
        self.assertEqual(sum(len(w) for w in windows), int(self.sr * 25))

    def test_embed_returns_one_vector_per_chunk(self) -> None:
        chunks = [_sine(440.0, 12.0), _sine(220.0, 7.0), _noise(11.0)]
        vecs = self.plugin.embed(chunks, self.sr)
        self.assertEqual(len(vecs), len(chunks))
        for vec in vecs:
            self.assertEqual(vec.shape, (FEATURE_DIM,))
            self.assertEqual(vec.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_silence_gives_zero_vector(self) -> None:
        vec = self.plugin.embed(
            [np.zeros(int(self.sr * 11), dtype=np.float32)], self.sr)[0]
        self.assertEqual(float(np.linalg.norm(vec)), 0.0)

    def test_tone_vector_content(self) -> None:
        raw = self.plugin.raw_chunk_vector(
            _sine(440.0, 10.0, self.sr), self.sr)
        # layout: band stats then whole-spectrum features
        self.assertAlmostEqual(float(raw[36]) * (self.sr / 2), 440.0,
                               delta=2.0)               # dominant frequency
        self.assertAlmostEqual(float(raw[37]) * (self.sr / 2), 440.0,
                               delta=5.0)               # spectral centroid
        self.assertLess(float(raw[38]), 0.25)           # relative entropy
        embedded = self.plugin.embed([_sine(440.0, 10.0, self.sr)], self.sr)[0]
        self.assertEqual(embedded.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(embedded)), 1.0, places=4)

    def test_window_sec_setting_changes_windows(self) -> None:
        self.plugin.window_sec = 5.0
        windows = self.plugin._split_chunk_windows(
            np.arange(int(self.sr * 12), dtype=np.float32), self.sr)
        self.assertEqual([len(w) for w in windows],
                         [240000, 240000, 96000])


class ConfigMigrationTests(unittest.TestCase):
    """One-time FFT auto-enable for config files written before the plugin."""

    def test_legacy_config_gains_fft_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            payload = {"models": ["mert"], "chunk_seconds": 20.0}
            (data_dir / "config.json").write_text(json.dumps(payload),
                                                  encoding="utf-8")
            with mock.patch("app.config.CONFIG_PATH", data_dir / "config.json"):
                loaded = AppConfig.load()
                self.assertIn("fft", loaded.models)
                # v3 migration: the M2D-CLAP/MuQ/MuQ-MuLan roster is appended
                # exactly once to this legacy config as well.
                for name in ("m2dclap", "muq", "muqlan"):
                    self.assertIn(name, loaded.models)
                # A deliberately disabled FFT (saved with models_version) stays.
                loaded.models = ["mert"]
                loaded.save()
                reloaded = AppConfig.load()
        self.assertNotIn("fft", reloaded.models)
        # ... and so is a deliberate opt-out of the v3/v4 rosters: after
        # this save (models_version = current) no migration re-appends.
        for name in ("m2dclap", "muq", "muqlan", "lpmc", "qwen2audio"):
            self.assertNotIn(name, reloaded.models)
        self.assertEqual(reloaded.models_version, AppConfig.models_version)

    def test_fresh_config_defaults_include_fft(self) -> None:
        cfg = AppConfig()
        self.assertIn("fft", cfg.models)
        self.assertEqual(cfg.fft_window_sec, 10.0)


class SimilarityIntegrationTests(unittest.TestCase):
    """The FFT feature vectors must behave like CLAP/MERT vectors downstream:
    chunk centroids become track vectors and drive similarity search."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="holosmart-fft-sim-")
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(Path(self._tmp.name) / "library.db")
        plugin = FftPlugin()
        with self.db.transaction() as conn:
            self.fid = repo.add_folder(conn, "/music")
            self.tone_id = repo.upsert_track(
                conn, self.fid, "/music/tone.wav", {"filename": "tone.wav"})
            self.noise_id = repo.upsert_track(
                conn, self.fid, "/music/noise.wav", {})
            for track_id, kind in ((self.tone_id, "sine"),
                                   (self.noise_id, "noise")):
                samples = _sine(440.0, 10.0) if kind == "sine" else _noise(10.0)
                vec = plugin.embed([samples], 48000)[0]
                ids = repo.replace_chunks(conn, track_id, [(0, 0.0, 10.0)])
                repo.add_chunk_embedding(conn, ids[0], "fft", vec)

    def test_similar_tracks_by_fft_centroid(self) -> None:
        from app.similarity.search import ensure_track_embeddings, similar_tracks

        with self.db.transaction() as conn:
            ensure_track_embeddings(conn, [self.tone_id, self.noise_id],
                                    None, None)
            centroid = repo.get_track_embedding(conn, self.tone_id, "fft")
            self.assertIsNotNone(centroid)
            self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0,
                                   places=4)
            results = similar_tracks(conn, self.tone_id, method="fft",
                                     limit=5)
        self.assertEqual([r.track_id for r in results],
                         [self.tone_id, self.noise_id])   # seed first
        self.assertEqual(results[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()