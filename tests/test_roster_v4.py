"""Tests for the roster-v4 plugins: LP-MusicCaps and Qwen2-Audio.

Both plugins are exercised against STUB models (no network, no real
weights): the embedding paths' windowing/pooling/normalization math and
the caption path's tag semantics are what matters here.  The vendored
LP-MusicCaps encoder is additionally tested for its real forward shape.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.models.lpmusiccaps_model import LPMC_MODEL_ID, LpMusicCapsPlugin
from app.models.qwen2audio_model import (
    QWEN2AUDIO_MODEL_ID, QWEN_WINDOW_SEC, Qwen2AudioPlugin)


def _l2(vec: np.ndarray) -> np.ndarray:
    return vec / max(np.linalg.norm(vec), 1e-12)


def _loaded(plugin, model, device="cpu") -> None:
    plugin._model = model
    plugin._device = device
    plugin._loaded = True


# --------------------------------------------------------------------------
# LP-MusicCaps
# --------------------------------------------------------------------------
class _FakeLpmc:
    """Stub BartCaptionModel: pooled mel-CNN vector + canned captions."""

    def __init__(self) -> None:
        self.batches: list[np.ndarray] = []

    def forward_encoder(self, wavs):
        import torch
        self.batches.append(wavs.numpy())
        head = wavs[:, :160].mean(dim=1, keepdim=True)
        tail = wavs[:, -160:].mean(dim=1, keepdim=True)
        pooled = torch.cat([head, tail], dim=1)      # (b, 2)
        return pooled.unsqueeze(1), pooled.unsqueeze(1)

    def generate(self, wavs, num_beams=5):
        n = wavs.shape[0]
        return [f"caption {i}" for i in range(n)]


class LpMusicCapsTests(unittest.TestCase):
    def test_roster_and_settings_defaults(self) -> None:
        plugin = LpMusicCapsPlugin()
        self.assertEqual(plugin.name, "lpmc")
        self.assertEqual(plugin.display_name, "LP-MusicCaps")
        self.assertEqual(plugin.embedding_dim, 768)
        self.assertTrue(plugin.provides_text)
        self.assertEqual(plugin.preferred_sample_rate, 16000)
        self.assertEqual(plugin.default_model_id, LPMC_MODEL_ID)
        # only deps that exist on a stock torch/transformers install
        self.assertEqual(plugin.requirements,
                         ("torch", "torchaudio", "transformers"))

    def test_apply_config_adopts_checkpoint_and_batch(self) -> None:
        plugin = LpMusicCapsPlugin()
        plugin.apply_config(type("Cfg", (), {
            "lpmc_model_id": "custom/lpmc",
            "lpmc_checkpoint_file": "supervised.pth",
            "lpmc_batch_size": 2}))
        self.assertEqual(plugin.model_id, "custom/lpmc")
        self.assertEqual(plugin.checkpoint_file, "supervised.pth")
        self.assertEqual(plugin.batch_size, 2)

    def test_embed_pools_windows_and_normalizes(self) -> None:
        plugin = LpMusicCapsPlugin()
        plugin.batch_size = 4
        _loaded(plugin, _FakeLpmc())
        # 25 s chunk @ 16 kHz -> 3 windows (10/10/5 s, last padded by the
        # batcher), each pooled to 2 dims, then averaged + L2-normalized
        sr = 16000
        chunk = np.concatenate([
            np.full(10 * sr, 0.5, dtype=np.float32),
            np.full(10 * sr, -0.5, dtype=np.float32),
            np.full(5 * sr, 0.25, dtype=np.float32)])
        vecs = plugin._embed([chunk], sr)
        self.assertEqual(vecs[0].shape, (2,))
        self.assertAlmostEqual(float(np.linalg.norm(vecs[0])), 1.0,
                               places=5)
        # captions: one per window
        tags = plugin.describe([chunk], sr)
        self.assertEqual(len(tags[0]), 3)
        self.assertEqual({text for text, _score in tags[0]},
                         {"caption 0", "caption 1", "caption 2"})
        # every stored caption tag carries score 1.0
        self.assertTrue(all(score == 1.0 for _t, score in tags[0]))

    def test_embed_empty_chunk_yields_zero_vector(self) -> None:
        plugin = LpMusicCapsPlugin()
        _loaded(plugin, _FakeLpmc())
        vecs = plugin._embed([np.zeros(0, dtype=np.float32)], 16000)
        self.assertEqual(vecs[0].shape, (768,))
        self.assertEqual(float(np.abs(vecs[0]).sum()), 0.0)


class VendorLpmcEncoderTests(unittest.TestCase):
    def test_audio_encoder_forward_shape(self) -> None:
        import torch

        from app.models.vendor.lpmc_bart import AudioEncoder

        # 5 stride-2 convs over ~1001 frames -> ~32 tokens of width 768
        encoder = AudioEncoder(n_mels=128, n_ctx=32, audio_dim=768,
                               text_dim=768, num_of_stride_conv=5)
        wavs = torch.zeros(2, 160000)     # 10 s @ 16 kHz
        out = encoder(wavs)
        self.assertEqual(tuple(out.shape)[0], 2)
        self.assertEqual(tuple(out.shape)[2], 768)
        self.assertLessEqual(tuple(out.shape)[1], 33)


# --------------------------------------------------------------------------
# Qwen2-Audio
# --------------------------------------------------------------------------
class _FakeTowerOut:
    def __init__(self, tensor):
        self.last_hidden_state = tensor


class _FakeTower:
    """Stub Qwen2AudioEncoder: pooled hidden = (head, tail, head-tail)."""

    dim = 3

    def __call__(self, input_features):
        import torch
        # input_features: (b, 128 mels, 3000 frames)
        head = input_features[:, :64, :].mean(dim=(1, 2), keepdim=True)
        tail = input_features[:, 64:, :].mean(dim=(1, 2), keepdim=True)
        # (n, time=1, dim=3): one frame per window, 3 features per frame
        frames = torch.cat([head, tail, head - tail], dim=2)
        return _FakeTowerOut(frames)


class _FakeFeatureExtractor:
    """Stub Whisper feature extractor: any waveform -> (128, 3000) mels."""

    def __call__(self, wavs, sampling_rate=16000, return_tensors="pt",
                 padding="max_length"):
        import numpy as np
        import torch

        arr = np.asarray(wavs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        feats = np.zeros((arr.shape[0], 128, 3000), dtype=np.float32)
        # encode head/tail means so the fake tower sees window content
        head = arr[:, :1000].mean(axis=1)[:, None, None]
        tail = arr[:, -1000:].mean(axis=1)[:, None, None]
        feats[:, :64, :] = head
        feats[:, 64:, :] = tail
        return {"input_features": torch.from_numpy(feats)}


class _FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = _FakeFeatureExtractor()


class Qwen2AudioTests(unittest.TestCase):
    def test_roster_and_settings_defaults(self) -> None:
        plugin = Qwen2AudioPlugin()
        self.assertEqual(plugin.name, "qwen2audio")
        self.assertEqual(plugin.display_name, "Qwen2-Audio")
        self.assertEqual(plugin.embedding_dim, 1280)
        self.assertFalse(plugin.provides_text)
        self.assertEqual(plugin.preferred_sample_rate, 16000)
        self.assertEqual(plugin.default_model_id, QWEN2AUDIO_MODEL_ID)
        # the tower demands exactly 30 s per window
        self.assertEqual(plugin.window_sec, QWEN_WINDOW_SEC)
        self.assertEqual(QWEN_WINDOW_SEC, 30.0)

    def test_apply_config_batch_and_unload_on_model_change(self) -> None:
        plugin = Qwen2AudioPlugin()
        plugin.apply_config(type("Cfg", (), {
            "qwen2audio_model_id": QWEN2AUDIO_MODEL_ID,
            "qwen2audio_batch_size": 3}))
        self.assertEqual(plugin.batch_size, 3)
        _loaded(plugin, _FakeTower())
        plugin._loaded_model_id = plugin.model_id
        plugin.apply_config(type("Cfg", (), {
            "qwen2audio_model_id": "other/model",
            "qwen2audio_batch_size": 1}))
        self.assertFalse(plugin._loaded)      # model change dropped weights
        self.assertEqual(plugin.model_id, "other/model")

    def test_embed_windows_pools_and_normalizes(self) -> None:
        plugin = Qwen2AudioPlugin()
        _loaded(plugin, _FakeTower())
        plugin._processor = _FakeProcessor()
        sr = 16000
        # 70 s chunk -> 3 exact 30 s windows (last: 10 s, padded)
        chunk = np.concatenate([
            np.full(30 * sr, 0.5, dtype=np.float32),
            np.full(30 * sr, -0.5, dtype=np.float32),
            np.full(10 * sr, 0.25, dtype=np.float32)])
        vecs = plugin._embed([chunk], sr)
        self.assertEqual(vecs[0].shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(vecs[0])), 1.0,
                               places=5)
        # resampling from 44.1 kHz also works
        chunk_44 = np.concatenate([
            np.full(int(30 * 44100), 0.5, dtype=np.float32),
            np.full(int(5 * 44100), 0.25, dtype=np.float32)])
        vecs44 = plugin._embed([chunk_44], 44100)
        self.assertEqual(vecs44[0].shape, (3,))

    def test_empty_chunk_yields_zero_vector(self) -> None:
        plugin = Qwen2AudioPlugin()
        _loaded(plugin, _FakeTower())
        plugin._processor = _FakeProcessor()
        vecs = plugin._embed([np.zeros(0, dtype=np.float32)], 16000)
        self.assertEqual(vecs[0].shape, (1280,))
        self.assertEqual(float(np.abs(vecs[0]).sum()), 0.0)


# --------------------------------------------------------------------------
# roster v4 wiring
# --------------------------------------------------------------------------
class RosterV4Tests(unittest.TestCase):
    def test_registry_canonical_order(self) -> None:
        from app.models.registry import list_plugins

        self.assertEqual(
            [p.name for p in list_plugins()],
            ["clap", "mert", "mert330", "m2dclap", "muq", "muqlan",
             "lpmc", "qwen2audio", "openl3", "fft"])

    def test_config_migration_appends_v4_models_once(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        from app.config import AppConfig

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            # a v3-era config: both new models absent
            raw = {"models": ["clap", "fft"], "models_version": 3}
            path.write_text(json.dumps(raw), encoding="utf-8")
            with mock.patch("app.config.CONFIG_PATH", path):
                cfg = AppConfig.load()
            self.assertEqual(cfg.models_version, 4)
            self.assertIn("lpmc", cfg.models)
            self.assertIn("qwen2audio", cfg.models)
            self.assertIn("fft", cfg.models)
            # loading again must NOT re-append (save() persists v4)
            with mock.patch("app.config.CONFIG_PATH", path):
                cfg2 = AppConfig.load()
            self.assertEqual(cfg2.models, cfg.models)


if __name__ == "__main__":
    unittest.main()
