"""Tests for the M2D-CLAP / MuQ / MuQ-MuLan plugins.

Headless: no weight downloads, no torch model loads — heavy work is faked
by injecting stub models and stub runtimes.  Network-touching bootstrap
helpers are exercised with monkeypatched downloads.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.config import AppConfig
from app.models.m2dclap_model import M2dClapPlugin
from app.models.muq_model import (MuQMuLanPlugin, MuQPlugin, pad_to_max,
                                  _patch_wav2vec2_easydict_configs,
                                  _wrap_conformer_hidden_states)


def _loaded_with(plugin, model) -> None:
    """Force a plugin into the loaded state with a stub model object."""
    plugin._model = model
    plugin._device = "cpu"
    plugin._loaded = True


def _l2(vec: np.ndarray) -> np.ndarray:
    return vec / max(np.linalg.norm(vec), 1e-12)


class _FakeMuQOutput:
    def __init__(self, tensor):
        self.last_hidden_state = tensor


class _FakeMuQ:
    """Stands in for muq.MuQ: per-window hidden state = (t0, t1, d0, d1)."""

    dim = 4

    def __call__(self, wavs, output_hidden_states=True):
        return self.forward(wavs, output_hidden_states)

    def forward(self, wavs, output_hidden_states=True):
        import torch

        t = wavs.shape[1]
        # direction (not only magnitude) varies with the window content
        head = wavs[:, :min(512, t)].mean(dim=1, keepdim=True)
        tail = wavs[:, -min(512, t):].mean(dim=1, keepdim=True)
        col = torch.cat([head, tail, head - tail, head * tail], dim=1)
        hidden = col.unsqueeze(1)               # (b, 1, dim)
        return _FakeMuQOutput(hidden)


class _FakeMuLan:
    """Stands in for muq.MuQMuLan: audio → 512-d (deterministic), text → id."""

    dim = 512

    def __call__(self, wavs=None, texts=None):
        return self.forward(wavs=wavs, texts=texts)

    def forward(self, wavs=None, texts=None):
        import torch

        if wavs is not None:
            head = wavs[:, :512].mean(dim=1, keepdim=True)
            return head.repeat(1, self.dim)
        rows = []
        for i, text in enumerate(texts):
            row = torch.zeros(self.dim)
            row[i % self.dim] = 1.0 + float(len(text))
            rows.append(row)
        return torch.stack(rows)


class _FakeM2D:
    """Stands in for PortableM2D (CLAP embedding + text paths)."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"sample_rate": 16000})()

    def to(self, device):
        return self

    def eval(self):
        return self

    def encode_clap_audio(self, wavs):
        import torch

        head = wavs[:, :160].mean(dim=1, keepdim=True)
        return head.repeat(1, 4)

    def encode_clap_text(self, texts):
        import torch

        rows = []
        for i, text in enumerate(texts):
            row = torch.zeros(4)
            row[i % 4] = 1.0 + float(len(text))
            rows.append(row)
        return torch.stack(rows)


class PadTests(unittest.TestCase):
    def test_pad_to_max_pads_the_tail(self) -> None:
        batch = [np.ones(4, dtype=np.float32),
                 np.ones(2, dtype=np.float32) * 2]
        padded = pad_to_max(batch)
        self.assertEqual(padded[0].shape, (4,))
        self.assertEqual(padded[1].shape, (4,))
        np.testing.assert_array_equal(padded[1], [2, 2, 0, 0])

    def test_pad_to_max_single_and_equal_batches_untouched(self) -> None:
        one = [np.ones(3, dtype=np.float32)]
        self.assertIs(pad_to_max(one)[0], one[0])
        equal = [np.ones(3), np.zeros(3)]
        out = pad_to_max(equal)
        self.assertIs(out[0], equal[0])
        self.assertIs(out[1], equal[1])


class MuQPluginTests(unittest.TestCase):
    def test_requirements_cover_muqs_broken_metadata(self) -> None:
        """The 'muq' PyPI package ships no Requires-Dist, so its import-time
        dependencies (easydict/torchaudio/x_clip — pulled by 'from muq import
        …') must be gated here; otherwise is_available lies and analysis
        crashes mid-run with 'No module named easydict'."""
        import importlib.util

        self.assertEqual(
            MuQPlugin.requirements,
            ("torch", "muq", "easydict", "torchaudio", "x_clip"))
        self.assertEqual(MuQMuLanPlugin.requirements, MuQPlugin.requirements)
        for module in MuQPlugin.requirements:
            self.assertIsNotNone(importlib.util.find_spec(module),
                                 f"{module} must be importable here")

    def test_is_available_needs_the_muq_package(self) -> None:
        plugin = MuQPlugin()
        with mock.patch("app.models.base.module_available",
                        side_effect=lambda m: m in MuQPlugin.requirements):
            self.assertTrue(plugin.is_available())
            self.assertIsNone(plugin.availability_error())
        with mock.patch("app.models.base.module_available",
                        side_effect=lambda m: m != "easydict"):
            self.assertFalse(plugin.is_available())
            self.assertIn("easydict", plugin.availability_error())

    def test_apply_config_windowing_and_batch(self) -> None:
        plugin = MuQPlugin()
        cfg = AppConfig()
        cfg.muq_window_sec = 12.0
        cfg.muq_window_overlap_sec = 30.0        # clamped below window
        cfg.muq_batch_size = 0                    # clamped to 1
        plugin.apply_config(cfg)
        self.assertEqual(plugin.window_sec, 12.0)
        self.assertEqual(plugin.window_overlap_sec, 11.5)
        self.assertEqual(plugin.batch_size, 1)
        plugin.apply_config(object())             # partial fake: no fields
        self.assertEqual(plugin.window_sec, 12.0)  # unchanged

    def test_model_id_change_unloads(self) -> None:
        plugin = MuQPlugin()
        _loaded_with(plugin, _FakeMuQ())
        plugin._loaded_model_id = plugin.model_id
        cfg = AppConfig()
        cfg.muq_model_id = "OpenMuQ/MuQ-large-music4all-iter"
        plugin.apply_config(cfg)
        self.assertFalse(plugin._loaded)

    def test_embed_windows_and_pools_long_chunks(self) -> None:
        plugin = MuQPlugin()
        _loaded_with(plugin, _FakeMuQ())
        sr = 24000
        long_chunk = np.zeros(int(sr * 25), dtype=np.float32)  # 25 s → 3 windows
        long_chunk[10] = 1.0        # spike inside the fake's first-512 window
        vecs = plugin.embed([long_chunk], sr)
        self.assertEqual(len(vecs), 1)
        vec = vecs[0]
        self.assertEqual(vec.shape, (4,))
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)
        short = plugin.embed([np.zeros(sr, dtype=np.float32)], sr)[0]
        self.assertEqual(short.shape, (4,))
        # windowed mean differs from a single-window embedding
        self.assertFalse(np.allclose(vec, short))

    def test_embed_batch_of_equal_length_chunks(self) -> None:
        plugin = MuQPlugin()
        plugin.batch_size = 2
        _loaded_with(plugin, _FakeMuQ())
        sr = 8000   # gets resampled to 24 kHz
        chunks = [np.full(sr, 0.25, dtype=np.float32),
                  np.full(sr, 0.75, dtype=np.float32)]
        vecs = plugin.embed(chunks, sr)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(vecs[0].shape, (4,))
        self.assertFalse(np.allclose(vecs[0], vecs[1]))


class MuQMuLanPluginTests(unittest.TestCase):
    def test_apply_config_tags_and_reload_semantics(self) -> None:
        plugin = MuQMuLanPlugin()
        _loaded_with(plugin, _FakeMuLan())
        plugin._loaded_model_id = plugin.model_id
        cfg = AppConfig()
        cfg.muqlan_tags = ["rock", "jazz", "", "rock"]
        plugin.apply_config(cfg)
        self.assertEqual(plugin.tag_candidates, ("rock", "jazz"))
        self.assertFalse(plugin._loaded)   # tag list change → reload

    def test_embed_pools_windows_and_normalizes(self) -> None:
        plugin = MuQMuLanPlugin()
        _loaded_with(plugin, _FakeMuLan())
        sr = 24000
        long_chunk = np.zeros(int(sr * 22), dtype=np.float32)
        long_chunk[500] = 1.0
        vec = plugin.embed([long_chunk], sr)[0]
        self.assertEqual(vec.shape, (512,))
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_describe_ranks_the_matching_tag_first(self) -> None:
        from app.models.muq_model import _l2_normalize_rows

        plugin = MuQMuLanPlugin()
        _loaded_with(plugin, _FakeMuLan())
        # tag text features for the current candidate list (fake text side)
        plugin._tag_texts = tuple(plugin.tag_candidates)
        plugin._text_features = _l2_normalize_rows(
            plugin._encode_texts(list(plugin.tag_candidates)))
        chunk = np.zeros(24000, dtype=np.float32)
        results = plugin.describe([chunk], 24000, top_k=2)
        self.assertEqual(len(results), 1)
        top2 = results[0]
        self.assertEqual(len(top2), 2)
        self.assertTrue(all(tag in plugin.tag_candidates for tag, _ in top2))
        self.assertGreaterEqual(top2[0][1], top2[1][1])


class TransformerV5ShimTests(unittest.TestCase):
    """_wrap_conformer_hidden_states restores transformers v4 semantics on
    a v5 encoder (which stopped returning per-layer hidden_states)."""

    def _fake_conformer(self):
        import torch
        from torch import nn
        from transformers.modeling_outputs import BaseModelOutput

        class Layer(nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, x):
                return x + self.offset

        class Conformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer(10.0), Layer(20.0)])
                self.norm_offset = 1000.0

            def forward(self, hidden_states, attention_mask=None, **kwargs):
                for layer in self.layers:
                    hidden_states = layer(hidden_states)
                return BaseModelOutput(
                    last_hidden_state=hidden_states + self.norm_offset)

        return Conformer(), torch

    def test_wrapped_forward_returns_v4_style_hidden_states(self) -> None:
        conformer, torch = self._fake_conformer()
        _wrap_conformer_hidden_states(conformer)
        self.assertTrue(conformer._holosmart_v5_hidden_states_patch)
        out = conformer(torch.zeros(1, 3), output_hidden_states=True)
        # tuple = raw layer outputs (chained: 0->10->30) + final normed state
        self.assertEqual(len(out.hidden_states), 3)
        np.testing.assert_allclose(
            out.hidden_states[0].numpy(), 10.0 * np.ones((1, 3)))
        np.testing.assert_allclose(
            out.hidden_states[1].numpy(), 30.0 * np.ones((1, 3)))
        np.testing.assert_allclose(
            out.hidden_states[2].numpy(), 1030.0 * np.ones((1, 3)))
        # hidden_states[-1] IS the last_hidden_state (what muq indexes)
        self.assertIs(out.hidden_states[-1], out.last_hidden_state)

    def test_false_leaves_v5_behavior_untouched(self) -> None:
        conformer, torch = self._fake_conformer()
        _wrap_conformer_hidden_states(conformer)
        out = conformer(torch.zeros(1, 2), output_hidden_states=False)
        self.assertNotIn("hidden_states", out)
        np.testing.assert_allclose(
            out.last_hidden_state.numpy(), 1030.0 * np.ones((1, 2)))

    def test_wrap_is_idempotent(self) -> None:
        conformer, _ = self._fake_conformer()
        _wrap_conformer_hidden_states(conformer)
        wrapped = conformer.forward
        _wrap_conformer_hidden_states(conformer)
        self.assertIs(conformer.forward, wrapped)

    def test_config_stamping_applies_to_module_tree(self) -> None:
        import torch
        from types import SimpleNamespace
        from torch import nn

        class Holder(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Identity()
                self.inner.config = SimpleNamespace()   # attribute config

        holder = Holder()
        _patch_wav2vec2_easydict_configs(holder)
        self.assertEqual(holder.inner.config._attn_implementation, "eager")


class M2dClapBootstrapTests(unittest.TestCase):
    """The download-once bootstrap: runtime file + weights zip → import."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)

    def _plugin_with_stub_runtime(self) -> M2dClapPlugin:
        base = self.data_dir / "models" / "m2d"
        base.mkdir(parents=True, exist_ok=True)
        # official runtime replaced by a tiny stub with the same import API
        (base / "portable_m2d.py").write_text(textwrap.dedent('''
            class PortableM2D:
                def __init__(self, weight_file=None, flat_features=True):
                    self.cfg = type("Cfg", (), {"sample_rate": 16000})()

                def to(self, device):
                    return self

                def eval(self):
                    return self

                def encode_clap_audio(self, wavs):
                    import torch
                    head = wavs[:, :160].mean(dim=1, keepdim=True)
                    return head.repeat(1, 4)

                def encode_clap_text(self, texts):
                    import torch
                    rows = []
                    for i, text in enumerate(texts):
                        row = torch.zeros(4)
                        row[i % 4] = 1.0 + float(len(text))
                        rows.append(row)
                    return torch.stack(rows)
        '''), encoding="utf-8")
        ckpt_dir = base / "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "checkpoint-30.pth").write_bytes(b"stub")
        plugin = M2dClapPlugin()
        plugin.apply_config(AppConfig())   # adopts the default tag list
        patcher = mock.patch.object(plugin, "is_available", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)      # deps (timm/…) are not installed
        return plugin

    def _patch_data_dir(self, plugin: M2dClapPlugin) -> None:
        patcher = mock.patch("app.config.DATA_DIR", self.data_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_apply_config_tag_semantics(self) -> None:
        plugin = M2dClapPlugin()
        cfg = AppConfig()
        cfg.m2dclap_tags = ["rock", "", "ROCK", "jazz"]
        cfg.m2dclap_batch_size = 0
        plugin.apply_config(cfg)
        self.assertEqual(plugin.tag_candidates, ("rock", "jazz"))
        self.assertEqual(plugin.batch_size, 1)

    def test_load_uses_downloaded_runtime_and_weights(self) -> None:
        plugin = self._plugin_with_stub_runtime()
        self._patch_data_dir(plugin)
        downloads = []
        with mock.patch.object(plugin, "_download",
                               side_effect=lambda url, dest:
                               downloads.append(url)):
            plugin.ensure_loaded()
        self.assertEqual(downloads, [])   # both files already present
        self.assertTrue(plugin.is_loaded)
        sr = 16000
        chunk = np.zeros(sr, dtype=np.float32)
        chunk[100] = 1.0
        vec = plugin.embed([chunk], sr)[0]
        self.assertEqual(vec.shape, (4,))
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)
        results = plugin.describe([chunk], sr, top_k=3)
        self.assertEqual(len(results[0]), 3)
        self.assertGreaterEqual(results[0][0][1], results[0][-1][1])

    def test_missing_runtime_and_weights_are_downloaded_then_imported(self) -> None:
        plugin = self._plugin_with_stub_runtime()
        self._patch_data_dir(plugin)
        # simulate a fresh machine: remove the pre-seeded stub files
        plugin._runtime_path().unlink()
        plugin._checkpoint_path().unlink()

        def fake_download(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            if url.endswith("portable_m2d.py"):
                dest.write_text(textwrap.dedent('''
                    class PortableM2D:
                        def __init__(self, weight_file=None, flat_features=True):
                            self.cfg = type("Cfg", (), {"sample_rate": 16000})()
                        def to(self, device):
                            return self
                        def eval(self):
                            return self
                        def encode_clap_audio(self, wavs):
                            import torch
                            return wavs.mean(dim=1, keepdim=True).repeat(1, 4)
                        def encode_clap_text(self, texts):
                            import torch
                            rows = []
                            for i, text in enumerate(texts):
                                row = torch.zeros(4)
                                row[i % 4] = 1.0 + float(len(text))
                                rows.append(row)
                            return torch.stack(rows)
                '''), encoding="utf-8")
            else:
                # the release zip: contains the checkpoint file
                import zipfile
                with zipfile.ZipFile(dest, "w") as zf:
                    zf.writestr(
                        "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/"
                        "checkpoint-30.pth", b"stub")

        with mock.patch.object(plugin, "_download",
                               side_effect=fake_download):
            plugin.ensure_loaded()
        self.assertTrue(plugin._runtime_path().exists())
        self.assertTrue(plugin._checkpoint_path().exists())
        vec = plugin.embed([np.zeros(16000, dtype=np.float32)], 16000)[0]
        self.assertEqual(vec.shape, (4,))

    def test_tag_template_and_reload_on_tag_change(self) -> None:
        plugin = self._plugin_with_stub_runtime()
        self._patch_data_dir(plugin)
        with mock.patch.object(plugin, "_download", side_effect=lambda u, d: None):
            plugin.ensure_loaded()
        cfg = AppConfig()
        cfg.m2dclap_tags = ["metal", "jazz"]
        plugin.apply_config(cfg)
        self.assertFalse(plugin._loaded)   # tag list change → reload
