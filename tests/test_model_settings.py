"""Tests for model settings: config fields, plugin apply_config, dialog, pipeline.

Headless: no torch, no network; Qt only offscreen where a widget is needed.
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

from app.config import AppConfig  # noqa: E402
from app.models.clap_model import CLAP_MODEL_ID, ClapPlugin  # noqa: E402
from app.models import clap_model  # noqa: E402
from app.models.mert_model import MertPlugin  # noqa: E402
from app.models.registry import get_plugin  # noqa: E402

NEW_FIELDS = (
    "clap_model_id", "clap_tag_top_k", "clap_batch_size",
    "mert_model_id", "mert_window_sec", "mert_window_overlap_sec",
    "mert_batch_size",
    "mert330_model_id", "mert330_window_sec", "mert330_window_overlap_sec",
    "mert330_batch_size",
    "fft_window_sec",
    "m2dclap_tag_top_k", "m2dclap_batch_size",
    "muq_model_id", "muq_window_sec", "muq_window_overlap_sec",
    "muq_batch_size",
    "muqlan_model_id", "muqlan_window_sec", "muqlan_window_overlap_sec",
    "muqlan_batch_size", "muqlan_tag_top_k",
)

DEFAULTS = {
    "clap_model_id": "laion/clap-htsat-unfused",
    "clap_tag_top_k": 5,
    "clap_batch_size": 8,
    "mert_model_id": "m-a-p/MERT-v1-95M",
    "mert_window_sec": 10.0,
    "mert_window_overlap_sec": 1.0,
    "mert_batch_size": 8,
    "mert330_model_id": "m-a-p/MERT-v1-330M",
    "mert330_window_sec": 10.0,
    "mert330_window_overlap_sec": 1.0,
    "mert330_batch_size": 4,
    "fft_window_sec": 10.0,
    "m2dclap_tag_top_k": 5,
    "m2dclap_batch_size": 4,
    "muq_model_id": "OpenMuQ/MuQ-large-msd-iter",
    "muq_window_sec": 10.0,
    "muq_window_overlap_sec": 1.0,
    "muq_batch_size": 4,
    "muqlan_model_id": "OpenMuQ/MuQ-MuLan-large",
    "muqlan_window_sec": 10.0,
    "muqlan_window_overlap_sec": 1.0,
    "muqlan_batch_size": 4,
    "muqlan_tag_top_k": 5,
}


class ConfigRoundtripTests(unittest.TestCase):
    def test_new_fields_have_backward_compatible_defaults(self) -> None:
        cfg = AppConfig()
        for field, expected in DEFAULTS.items():
            self.assertEqual(getattr(cfg, field), expected, field)

    def test_config_roundtrip_persists_new_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            cfg = AppConfig()
            cfg.clap_model_id = "laion/clap-htsat-fused"
            cfg.clap_tag_top_k = 7
            cfg.clap_batch_size = 16
            cfg.mert_model_id = "m-a-p/MERT-v1-95M-fake"
            cfg.mert_window_sec = 12.5
            cfg.mert_window_overlap_sec = 2.0
            cfg.mert_batch_size = 4
            cfg.mert330_model_id = "m-a-p/MERT-v1-330M-fake"
            cfg.mert330_window_sec = 11.0
            cfg.mert330_window_overlap_sec = 1.5
            cfg.mert330_batch_size = 2
            # v3 roster fields (M2D-CLAP / MuQ / MuQ-MuLan)
            cfg.muq_model_id = "OpenMuQ/MuQ-large-music4all-iter"
            cfg.muq_window_sec = 11.5
            cfg.muq_window_overlap_sec = 1.5
            cfg.muq_batch_size = 2
            cfg.muqlan_window_sec = 9.0
            cfg.muqlan_batch_size = 3
            cfg.muqlan_tag_top_k = 9
            cfg.m2dclap_batch_size = 6
            cfg.m2dclap_tag_top_k = 8
            with mock.patch("app.config.DATA_DIR", data_dir), \
                    mock.patch("app.config.CONFIG_PATH", data_dir / "config.json"):
                cfg.save()
                loaded = AppConfig.load()
        self.assertEqual(loaded.clap_model_id, "laion/clap-htsat-fused")
        self.assertEqual(loaded.clap_tag_top_k, 7)
        self.assertEqual(loaded.clap_batch_size, 16)
        self.assertEqual(loaded.mert_model_id, "m-a-p/MERT-v1-95M-fake")
        self.assertEqual(loaded.mert_window_sec, 12.5)
        self.assertEqual(loaded.mert_window_overlap_sec, 2.0)
        self.assertEqual(loaded.mert_batch_size, 4)
        self.assertEqual(loaded.mert330_model_id, "m-a-p/MERT-v1-330M-fake")
        self.assertEqual(loaded.mert330_window_sec, 11.0)
        self.assertEqual(loaded.mert330_window_overlap_sec, 1.5)
        self.assertEqual(loaded.mert330_batch_size, 2)
        self.assertEqual(loaded.muq_model_id, "OpenMuQ/MuQ-large-music4all-iter")
        self.assertEqual(loaded.muq_window_sec, 11.5)
        self.assertEqual(loaded.muq_window_overlap_sec, 1.5)
        self.assertEqual(loaded.muq_batch_size, 2)
        self.assertEqual(loaded.muqlan_window_sec, 9.0)
        self.assertEqual(loaded.muqlan_batch_size, 3)
        self.assertEqual(loaded.muqlan_tag_top_k, 9)
        self.assertEqual(loaded.m2dclap_batch_size, 6)
        self.assertEqual(loaded.m2dclap_tag_top_k, 8)

    def test_v3_models_migration_appends_once(self) -> None:
        """A config written before the M2D-CLAP/MuQ roster gains the new
        models exactly once; a deliberate opt-out survives reloads."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            with mock.patch("app.config.DATA_DIR", data_dir), \
                    mock.patch("app.config.CONFIG_PATH",
                               data_dir / "config.json"):
                cfg = AppConfig()
                cfg.models = ["clap", "mert"]
                cfg.models_version = 2          # pre-v3 build
                cfg.save()
                loaded = AppConfig.load()
                for name in ("m2dclap", "muq", "muqlan"):
                    self.assertIn(name, loaded.models)
                self.assertEqual(loaded.models_version,
                                 AppConfig.models_version)   # bumped to current
                for name in ("lpmc", "qwen2audio"):
                    self.assertIn(name, loaded.models)   # v4 rides along
                loaded.models = ["clap"]        # deliberate opt-out
                loaded.save()
                reloaded = AppConfig.load()
        for name in ("m2dclap", "muq", "muqlan"):
            self.assertNotIn(name, reloaded.models)

    def test_clap_tags_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            cfg = AppConfig()
            self.assertIsNone(cfg.clap_tags)          # default = built-in list
            cfg.clap_tags = ["rock", "ambient", "field recording"]
            with mock.patch("app.config.DATA_DIR", data_dir), \
                    mock.patch("app.config.CONFIG_PATH", data_dir / "config.json"):
                cfg.save()
                loaded = AppConfig.load()
        self.assertEqual(loaded.clap_tags,
                         ["rock", "ambient", "field recording"])

    def test_load_ignores_unknown_keys_still_fine(self) -> None:
        # Old config files (without the new keys) must load to defaults.
        import json

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            data_dir.mkdir(exist_ok=True)
            payload = {"chunk_seconds": 30.0, "playlist_length": 9}
            (data_dir / "config.json").write_text(json.dumps(payload),
                                                  encoding="utf-8")
            with mock.patch("app.config.DATA_DIR", data_dir), \
                    mock.patch("app.config.CONFIG_PATH", data_dir / "config.json"):
                loaded = AppConfig.load()
        self.assertEqual(loaded.chunk_seconds, 30.0)
        self.assertEqual(loaded.playlist_length, 9)
        for field, expected in DEFAULTS.items():
            self.assertEqual(getattr(loaded, field), expected, field)


class _ApplyConfigMixin:
    """Shared assertions for ClapPlugin.apply_config / MertPlugin.apply_config."""

    #: plugin attr name -> AppConfig field name
    config_field_map: dict = {}

    def _make_config(self, **over):
        cfg = AppConfig()
        for attr, value in over.items():
            setattr(cfg, self.config_field_map[attr], value)
        return cfg

    def _set_loaded(self, plugin, model_id):
        plugin._loaded = True
        plugin._loaded_model_id = model_id

    def test_default_instance_attrs_match_module_defaults(self):
        plugin = self.plugin_factory()
        for attr, expected in self.expected_defaults.items():
            self.assertEqual(getattr(plugin, attr), expected)

    def test_apply_config_sets_instance_attrs(self):
        plugin = self.plugin_factory()
        plugin.apply_config(self._make_config(**self.changed_values))
        for attr, expected in self.changed_values.items():
            self.assertEqual(getattr(plugin, attr), expected, attr)

    def test_apply_config_tolerates_partial_fakes(self):
        plugin = self.plugin_factory()
        before = {a: getattr(plugin, a) for a in self.expected_defaults}
        plugin.apply_config(object())  # no attributes at all
        for attr, expected in before.items():
            self.assertEqual(getattr(plugin, attr), expected, attr)

    def test_model_id_change_unloads_loaded_plugin(self):
        plugin = self.plugin_factory()
        old_id = self.expected_defaults["model_id"]
        self._set_loaded(plugin, old_id)
        plugin.apply_config(self._make_config(**{
            "model_id": "some/other-model"}))
        self.assertFalse(plugin.is_loaded)
        self.assertIsNone(plugin._loaded_model_id)
        self.assertEqual(plugin.model_id, "some/other-model")

    def test_unchanged_model_id_does_not_unload(self):
        plugin = self.plugin_factory()
        old_id = self.expected_defaults["model_id"]
        self._set_loaded(plugin, old_id)
        plugin.apply_config(self._make_config())  # defaults -> same model id
        self.assertTrue(plugin.is_loaded)
        self.assertEqual(plugin._loaded_model_id, old_id)

    def test_forced_loaded_without_loaded_model_id_unchanged_keeps_loaded(self):
        plugin = self.plugin_factory()
        plugin._loaded = True  # _loaded_model_id left as None
        plugin.apply_config(self._make_config())
        self.assertTrue(plugin.is_loaded)


class ClapApplyConfigTests(_ApplyConfigMixin, unittest.TestCase):
    def setUp(self):
        self.plugin_factory = ClapPlugin
        self.config_field_map = {
            "model_id": "clap_model_id",
            "tag_top_k": "clap_tag_top_k",
            "batch_size": "clap_batch_size",
        }
        self.expected_defaults = {
            "model_id": CLAP_MODEL_ID,
            "tag_top_k": 5,
            "batch_size": 8,
        }
        self.changed_values = {
            "model_id": "laion/clap-htsat-fused",
            "tag_top_k": 9,
            "batch_size": 3,
        }

    def test_embed_uses_instance_batch_size(self):
        # _embed batches with self.batch_size: verify via a stubbed processor.
        plugin = ClapPlugin()
        seen_batches: list[int] = []

        class FakeProcessor:
            def __call__(self, audio, sampling_rate, return_tensors, padding):
                seen_batches.append(len(audio))
                return {"input_features": None, "is_longer": None}

        class FakeModel:
            def get_audio_features(self, **kwargs):
                return None

        plugin._model = FakeModel()
        plugin._processor = FakeProcessor()
        plugin._device = "cpu"
        plugin._loaded = True
        plugin.batch_size = 2
        chunks = [np.zeros(4, dtype=np.float32) for _ in range(5)]
        with mock.patch("app.models.clap_model._extract_embeds",
                        return_value=np.ones((1, 4), dtype=np.float32)):
            plugin._embed(chunks, 48000)
        self.assertEqual(seen_batches, [2, 2, 1])

    def test_apply_config_adopts_custom_tag_list(self):
        plugin = ClapPlugin()
        self.assertEqual(plugin.tag_candidates,
                         clap_model.CANDIDATE_TAGS)      # default
        cfg = AppConfig()
        cfg.clap_tags = ["field recording", "rain", "storm"]
        plugin.apply_config(cfg)
        self.assertEqual(plugin.tag_candidates,
                         ("field recording", "rain", "storm"))
        # None / empty → back to the built-in candidate list.
        cfg.clap_tags = None
        plugin.apply_config(cfg)
        self.assertEqual(plugin.tag_candidates, clap_model.CANDIDATE_TAGS)
        cfg.clap_tags = []
        plugin.apply_config(cfg)
        self.assertEqual(plugin.tag_candidates, clap_model.CANDIDATE_TAGS)

    def test_tag_list_change_unloads_loaded_weights(self):
        # Text features are baked for one tag list: a change must drop the
        # cached weights so the next analysis recomputes them.
        plugin = ClapPlugin()
        plugin._loaded = True
        plugin._loaded_model_id = plugin.model_id
        cfg = AppConfig()
        cfg.clap_tags = list(clap_model.CANDIDATE_TAGS[:3])
        plugin.apply_config(cfg)
        self.assertFalse(plugin._loaded)
        # An unchanged list must NOT unload.
        plugin._loaded = True
        cfg = AppConfig()
        cfg.clap_tags = list(clap_model.CANDIDATE_TAGS[:3])
        plugin.apply_config(cfg)
        self.assertTrue(plugin._loaded)
        # And a model-id change still unloads (pre-existing behaviour).
        plugin._loaded = True
        plugin._loaded_model_id = plugin.model_id
        cfg = AppConfig()
        cfg.clap_model_id = "laion/clap-htsat-fused"
        plugin.apply_config(cfg)
        self.assertFalse(plugin._loaded)


class MertApplyConfigTests(_ApplyConfigMixin, unittest.TestCase):
    def setUp(self):
        self.plugin_factory = MertPlugin
        self.config_field_map = {
            "model_id": "mert_model_id",
            "window_sec": "mert_window_sec",
            "window_overlap_sec": "mert_window_overlap_sec",
            "batch_size": "mert_batch_size",
        }
        self.expected_defaults = {
            "model_id": "m-a-p/MERT-v1-95M",
            "window_sec": 10.0,
            "window_overlap_sec": 1.0,
            "batch_size": 8,
        }
        self.changed_values = {
            "model_id": "m-a-p/MERT-v1-95M-alt",
            "window_sec": 8.0,
            "window_overlap_sec": 2.0,
            "batch_size": 2,
        }

    def test_overlap_clamped_below_window(self):
        plugin = MertPlugin()
        plugin.apply_config(self._make_config(window_sec=10.0,
                                              window_overlap_sec=10.0))
        self.assertEqual(plugin.window_sec, 10.0)
        self.assertEqual(plugin.window_overlap_sec, 9.5)  # window - 0.5

        plugin2 = MertPlugin()
        plugin2.apply_config(self._make_config(window_sec=2.0,
                                               window_overlap_sec=5.0))
        self.assertEqual(plugin2.window_sec, 2.0)
        self.assertEqual(plugin2.window_overlap_sec, 1.5)

    def test_window_and_overlap_stay_positive(self):
        plugin = MertPlugin()
        plugin.apply_config(self._make_config(window_sec=0.0,
                                              window_overlap_sec=-3.0))
        self.assertGreater(plugin.window_sec, 0.0)
        self.assertGreaterEqual(plugin.window_overlap_sec, 0.0)

    def test_unload_resets_loaded_model_id(self):
        plugin = MertPlugin()
        self._set_loaded(plugin, "m-a-p/MERT-v1-95M")
        plugin.unload()
        self.assertFalse(plugin.is_loaded)
        self.assertIsNone(plugin._loaded_model_id)


class Mert330ApplyConfigTests(_ApplyConfigMixin, unittest.TestCase):
    def setUp(self):
        from app.models.mert_model import Mert330Plugin
        self.plugin_factory = Mert330Plugin
        self.config_field_map = {
            "model_id": "mert330_model_id",
            "window_sec": "mert330_window_sec",
            "window_overlap_sec": "mert330_window_overlap_sec",
            "batch_size": "mert330_batch_size",
        }
        self.expected_defaults = {
            "model_id": "m-a-p/MERT-v1-330M",
            "window_sec": 10.0,
            "window_overlap_sec": 1.0,
            "batch_size": 4,
        }
        self.changed_values = {
            "model_id": "m-a-p/MERT-v1-330M-alt",
            "window_sec": 9.0,
            "window_overlap_sec": 1.5,
            "batch_size": 2,
        }

    def test_settings_prefixes_are_isolated(self):
        # MERT and MERT-330M read only their own config fields.
        from app.models.mert_model import Mert330Plugin

        cfg = AppConfig()
        cfg.mert_model_id = "m-a-p/MERT-v1-95M-a"
        cfg.mert330_model_id = "m-a-p/MERT-v1-330M-a"
        mert, mert330 = MertPlugin(), Mert330Plugin()
        mert.apply_config(cfg)
        mert330.apply_config(cfg)
        self.assertEqual(mert.model_id, "m-a-p/MERT-v1-95M-a")
        self.assertEqual(mert330.model_id, "m-a-p/MERT-v1-330M-a")
        cfg.mert330_batch_size = 7
        mert.apply_config(cfg)
        self.assertEqual(mert.batch_size, 8)          # unaffected
        mert330.apply_config(cfg)
        self.assertEqual(mert330.batch_size, 7)


class RegistrySingletonTests(unittest.TestCase):
    def test_get_plugin_singletons_unchanged(self) -> None:
        self.assertIs(get_plugin("clap"), get_plugin("clap"))
        self.assertIs(get_plugin("mert"), get_plugin("mert"))

    def test_singletons_expose_apply_config(self) -> None:
        self.assertTrue(callable(getattr(get_plugin("clap"), "apply_config", None)))
        self.assertTrue(callable(getattr(get_plugin("mert"), "apply_config", None)))


class SettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def test_dialog_applies_clap_and_mert_settings(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        dialog = SettingsDialog(config)
        dialog._clap_model_combo.setCurrentText("laion/clap-htsat-fused")
        dialog._clap_top_k.setValue(12)
        dialog._clap_batch.setValue(16)
        dialog._mert_model_combo.setCurrentText("m-a-p/MERT-v1-95M")
        dialog._mert_window.setValue(8.0)
        dialog._mert_overlap.setValue(2.0)
        dialog._mert_batch.setValue(4)
        dialog.apply()
        self.assertEqual(config.clap_model_id, "laion/clap-htsat-fused")
        self.assertEqual(config.clap_tag_top_k, 12)
        self.assertEqual(config.clap_batch_size, 16)
        self.assertEqual(config.mert_model_id, "m-a-p/MERT-v1-95M")
        self.assertEqual(config.mert_window_sec, 8.0)
        self.assertEqual(config.mert_window_overlap_sec, 2.0)
        self.assertEqual(config.mert_batch_size, 4)

    def test_dialog_applies_mert330_settings(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        dialog = SettingsDialog(config)
        dialog._mert330_model_combo.setCurrentText("m-a-p/MERT-v1-330M")
        dialog._mert330_window.setValue(9.0)
        dialog._mert330_overlap.setValue(1.5)
        dialog._mert330_batch.setValue(2)
        dialog.apply()
        self.assertEqual(config.mert330_model_id, "m-a-p/MERT-v1-330M")
        self.assertEqual(config.mert330_window_sec, 9.0)
        self.assertEqual(config.mert330_window_overlap_sec, 1.5)
        self.assertEqual(config.mert330_batch_size, 2)

    def test_dialog_clamps_mert_overlap_to_window(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        dialog = SettingsDialog(config)
        dialog._mert_window.setValue(4.0)
        dialog._mert_overlap.setValue(5.0)  # >= window
        dialog.apply()
        self.assertEqual(config.mert_window_sec, 4.0)
        self.assertEqual(config.mert_window_overlap_sec, 3.5)  # 4.0 - 0.5

    def test_dialog_applies_v3_model_settings(self) -> None:
        """M2D-CLAP / MuQ / MuQ-MuLan pages write their config fields."""
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        dialog = SettingsDialog(config)
        dialog._muq_model_combo.setCurrentText("OpenMuQ/MuQ-large-music4all-iter")
        dialog._muq_window.setValue(12.0)
        dialog._muq_overlap.setValue(2.0)
        dialog._muq_batch.setValue(3)
        dialog._muqlan_batch.setValue(5)
        dialog._muqlan_top_k.setValue(9)
        dialog._muqlan_tags_edit.setPlainText("rock\njazz\n")
        dialog._m2dclap_batch.setValue(6)
        dialog._m2dclap_top_k.setValue(8)
        dialog._m2dclap_tags_edit.setPlainText("metal\nambient\n")
        dialog.apply()
        self.assertEqual(config.muq_model_id,
                         "OpenMuQ/MuQ-large-music4all-iter")
        self.assertEqual(config.muq_window_sec, 12.0)
        self.assertEqual(config.muq_window_overlap_sec, 2.0)
        self.assertEqual(config.muq_batch_size, 3)
        self.assertEqual(config.muqlan_batch_size, 5)
        self.assertEqual(config.muqlan_tag_top_k, 9)
        self.assertEqual(config.muqlan_tags, ["rock", "jazz"])
        self.assertEqual(config.m2dclap_batch_size, 6)
        self.assertEqual(config.m2dclap_tag_top_k, 8)
        self.assertEqual(config.m2dclap_tags, ["metal", "ambient"])

    def test_v3_model_pages_exist(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppConfig())
        labels = [dialog._category_list.item(i).text()
                  for i in range(dialog._category_list.count())]
        for page in ("M2D-CLAP", "MuQ", "MuQ-MuLan"):
            self.assertIn(page, labels)

    # ---- CLAP candidate-tag editor ----------------------------------------
    def test_tag_editor_prefills_effective_list(self) -> None:
        from app.models import clap_model
        from app.ui.settings_dialog import SettingsDialog

        # Default: the built-in candidate list, one per line.
        dialog = SettingsDialog(AppConfig())
        self.assertEqual(
            [line for line in
             dialog._clap_tags_edit.toPlainText().splitlines() if line],
            list(clap_model.CANDIDATE_TAGS))
        # A stored custom list prefills instead.
        config = AppConfig()
        config.clap_tags = ["field recording", "rain"]
        dialog = SettingsDialog(config)
        self.assertEqual(dialog._clap_tags_edit.toPlainText(),
                         "field recording\nrain")

    def test_tag_editor_parses_and_dedupes_on_apply(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        dialog = SettingsDialog(config)
        dialog._clap_tags_edit.setPlainText(
            "  rock \n\njazz\nROCK\nfield recording\n   \n")
        dialog.apply()
        # Trimmed, blanks dropped, duplicates collapse case-insensitively.
        self.assertEqual(config.clap_tags,
                         ["rock", "jazz", "field recording"])

    def test_tag_editor_stores_none_when_list_matches_default(self) -> None:
        from app.models import clap_model
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        dialog = SettingsDialog(config)
        dialog._clap_tags_edit.setPlainText(
            "\n".join(clap_model.CANDIDATE_TAGS))
        dialog.apply()
        self.assertIsNone(config.clap_tags)       # default → stored as None
        # An emptied editor also means "use the defaults".
        dialog._clap_tags_edit.setPlainText("  \n \n")
        dialog.apply()
        self.assertIsNone(config.clap_tags)

    def test_tag_editor_restore_default_button(self) -> None:
        from app.models import clap_model
        from app.ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(AppConfig())
        dialog._clap_tags_edit.setPlainText("just one tag")
        dialog._clap_tags_reset.click()
        self.assertEqual(dialog._clap_tags_edit.toPlainText(),
                         "\n".join(clap_model.CANDIDATE_TAGS))

    def test_dialog_prefills_from_config_and_allows_custom_id(self) -> None:
        from app.ui.settings_dialog import SettingsDialog

        config = AppConfig()
        config.clap_model_id = "laion/custom-clap"
        config.mert_model_id = "m-a-p/custom-mert"
        dialog = SettingsDialog(config)
        self.assertEqual(dialog._clap_model_combo.currentText(),
                         "laion/custom-clap")
        self.assertEqual(dialog._mert_model_combo.currentText(),
                         "m-a-p/custom-mert")
        # Custom ids survive apply() (editable combo, free text).
        dialog.apply()
        self.assertEqual(config.clap_model_id, "laion/custom-clap")
        self.assertEqual(config.mert_model_id, "m-a-p/custom-mert")


class RecordingPlugin:
    """Minimal fake plugin: records apply_config calls and describe top_k."""

    name = "rec"
    display_name = "Rec"
    embedding_dim = 4
    provides_text = True
    preferred_sample_rate = 48000
    requirements = ()

    def __init__(self):
        self._loaded = False
        self.applied_configs = []
        self.describe_top_ks = []
        self.tag_top_k = 5

    def is_available(self):
        return True

    def availability_error(self):
        return None

    def apply_config(self, config):
        self.applied_configs.append(config)
        self.tag_top_k = int(getattr(config, "clap_tag_top_k", 5))

    def embed(self, chunks, sr):
        return [np.ones(4, dtype=np.float32) for _ in chunks]

    def describe(self, chunks, sr, top_k=5):
        self.describe_top_ks.append(top_k)
        return [[("tagA", 0.9), ("tagB", 0.1)] for _ in chunks]


class PipelinePassThroughTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        from app.db.database import Database
        from app.db import repo
        self.db = Database(self.dir / "lib.db")

        self.wav = self.dir / "song.wav"
        sr = 8000
        t = np.linspace(0, 1.5, sr * 3 // 2, endpoint=False)
        sf.write(self.wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)

        with self.db.transaction() as conn:
            folder_id = repo.add_folder(conn, str(self.dir))
            self.track_id = repo.upsert_track(
                conn, folder_id, str(self.wav),
                {"filename": "song.wav", "extension": ".wav"})

    def tearDown(self):
        self._tmp.cleanup()

    def test_analyze_track_applies_config_and_top_k(self):
        import app.analysis.pipeline as pipeline
        from app.db import repo

        fake = RecordingPlugin()
        config = AppConfig()
        config.models = ["rec"]
        config.use_ollama = False
        config.chunk_seconds = 1.0
        config.overlap_percent = 50.0
        config.clap_tag_top_k = 3

        original = pipeline.get_plugin
        pipeline.get_plugin = lambda name: fake
        try:
            pipeline.analyze_track(self.db, self.track_id, config)
        finally:
            pipeline.get_plugin = original

        self.assertEqual(len(fake.applied_configs), 1)
        self.assertIs(fake.applied_configs[0], config)
        self.assertEqual(fake.describe_top_ks, [3])  # clap_tag_top_k passed through

        with self.db.transaction() as conn:
            track = repo.get_track(conn, self.track_id)
        self.assertEqual(track["status"], "analyzed")


if __name__ == "__main__":
    unittest.main()
