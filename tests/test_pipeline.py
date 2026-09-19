"""Pipeline tests with a fake model plugin (no torch, no network, no GUI)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from app.analysis.pipeline import analyze_track
from app.config import AppConfig
from app.db.database import Database
from app.db import repo
from app.models.base import ModelPlugin


class FakePlugin(ModelPlugin):
    """Deterministic stand-in: 4-dim vectors from chunk mean, two tags per chunk."""

    name = "fake"
    display_name = "Fake"
    embedding_dim = 4
    provides_text = True
    preferred_sample_rate = 48000
    requirements = ()

    def _embed(self, chunks, sr):
        out = []
        for c in chunks:
            mean = float(np.mean(c)) if c.size else 0.0
            out.append(np.array([mean, 1.0, -1.0, 0.5], dtype=np.float32))
        return out

    def _describe(self, chunks, sr, top_k):
        return [[("tagA", 0.9), ("tagB", 0.1)] for _ in chunks]


class UnavailablePlugin(ModelPlugin):
    name = "gone"
    display_name = "Gone"
    requirements = ("definitely_not_installed_pkg_xyz",)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db = Database(self.dir / "lib.db")

        self.wav = self.dir / "song.wav"
        sr = 8000
        t = np.linspace(0, 2.5, sr * 3 // 1, endpoint=False)[: sr * 5 // 2]
        sf.write(self.wav, 0.3 * np.sin(2 * np.pi * 440 * t), sr)

        with self.db.transaction() as conn:
            folder_id = repo.add_folder(conn, str(self.dir))
            self.track_id = repo.upsert_track(
                conn, folder_id, str(self.wav),
                {"filename": "song.wav", "extension": ".wav"},
            )

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, models, **over):
        cfg = AppConfig()
        cfg.models = list(models)
        cfg.use_ollama = False
        cfg.chunk_seconds = 1.0
        cfg.overlap_percent = 50.0
        for k, v in over.items():
            setattr(cfg, k, v)
        return cfg

    def test_analyze_track_full_flow(self):
        import app.analysis.pipeline as pipeline

        original = pipeline.get_plugin
        pipeline.get_plugin = lambda name: FakePlugin()
        try:
            analyze_track(self.db, self.track_id, self._config(["fake"]))
        finally:
            pipeline.get_plugin = original

        with self.db.transaction() as conn:
            track = repo.get_track(conn, self.track_id)
            self.assertEqual(track["status"], "analyzed")
            # 2.5 s audio, 1.0 s chunks, 50% overlap (hop 0.5) -> starts 0,0.5,1,1.5
            # (the window at 1.5 already reaches the end, so no window at 2.0)
            chunks = repo.get_chunks(conn, self.track_id)
            self.assertEqual([c["idx"] for c in chunks], [0, 1, 2, 3])
            self.assertAlmostEqual(chunks[0]["start_sec"], 0.0, places=3)
            self.assertAlmostEqual(chunks[0]["end_sec"], 1.0, places=3)
            self.assertAlmostEqual(chunks[1]["start_sec"], 0.5, places=3)
            self.assertAlmostEqual(chunks[-1]["end_sec"], 2.5, places=2)

            embs = [repo.get_chunk_embeddings(conn, c["id"]) for c in chunks]
            for rows in embs:
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["model"], "fake")
                self.assertEqual(rows[0]["dim"], 4)
                self.assertEqual(len(rows[0]["vec"]), 4)

            tags = repo.get_track_tags(conn, self.track_id)
            self.assertEqual(len(tags), 8)  # 4 chunks x 2 tags
            self.assertEqual(tags[0]["text"], "tagA")  # highest score first

            self.assertIsNotNone(track["description"])
            self.assertIn("tagA", track["description"])

            # centroid stored by ensure_track_embeddings
            centroid = repo.get_track_embedding(conn, self.track_id, "fake")
            self.assertIsNotNone(centroid)
            self.assertEqual(centroid.shape, (4,))
            self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=3)

    def test_analyze_missing_file_sets_error(self):
        with self.db.transaction() as conn:
            folder_id = repo.add_folder(conn, str(self.dir / "nope"))
            bad_id = repo.upsert_track(conn, folder_id, str(self.dir / "ghost.mp3"), {})

        with self.assertRaises(RuntimeError):
            analyze_track(self.db, bad_id, self._config(["fake"]))

        with self.db.transaction() as conn:
            track = repo.get_track(conn, bad_id)
        self.assertEqual(track["status"], "error")
        self.assertIn("Decode failed", track["status_message"])

    def test_no_available_models_sets_error(self):
        import app.analysis.pipeline as pipeline

        original = pipeline.get_plugin
        pipeline.get_plugin = lambda name: UnavailablePlugin()
        try:
            with self.assertRaises(RuntimeError):
                analyze_track(self.db, self.track_id, self._config(["gone"]))
        finally:
            pipeline.get_plugin = original

        with self.db.transaction() as conn:
            track = repo.get_track(conn, self.track_id)
        self.assertEqual(track["status"], "error")
        self.assertIn("No analysis models available", track["status_message"])
        # chunks are still persisted for inspection
        with self.db.transaction() as conn:
            self.assertEqual(len(repo.get_chunks(conn, self.track_id)), 4)


if __name__ == "__main__":
    unittest.main()
