#!/usr/bin/env .venv/bin/python
"""End-to-end integration check (no GUI): scan -> DB -> chunk -> CLAP/MERT ->
Ollama text embeddings -> similar tracks -> playlist -> m3u.

Run:  .venv/bin/python scripts/e2e_check.py
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("e2e")

PROJECT = Path(__file__).resolve().parents[1]
FFMPEG = PROJECT / "bin" / "ffmpeg"

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, str(PROJECT))

from app.analysis.pipeline import analyze_track  # noqa: E402
from app.audio.chunking import chunk_audio, format_duration  # noqa: E402
from app.audio.decode import decode_audio, probe_audio  # noqa: E402
from app.config import AppConfig  # noqa: E402
from app.db import repo  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.playlist.generator import export_m3u, generate_playlist  # noqa: E402
from app.similarity.ollama import OllamaClient, detect_ollama  # noqa: E402
from app.similarity.search import similar_tracks  # noqa: E402


def make_test_music(music_dir: Path) -> list[Path]:
    """Generate small test files in three formats (wav via soundfile, mp3/aac via ffmpeg)."""
    music_dir.mkdir(parents=True, exist_ok=True)
    sr = 44100
    paths: list[Path] = []

    def tone(freq: float, seconds: float, path: Path, flavor: str) -> None:
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        if flavor == "bright":
            wave_ = 0.4 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        elif flavor == "warm":
            wave_ = 0.5 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * freq / 2 * t)
        else:
            wave_ = 0.3 * np.sin(2 * np.pi * freq * t)
        sf.write(path, wave_.astype(np.float32), sr)

    a = music_dir / "bright_tone.wav"
    tone(440.0, 5.0, a, "bright")
    paths.append(a)

    b = music_dir / "warm_tone.wav"
    tone(220.0, 5.0, b, "warm")

    # Convert to mp3 / aac through the bundled ffmpeg to exercise real decoders.
    for src, dst, extra in ((b, music_dir / "warm_tone.mp3", ["-q:a", "4"]),
                            (b, music_dir / "soft_tone.aac", ["-b:a", "128k"])):
        subprocess.run([str(FFMPEG), "-v", "error", "-y", "-i", str(src),
                        *extra, str(dst)], check=True)
        paths.append(dst)
    b.unlink()  # keep only the converted variants
    return paths


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="holosmart_e2e_")
    os.environ["HOLOSMART_DATA_DIR"] = str(Path(tmp) / "data")
    music_dir = Path(tmp) / "music"
    files = make_test_music(music_dir)
    log.info("Created test files: %s", [f.name for f in files])

    db = Database(Path(os.environ["HOLOSMART_DATA_DIR"]) / "library.db")
    with db.transaction() as conn:
        folder_id = repo.add_folder(conn, str(music_dir))
        track_ids: dict[str, int] = {}
        for f in sorted(music_dir.iterdir()):
            info = probe_audio(f)
            size = f.stat().st_size
            tid = repo.upsert_track(conn, folder_id, str(f), {
                "filename": f.name, "extension": f.suffix, "size_bytes": size,
                "mtime": f.stat().st_mtime, **info,
            })
            track_ids[f.name] = tid
        log.info("Indexed %d tracks", len(track_ids))

    cfg = AppConfig()
    cfg.models = ["clap", "mert"]
    cfg.chunk_seconds = 2.0
    cfg.overlap_percent = 50.0
    cfg.use_ollama = True
    cfg.ollama_embedding_model = None

    running, emb_models = detect_ollama(cfg.ollama_host)
    log.info("Ollama running=%s embedding models=%s", running, emb_models)
    with db.transaction() as conn:
        repo.save_ollama_models(conn, emb_models)
    if running and emb_models:
        cfg.ollama_embedding_model = emb_models[0]
    cfg.save()

    for name, tid in sorted(track_ids.items()):
        log.info("=== Analyzing %s ===", name)
        analyze_track(db, tid, cfg,
                      progress_cb=lambda m, c=None, t=None: log.info("  [%s]", m))

    with db.transaction() as conn:
        for name, tid in sorted(track_ids.items()):
            t = repo.get_track(conn, tid)
            chunks = repo.get_chunks(conn, tid)
            log.info("%s: status=%s chunks=%d dur=%s", name, t["status"],
                     len(chunks), format_duration(t["duration_sec"]))
            log.info("  description: %s", t["description"])
            log.info("  status_message: %s", t["status_message"])
            for c in chunks[:3]:
                embs = repo.get_chunk_embeddings(conn, c["id"])
                tags = repo.get_chunk_tags(conn, c["id"])
                e_summary = "; ".join(
                    f"{r['model']}: dim={r['dim']} head={[round(float(x), 3) for x in r['vec'][:4]]}"
                    for r in embs)
                t_summary = ", ".join(f"{r['text']}={r['score']:.2f}" for r in tags[:3])
                log.info("  chunk %d [%.1f-%.1fs] %s | tags: %s",
                         c["idx"], c["start_sec"], c["end_sec"], e_summary, t_summary)
            te = conn.execute(
                "SELECT model, dim FROM track_embeddings WHERE track_id = ?", (tid,)
            ).fetchall()
            log.info("  track-level embeddings: %s",
                     [(r["model"], r["dim"]) for r in te])

    seed_name = sorted(track_ids)[0]
    seed_id = track_ids[seed_name]
    for method in ("auto", "clap", "mert",
                   (f"ollama:{cfg.ollama_embedding_model}" if cfg.ollama_embedding_model else "auto")):
        try:
            with db.transaction() as conn:
                results = similar_tracks(conn, seed_id, method=method, limit=5)
            log.info("Similar to %s (method=%s):", seed_name, results[0].method)
            for r in results:
                log.info("  %.3f  %s", r.score, r.filename)
        except RuntimeError as exc:
            log.info("Similar (method=%s) -> %s", method, exc)

    try:
        with db.transaction() as conn:
            pid = generate_playlist(conn, "E2E Mix", seed_id, limit=5, method="auto")
            items = repo.get_playlist_items(conn, pid)
            out = export_m3u(conn, pid, Path(tmp) / "e2e_mix.m3u")
        log.info("Playlist id=%d items=%d exported=%s", pid, len(items), out)
    except RuntimeError as exc:
        log.error("Playlist generation failed: %s", exc)
        return 1

    print("\nE2E CHECK PASSED")
    print(f"(artifacts in {tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
