"""Track analysis pipeline: decode -> chunk -> model plugins -> persist results.

This module is the glue between the DB layer, audio decoding/chunking, the model
plugin registry and the similarity store. It runs inside worker threads only.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np

from app.audio.chunking import chunk_audio
from app.audio.decode import decode_audio
from app.config import AppConfig
from app.db.database import Database
from app.db import repo
from app.models.registry import get_plugin

if TYPE_CHECKING:  # pragma: no cover
    from app.models.base import ModelPlugin

log = logging.getLogger(__name__)

#: Progress callback: (message, current, total).
ProgressCb = Callable[[str, int | None, int | None], None]

#: How many top tags (aggregated over chunks) form the human-readable description.
DESCRIPTION_TAG_COUNT = 8


def aggregate_description(tag_rows: list[tuple[str, float]]) -> str | None:
    """Aggregate per-chunk ``[(tag, score), ...]`` into a short text description.

    Public so other modules can reuse the same aggregation the analysis
    pipeline applies to the CLAP chunk tags.
    """
    if not tag_rows:
        return None
    totals: dict[str, float] = {}
    for tag, score in tag_rows:
        totals[tag] = totals.get(tag, 0.0) + float(score)
    top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:DESCRIPTION_TAG_COUNT]
    if not top:
        return None
    return "Music tags: " + ", ".join(tag for tag, _ in top)


def _run_plugin(
    plugin: "ModelPlugin",
    db: Database,
    chunk_samples: list[np.ndarray],
    sr: int,
    chunk_ids: list[int],
    notify: ProgressCb | None = None,
) -> tuple[int, list[tuple[str, float]]]:
    """Embed (+ optionally tag) all chunks with one plugin, batch by batch.

    Chunks are processed in batches of the plugin's ``batch_size`` (falling
    back to ``BATCH_SIZE``, then 8). Model inference happens OUTSIDE any DB
    transaction; each batch's results are persisted in one short transaction
    right after its inference, and every finished batch is reported through
    ``notify`` (cur/tot = chunk counts) so the UI can show per-chunk progress
    and the worker's stop flag is re-checked once per batch. Returns
    (vectors_stored, tag_rows).
    """
    notify = notify or (lambda msg, cur=None, tot=None: None)
    total = len(chunk_samples)
    batch = int(getattr(plugin, "batch_size", 0)
                or getattr(plugin, "BATCH_SIZE", 8) or 8)
    batch = max(1, batch)

    def batches() -> Iterator[tuple[list[np.ndarray], list[int]]]:
        for start in range(0, total, batch):
            yield chunk_samples[start:start + batch], chunk_ids[start:start + batch]

    # ---- embed phase: one inference + one short persist per batch ----------
    stored = 0
    done = 0
    for samples_batch, ids_batch in batches():
        vectors = plugin.embed(samples_batch, sr)
        with db.transaction() as conn:
            for chunk_id, vec in zip(ids_batch, vectors):  # zip: never crash
                vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                repo.add_chunk_embedding(conn, chunk_id, plugin.name, vec)
                stored += 1
        done += len(samples_batch)
        notify(f"{plugin.display_name}: chunk {done}/{total}", done, total)

    # ---- tags phase: one describe + one short persist per batch ------------
    tag_rows: list[tuple[str, float]] = []
    if plugin.provides_text:
        done = 0
        for samples_batch, ids_batch in batches():
            described = plugin.describe(
                samples_batch, sr, top_k=getattr(plugin, "tag_top_k", 5))
            if described:
                prepared: list[tuple[int, list[tuple[str, float]]]] = []
                for chunk_id, tags in zip(ids_batch, described):
                    clean = [(str(t), float(s)) for t, s in tags if t]
                    if clean:
                        prepared.append((chunk_id, clean))
                        tag_rows.extend(clean)
                with db.transaction() as conn:
                    for chunk_id, tags in prepared:
                        repo.add_chunk_tags(conn, chunk_id, plugin.name, tags)
            done += len(samples_batch)
            notify(f"{plugin.display_name}: tags {done}/{total}", done, total)

    return stored, tag_rows


def analyze_track(
    db: Database,
    track_id: int,
    config: AppConfig,
    progress_cb: ProgressCb | None = None,
    force: bool = False,
) -> None:
    """Analyze one track end-to-end and persist chunks/embeddings/tags/description.

    Raises RuntimeError on hard failures (decode problems, no usable models);
    the failure is always reflected in the tracks table before raising.
    """
    notify: ProgressCb = progress_cb or (lambda msg, cur=None, tot=None: None)

    with db.transaction() as conn:
        track = repo.get_track(conn, track_id)
    if track is None:
        raise RuntimeError(f"Track {track_id} not found in database")
    path = track["path"]
    log.info("Analyzing track %s (%s)", track_id, path)

    with db.transaction() as conn:
        repo.set_track_status(conn, track_id, "analyzing")

    # 1. Decode ----------------------------------------------------------------
    notify("Decoding audio", None, None)
    try:
        samples, sr = decode_audio(path, mono=True)
    except Exception as exc:
        msg = f"Decode failed: {exc}"
        with db.transaction() as conn:
            repo.set_track_status(conn, track_id, "error", msg)
        raise RuntimeError(msg) from exc
    if samples.size == 0:
        msg = "Audio file decoded to zero samples"
        with db.transaction() as conn:
            repo.set_track_status(conn, track_id, "error", msg)
        raise RuntimeError(msg)

    duration = len(samples) / float(sr)
    try:
        stat = os.stat(path)
        source_mtime: float | None = stat.st_mtime
        source_size: int | None = stat.st_size
    except OSError:  # pragma: no cover - file vanished mid-analysis
        source_mtime = source_size = None
    with db.transaction() as conn:
        meta: dict = {"duration_sec": duration,
                      "source_mtime": source_mtime,
                      "source_size": source_size}
        if track["sample_rate"] is None:
            meta["sample_rate"] = int(sr)
        repo.upsert_track(conn, track["folder_id"], path, meta)

    # 2. Chunk -------------------------------------------------------------------
    chunks = chunk_audio(
        samples, sr,
        chunk_seconds=config.chunk_seconds,
        overlap_percent=config.overlap_percent,
    )
    if not chunks:
        msg = "Chunking produced no chunks"
        with db.transaction() as conn:
            repo.set_track_status(conn, track_id, "error", msg)
        raise RuntimeError(msg)
    plan = [(c.idx, c.start_sec, c.end_sec) for c in chunks]

    # Incremental re-analysis: when the chunk plan is unchanged AND the file
    # on disk has not changed since the last analysis, the stored chunks (and
    # every model's vectors on them) stay; models that already cover all
    # chunks are skipped.  This makes "enable one more model and re-analyze"
    # additive instead of destructive.  Otherwise — chunking parameters
    # changed, the file changed, or the first-ever run — the chunks are
    # replaced, which cascades away all stored vectors.
    with db.transaction() as conn:
        existing = repo.get_chunks(conn, track_id)
        file_stable = (
            track["source_mtime"] is None
            or (source_mtime is not None
                and abs(float(track["source_mtime"]) - source_mtime) < 1e-6
                and (track["source_size"] is None
                     or track["source_size"] == source_size))
        )
        same_plan = (
            len(existing) == len(plan)
            and file_stable
            and all(int(e["idx"]) == p[0]
                    and abs(float(e["start_sec"]) - p[1]) < 1e-6
                    and abs(float(e["end_sec"]) - p[2]) < 1e-6
                    for e, p in zip(existing, plan))
        )
        if same_plan:
            chunk_ids = [int(e["id"]) for e in existing]
            covered: dict[str, int] = {
                str(r["model"]): int(r["n"]) for r in conn.execute(
                    "SELECT e.model AS model, COUNT(DISTINCT e.chunk_id) AS n "
                    "FROM embeddings e JOIN chunks c ON c.id = e.chunk_id "
                    "WHERE c.track_id = ? GROUP BY e.model", (track_id,))
            }
        else:
            chunk_ids = repo.replace_chunks(conn, track_id, plan)
            covered = {}
    chunk_samples = [c.samples for c in chunks]
    if same_plan:
        log.info("Track %s: keeping %d chunks (plan unchanged)",
                 track_id, len(chunk_ids))
    else:
        log.info("Track %s: %d chunks (%.1fs / %.0f%% overlap)",
                 track_id, len(chunks), config.chunk_seconds,
                 config.overlap_percent)

    # 3. Model plugins -------------------------------------------------------------
    all_tag_rows: list[tuple[str, float]] = []
    stored_any = False
    notes: list[str] = []
    for name in config.models:
        try:
            plugin = get_plugin(name)
        except KeyError:
            notes.append(f"unknown model '{name}' skipped")
            continue
        if not plugin.is_available():
            note = f"{plugin.display_name} unavailable ({plugin.availability_error()})"
            notes.append(note)
            log.warning("Skipping plugin: %s", note)
            continue
        if same_plan and not force and covered.get(name, 0) >= len(chunk_ids):
            # Full vector coverage from a previous run: keep it instead of
            # recomputing (and instead of losing it to a re-chunk).  A
            # skipped text model's stored tags still feed the description.
            notes.append(f"{plugin.display_name}: already analyzed (kept)")
            if plugin.provides_text:
                with db.transaction() as conn:
                    all_tag_rows.extend(
                        (str(r["text"]), r["score"])
                        for r in repo.get_track_tags(conn, track_id, name))
            continue
        apply_cfg = getattr(plugin, "apply_config", None)
        if callable(apply_cfg):
            apply_cfg(config)
        notify(f"Running {plugin.display_name}", None, None)
        try:
            stored, tag_rows = _run_plugin(plugin, db, chunk_samples, sr,
                                           chunk_ids, notify=notify)
        except Exception as exc:
            note = f"{plugin.display_name} failed: {exc}"
            notes.append(note)
            notify(f"⚠ {note}", None, None)
            log.exception("Plugin %s failed", plugin.display_name)
            continue
        stored_any = stored_any or stored > 0
        all_tag_rows.extend(tag_rows)
        note = f"{plugin.display_name}: {stored} chunk embeddings"
        if tag_rows:
            note += f", {len(tag_rows)} tags"
        notes.append(note)

    # 4. Human-readable description + track-level embeddings ------------------------
    description = aggregate_description(all_tag_rows)
    if description:
        with db.transaction() as conn:
            repo.set_track_description(conn, track_id, description)

    # Track-level vectors: chunk centroids for every model present (always), plus
    # an Ollama text embedding of the description when enabled/reachable.
    try:
        from app.similarity.ollama import OllamaClient
        from app.similarity.search import ensure_track_embeddings

        client: OllamaClient | None = None
        emb_model: str | None = None
        if config.use_ollama and description:
            notify("Ollama text embedding", None, None)
            candidate = OllamaClient(config.ollama_host)
            if candidate.is_running():
                client = candidate
                emb_model = config.ollama_embedding_model
                if not emb_model:
                    with db.transaction() as conn:
                        saved = repo.get_ollama_models(conn)
                    emb_model = saved[0] if saved else None
                if not emb_model:
                    notes.append("Ollama running but no embedding model detected")
            else:
                notes.append("Ollama not reachable, text embedding skipped")

        with db.transaction() as conn:
            ensure_track_embeddings(conn, [track_id], client, emb_model)
        if client is not None and emb_model:
            notes.append(f"Ollama text embedding ({emb_model}) stored")
    except Exception as exc:
        notes.append(f"Track-level embedding update failed: {exc}")
        log.warning("ensure_track_embeddings failed: %s", exc)

    # 5. Final status -----------------------------------------------------------------
    if not stored_any:
        # Fully-covered incremental run: nothing new was stored, but every
        # enabled model already covers all kept chunks — a success (the
        # "already analyzed (kept)" notes explain what happened), not an
        # error.  A mid-batch plugin failure leaves only PARTIAL coverage,
        # which still ends in the error state.
        with db.transaction() as conn:
            n_chunks = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE track_id = ?",
                (track_id,)).fetchone()[0]
            fully_covered = conn.execute(
                "SELECT e.model FROM embeddings e "
                "JOIN chunks c ON c.id = e.chunk_id "
                "WHERE c.track_id = ? GROUP BY e.model "
                "HAVING COUNT(DISTINCT e.chunk_id) = ? LIMIT 1",
                (track_id, n_chunks)).fetchone()
        if fully_covered is None:
            msg = "No analysis models available. " + "; ".join(notes)
            with db.transaction() as conn:
                repo.set_track_status(conn, track_id, "error", msg)
            raise RuntimeError(msg)

    msg = "; ".join(notes) if notes else None
    with db.transaction() as conn:
        repo.set_track_status(conn, track_id, "analyzed", msg)
    notify("Analysis complete", len(chunks), len(chunks))
    log.info("Track %s analyzed: %s", track_id, msg)
