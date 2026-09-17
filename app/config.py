"""Application configuration: paths, defaults, load/save of the JSON config file."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "HoloSmart Music Explorer"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HOLOSMART_DATA_DIR", PROJECT_ROOT / "data"))
DB_PATH = Path(os.environ.get("HOLOSMART_DB", DATA_DIR / "library.db"))
CONFIG_PATH = DATA_DIR / "config.json"

#: File extensions scanned as music. .m4a is AAC in an MP4 container.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".ape", ".aac", ".m4a")


def analysis_excluded_extensions(config: "AppConfig") -> tuple[str, ...]:
    """Extensions that are NOT subject to analysis under *config*.

    WAV is excluded by default (``analyze_wav`` off): such files stay in the
    library and remain playable, but they are greyed out in the library tree
    and do not count toward the analysis-progress percentages.
    """
    return () if getattr(config, "analyze_wav", False) else (".wav",)


@dataclass
class AppConfig:
    """User-tunable settings persisted to data/config.json."""

    chunk_seconds: float = 20.0
    overlap_percent: float = 50.0
    # Model plugin names to run during analysis; unknown names are ignored with a warning.
    models: list[str] = field(default_factory=lambda: ["clap", "mert", "openl3", "fft"])
    # Ollama integration (similarity via text embeddings of track descriptions).
    use_ollama: bool = True
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str | None = None
    # "auto" or "ollama:<model>" or "clap"/"mert"/"openl3"
    similarity_method: str = "auto"
    playlist_length: int = 15
    # CLAP plugin settings (defaults = the module constants in app.models.clap_model).
    clap_model_id: str = "laion/clap-htsat-unfused"
    clap_tag_top_k: int = 5
    clap_batch_size: int = 8
    # Candidate tags for CLAP zero-shot tagging (Settings → CLAP tag editor,
    # one tag per line). None = use the built-in CANDIDATE_TAGS list, so a
    # plugin update can ship a better default list; an explicit list (even
    # one identical to the default) is stored once the user edits it.
    clap_tags: list[str] | None = None
    # MERT plugin settings (defaults = the module constants in app.models.mert_model).
    mert_model_id: str = "m-a-p/MERT-v1-95M"
    mert_window_sec: float = 10.0
    mert_window_overlap_sec: float = 1.0
    mert_batch_size: int = 8
    # MERT-330M plugin settings (large MERT variant; opt-in because the
    # weights download is ~1.3 GB and inference is ~4x slower than 95M).
    mert330_model_id: str = "m-a-p/MERT-v1-330M"
    mert330_window_sec: float = 10.0
    mert330_window_overlap_sec: float = 1.0
    mert330_batch_size: int = 4
    # FFT plugin settings (defaults = the module constants in app.models.fft_model).
    fft_window_sec: float = 10.0
    # How many tracks the analysis worker processes concurrently (1 = the old
    # sequential behavior). Values < 1 are clamped to 1 where the worker uses
    # the value, never at load time.
    # Legacy: no longer widens any run (FFT-only runs parallelize across
    # cores; every other model runs sequentially). Kept so existing
    # config.json files load unchanged.
    analysis_parallelism: int = 2
    # Performance checkbox: skip files longer than 20 minutes during batch
    # analysis (threshold fixed in app.ui.workers.LONG_TRACK_SEC). The files
    # stay in the library and remain playable; explicitly re-analyzing one
    # selected file ignores the skip.
    analysis_skip_long_files: bool = False
    # WAV files are excluded from analysis runs by default: raw-PCM decode +
    # embed work is disproportionately expensive and WAV is rarely the
    # primary format. The files still scan into the library and stay
    # playable; set True (config.json) to include them in batch analysis.
    # An explicitly forced single-file analysis ignores the exclusion.
    analyze_wav: bool = False
    # Version of the persisted settings shape. Bumped when a new model plugin
    # ships that should be auto-enabled ONCE in existing installations: a
    # config file saved without this field (i.e. written by an older build)
    # gets the new models appended exactly once; afterwards the user's own
    # on/off choice is respected because save() persists the version.
    models_version: int = 2

    @staticmethod
    def load() -> "AppConfig":
        cfg = AppConfig()
        try:
            if CONFIG_PATH.exists():
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                known = {f for f in cfg.__dataclass_fields__}  # type: ignore[attr-defined]
                for key, value in raw.items():
                    if key in known:
                        setattr(cfg, key, value)
                # One-time migration for configs written before the FFT plugin
                # existed (no models_version field): enable "fft" once. After
                # the first save the version field is present, so a deliberately
                # disabled FFT stays disabled.
                if "models_version" not in raw and "fft" not in cfg.models:
                    cfg.models.append("fft")
        except Exception:  # corrupted config must never prevent startup
            pass
        return cfg

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
