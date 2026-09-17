"""Audio decoding and metadata probing.

Decoding strategy (in order): project-local/system ffmpeg pipe (supports every
target format incl. aac/ape), libsndfile via soundfile (wav/flac/mp3), stdlib
wave. Probing strategy: ffprobe JSON, mutagen, soundfile.info. All values that
cannot be determined are None; probe_audio never raises for readable files.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_BIN = _PROJECT_ROOT / "bin"


def ffmpeg_path() -> str | None:
    """Path of the ffmpeg binary: project-local bin/ first, then PATH."""
    local = _LOCAL_BIN / "ffmpeg"
    if local.is_file():
        return str(local)
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    """Path of the ffprobe binary: project-local bin/ first, then PATH."""
    local = _LOCAL_BIN / "ffprobe"
    if local.is_file():
        return str(local)
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def _suffix(path: str | Path) -> str:
    return Path(path).suffix.lower().lstrip(".")


def probe_audio(path: str | Path) -> dict:
    """Probe codec/container/sample-format metadata. Keys may be None."""
    path = Path(path)
    info: dict[str, Any] = {
        "container": _suffix(path) or None,
        "codec": None,
        "sample_rate": None,
        "channels": None,
        "bit_depth": None,
        "bitrate_kbps": None,
        "duration_sec": None,
    }
    if not path.is_file():
        return info

    # 1) ffprobe (richest, handles ape/aac)
    pp = ffprobe_path()
    if pp:
        try:
            proc = subprocess.run(
                [pp, "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                stream = next((s for s in data.get("streams", [])
                               if s.get("codec_type") == "audio"), None)
                fmt = data.get("format", {})
                if stream:
                    info["codec"] = stream.get("codec_name")
                    info["sample_rate"] = _int(stream.get("sample_rate"))
                    info["channels"] = _int(stream.get("channels"))
                    info["bit_depth"] = (_int(stream.get("bits_per_raw_sample"))
                                         or _int(stream.get("bits_per_sample")))
                    rate = _float(stream.get("bit_rate"))
                else:
                    rate = None
                duration = (_float(fmt.get("duration"))
                            or (stream and _float(stream.get("duration"))))
                if duration is None:
                    duration = _float(stream.get("duration")) if stream else None
                if rate is None and fmt.get("bit_rate"):
                    rate = _float(fmt.get("bit_rate"))
                info["duration_sec"] = duration
                info["bitrate_kbps"] = rate / 1000.0 if rate else None
                if fmt.get("format_name"):
                    names = fmt["format_name"].split(",")
                    info["container"] = names[0].strip() if names else info["container"]
                return info
        except (subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
            pass

    # 2) mutagen
    try:
        from mutagen import File as MutagenFile
        mfile = MutagenFile(str(path))
        if mfile is not None and getattr(mfile, "info", None) is not None:
            minfo = mfile.info
            info["sample_rate"] = getattr(minfo, "sample_rate", None)
            info["channels"] = getattr(minfo, "channels", None)
            info["bit_depth"] = getattr(minfo, "bits_per_sample", None)
            bitrate = getattr(minfo, "bitrate", None)
            info["bitrate_kbps"] = bitrate / 1000.0 if bitrate else None
            info["duration_sec"] = getattr(minfo, "length", None)
            if info["duration_sec"]:
                info["duration_sec"] = float(info["duration_sec"])
            if info["sample_rate"] is not None:
                return info
    except Exception:
        pass

    # 3) soundfile
    try:
        import soundfile as sf
        sfi = sf.info(str(path))
        info["sample_rate"] = sfi.samplerate
        info["channels"] = sfi.channels
        enc = (sfi.subtype or "")
        if enc.startswith("PCM_"):
            digits = "".join(ch for ch in enc[4:] if ch.isdigit())
            info["bit_depth"] = int(digits) if digits else None
        info["duration_sec"] = sfi.duration
        info["codec"] = sfi.format.lower()
    except Exception:
        pass
    return info


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decode_audio(path: str | Path, target_sr: int | None = None,
                 mono: bool = True) -> tuple[np.ndarray, int]:
    """Decode audio to float32 samples.

    Returns ``(samples, sample_rate)`` where samples is 1-D when ``mono`` else
    ``(n, channels)``. Uses the ffmpeg pipe for full format coverage (mp3, wav,
    flac, ape, aac/m4a); falls back to soundfile / stdlib wave when ffmpeg is
    unavailable. Raises RuntimeError with an actionable message on failure.
    """
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"File not found: {path}")

    ff = ffmpeg_path()
    if ff:
        cmd = [ff, "-v", "error", "-i", str(path),
               "-f", "f32le", "-acodec", "pcm_f32le"]
        if mono:
            cmd += ["-ac", "1"]
        if target_sr:
            cmd += ["-ar", str(int(target_sr))]
        cmd += ["pipe:1"]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=600)
            if proc.returncode == 0 and proc.stdout:
                samples = np.frombuffer(proc.stdout, dtype="<f4").copy()
                sr = int(target_sr) if target_sr else (
                    probe_audio(path).get("sample_rate") or 44100)
                return samples, sr
            err = proc.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err[:400]}")
        except FileNotFoundError:
            pass  # fall through to non-ffmpeg decoders
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Decoding timed out: {path}") from exc

    # soundfile (wav / flac / mp3 via libsndfile 1.2+)
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if mono:
            data = data.mean(axis=1)
        samples = np.ascontiguousarray(data, dtype=np.float32).reshape(-1) if mono \
            else np.ascontiguousarray(data, dtype=np.float32)
        if target_sr and sr != int(target_sr):
            from app.audio.resample import resample
            samples = resample(samples, sr, int(target_sr))
            sr = int(target_sr)
        return samples, int(sr)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot decode '{path}' ({_suffix(path) or 'unknown format'}): {exc}. "
            "Formats aac/ape require ffmpeg — install it (project bin/ffmpeg "
            "or `brew install ffmpeg`)."
        ) from exc


def decode_wav_stdlib(path: str | Path) -> tuple[np.ndarray, int]:
    """Last-resort WAV decoder using only the standard library (16-bit PCM)."""
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {width * 8} bit")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr
