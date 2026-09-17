"""FFT spectral-statistics analysis plugin.

A dependency-light analysis model (numpy only, no weights): every analysis
chunk is split into windows of ``fft_window_sec`` (default **10 s**,
non-overlapping), each window is Hann-windowed and transformed with a real
FFT, and the amplitude spectrum is reduced to a fixed-length feature vector:

* per frequency band (``BANDS``: 0-50, 50-300, 300-1000, 1000-5000,
  5000-15000, 15000 Hz-Nyquist) — mean amplitude, standard deviation, skew,
  (excess) kurtosis, RMS and crest factor of the band's bin amplitudes;
* for the whole spectrum — the dominant frequency (bin with the largest
  amplitude), the spectral centroid (amplitude-weighted center of mass), the
  Shannon entropy of the normalized power distribution, and the Hurst
  exponent (rescaled-range R/S) of the log-amplitude spectrum treated as a
  profile over frequency.

The per-window statistics are averaged into the chunk's feature vector and
stored under the model name ``"fft"`` in the regular ``embeddings`` table, so
track-level centroids and similarity search work exactly like they do for the
CLAP/MERT vectors.

Scaling: band amplitudes are expressed relative to the window's overall RMS
and frequencies relative to the Nyquist frequency, which makes every stored
component dimensionless and loudness-invariant — two renders of the same
recording at different levels produce nearly identical vectors.  The stored
vector is the L2-normalized, range-clipped view of those dimensionless
statistics (:func:`compress_and_normalize`).
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from app.models.base import AudioChunks, ModelPlugin

log = logging.getLogger(__name__)

#: Frequency bands in Hz as ``(low, high)`` pairs; ``None`` means "up to and
#: including the Nyquist frequency".  Bands are half-open ``[low, high)``.
BANDS: tuple[tuple[float, float | None], ...] = (
    (0.0, 50.0),
    (50.0, 300.0),
    (300.0, 1000.0),
    (1000.0, 5000.0),
    (5000.0, 15000.0),
    (15000.0, None),
)

#: Statistics computed per band, in vector order: mean amplitude, standard
#: deviation, skew, excess kurtosis, RMS, crest factor.
BAND_STATS = 6

#: Whole-spectrum features, in vector order: dominant frequency (relative to
#: Nyquist), spectral centroid (relative to Nyquist), Shannon spectral
#: entropy (relative to the bin count), Hurst exponent.
WHOLE_FEATURES = 4

#: Total feature-vector width: 6 bands x 6 band stats + 4 spectral features.
FEATURE_DIM = len(BANDS) * BAND_STATS + WHOLE_FEATURES

#: Minimum window length (seconds) accepted from configuration.
MIN_WINDOW_SEC = 0.5

#: Stored components are dimensionless; this clip only guards pathological
#: shape statistics (e.g. a single-bin band) from dominating the vector.
COMPRESS_CLIP = 10.0

#: Floor added before taking the logarithm of spectral amplitudes.
_LOG_FLOOR = 1e-12


#: Statistics per band, in vector order (see the module docstring of
#: :func:`fft_features`).
_BAND_STAT_NAMES = ("mean", "std", "skew", "kurtosis", "RMS", "crest")

#: Whole-spectrum feature names, in vector order.
_WHOLE_FEATURE_NAMES = ("dominant frequency (×Nyquist)",
                        "spectral centroid (×Nyquist)",
                        "spectral entropy (×bins)",
                        "Hurst exponent")


def feature_names() -> tuple[str, ...]:
    """Human-readable name for every dimension of the feature vector.

    Order matches :func:`fft_features` exactly: per band (in
    :data:`BANDS` order) the six band statistics, then the four
    whole-spectrum features.
    """
    names: list[str] = []
    for low, high in BANDS:
        band = f"{low:g}–{high:g} Hz" if high is not None else f"{low:g} Hz+"
        names.extend(f"{band} {stat}" for stat in _BAND_STAT_NAMES)
    names.extend(_WHOLE_FEATURE_NAMES)
    assert len(names) == FEATURE_DIM
    return tuple(names)


def band_indices(freqs: np.ndarray, low: float, high: float | None) -> np.ndarray:
    """Boolean mask of the FFT bins belonging to the ``[low, high)`` band.

    ``high=None`` selects everything up to and including the Nyquist bin
    (the last bin of a real FFT), i.e. the "15000 Hz+" band.
    """
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if high is None:
        return freqs >= low
    return (freqs >= low) & (freqs < high)


def moment_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, std, skew, excess kurtosis) of a 1-D sample, population moments.

    Skew is the Fisher-Pearson coefficient ``m3 / m2**1.5`` and kurtosis the
    excess variant ``m4 / m2**2 - 3`` (a normal distribution yields 0).  An
    empty input or zero variance yields zeros — empty frequency bands (e.g.
    above Nyquist) stay at zero instead of producing NaN.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(x.mean())
    d = x - mean
    m2 = float(np.mean(d * d))
    if m2 <= 0.0:
        return mean, 0.0, 0.0, 0.0
    m3 = float(np.mean(d ** 3))
    m4 = float(np.mean(d ** 4))
    skew = m3 / m2 ** 1.5
    kurtosis = m4 / (m2 * m2) - 3.0
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 0.0
    return mean, float(np.sqrt(m2)), skew, kurtosis


def shannon_entropy(amplitudes: np.ndarray) -> float:
    """Shannon entropy (bits) of the power distribution of a spectrum.

    ``p = amplitude**2 / sum(amplitude**2)`` and
    ``H = -sum(p * log2(p))`` over bins with ``p > 0``.  A silent spectrum
    (zero total power) yields 0.0.
    """
    power = np.asarray(amplitudes, dtype=np.float64).reshape(-1) ** 2
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    p = power / total
    nz = p[p > 0.0]
    return float(-(nz * np.log2(nz)).sum())


def hurst_exponent(series: np.ndarray) -> float:
    """Hurst exponent of a 1-D series via rescaled-range (R/S) analysis.

    For segment sizes on a geometric ladder between 8 points and ``N // 2``
    the rescaled range ``R/S`` — range of the cumulative deviations from the
    segment mean divided by the segment's standard deviation — is averaged
    over all segments of that size; ``H`` is the slope of ``log(mean R/S)``
    against ``log(size)``, clipped to ``[0, 1]``.  Degenerate inputs (fewer
    than 8 points, zero variance, or a non-finite fit) return ``0.5`` — the
    random-walk baseline — so the feature vector never carries NaN.  All
    segment statistics are computed vectorized (one reshape per ladder size).
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 8 or float(np.std(x)) <= 0.0:
        return 0.5
    sizes = sorted({int(round(s)) for s in np.geomspace(8, max(8, n // 2), 6)})
    points: list[tuple[float, float]] = []
    for size in sizes:
        count = n // size
        if size < 2 or count < 2:
            continue
        segments = np.asarray(x[: count * size], dtype=np.float64).reshape(
            count, size)
        std = segments.std(axis=1)
        deviations = np.cumsum(segments - segments.mean(axis=1, keepdims=True),
                               axis=1)
        ranges = deviations.max(axis=1) - deviations.min(axis=1)
        ratios = ranges[std > 0.0] / std[std > 0.0]
        if ratios.size:
            points.append((float(size), float(ratios.mean())))
    if len(points) < 2:
        return 0.5
    log_sizes = np.log([size for size, _ in points])
    log_rs = np.log([rs for _, rs in points])
    slope = float(np.polyfit(log_sizes, log_rs, 1)[0])
    if not np.isfinite(slope):
        return 0.5
    return max(0.0, min(1.0, slope))


def spectral_features(samples: np.ndarray, sr: int) -> tuple[float, ...]:
    """Raw ``FEATURE_DIM``-float feature tuple of one window of audio.

    Layout: for each band in :data:`BANDS` order, the band stats
    ``(mean amplitude, std, skew, excess kurtosis, RMS, crest factor)`` of
    the band's bin amplitudes — mean/std/RMS divided by the window's overall
    RMS so they are loudness-invariant — followed by ``(dominant frequency,
    spectral centroid, spectral entropy, Hurst exponent)`` computed over the
    whole spectrum; frequencies are relative to Nyquist and the entropy to
    the bin count, so all components are dimensionless.  The input is
    Hann-windowed here, amplitudes are
    normalized to the window length (``2*|X|/n``, DC not doubled), and every
    returned value is finite (NaN/inf collapse to 0).  Silent windows (zero
    total power) yield all-zero features so silent chunks never masquerade as
    spectrally structured ones.
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 2 or sr <= 0:
        return (0.0,) * FEATURE_DIM

    spectrum = np.fft.rfft(x * np.hanning(n))
    amps = np.abs(spectrum) * 2.0 / n
    amps[0] = abs(spectrum[0]) / n          # DC bin must not be doubled
    if float(amps.sum()) <= 0.0:            # digital silence
        return (0.0,) * FEATURE_DIM
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    nyquist = sr / 2.0
    rms_all = float(np.sqrt(np.mean(x * x)))
    if rms_all <= 0.0:
        return (0.0,) * FEATURE_DIM

    features: list[float] = []
    for low, high in BANDS:
        band = amps[band_indices(freqs, low, high)]
        if band.size == 0:
            features.extend([0.0] * BAND_STATS)
            continue
        mean_amp = float(band.mean()) / rms_all
        std_amp = float(band.std()) / rms_all
        _, _, skew, kurt = moment_stats(band)
        band_rms = float(np.sqrt(np.mean(band * band)))
        rel_rms = band_rms / rms_all
        crest = float(band.max()) / band_rms if band_rms > 0.0 else 0.0
        features.extend([mean_amp, std_amp, skew, kurt, rel_rms, crest])

    total = float(amps.sum())
    dominant = float(freqs[int(np.argmax(amps))])
    centroid = float(np.dot(freqs, amps)) / total
    n_bins = max(2, amps.size)
    entropy_rel = shannon_entropy(amps ** 2 / (amps ** 2).sum()) / float(
        np.log2(n_bins))
    hurst = hurst_exponent(np.log10(amps + _LOG_FLOOR))
    features.extend([dominant / nyquist, centroid / nyquist,
                     entropy_rel, hurst])

    return tuple(float(v) if np.isfinite(v) else 0.0 for v in features)


def compress_and_normalize(raw: np.ndarray) -> np.ndarray:
    """Clip pathological values and L2-normalize the dimensionless vector.

    The spectral statistics are already scale-free ratios (amplitudes
    relative to the window RMS, frequencies relative to Nyquist, entropy
    relative to the bin count), so no log compression is required; only a
    safety clip of skew/kurtosis/crest outliers and the unit-norm scaling
    that keeps cosine comparisons comparable with the CLAP/MERT vectors.
    """
    vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vec.size != FEATURE_DIM:             # defensive: never emit ragged vectors
        vec = np.resize(vec, FEATURE_DIM)
    vec = np.clip(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0),
                  -COMPRESS_CLIP, COMPRESS_CLIP)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return vec.astype(np.float32)


class FftPlugin(ModelPlugin):
    """FFT band-statistics feature vectors (dependency-light analysis model)."""

    name: ClassVar[str] = "fft"
    display_name: ClassVar[str] = "FFT"
    embedding_dim: ClassVar[int | None] = FEATURE_DIM
    provides_text: ClassVar[bool] = False
    #: Informational only: the plugin analyzes audio at its decoded sample
    #: rate (no resampling); bands above Nyquist simply come out empty.
    preferred_sample_rate: ClassVar[int] = 48000
    #: numpy ships with the app; the plugin is always available.
    requirements: ClassVar[tuple[str, ...]] = ("numpy",)

    BATCH_SIZE = 8

    def __init__(self) -> None:
        super().__init__()
        # Editable setting (module default 10 s; apply_config() overrides it
        # from the persisted AppConfig).
        self.window_sec: float = 10.0

    # ---- settings -----------------------------------------------------------
    def apply_config(self, config: object) -> None:
        """Adopt the window length from an ``AppConfig`` (``fft_window_sec``).

        Tolerant of partially-built configs and test fakes; values below
        :data:`MIN_WINDOW_SEC` are clamped.
        """
        try:
            self.window_sec = max(
                MIN_WINDOW_SEC,
                float(getattr(config, "fft_window_sec", self.window_sec)))
        except (TypeError, ValueError):
            pass

    # ---- inference ----------------------------------------------------------
    def _embed(self, chunks: AudioChunks, sr: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for chunk in chunks:
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
            out.append(self._chunk_vector(samples, int(sr)))
        return out

    def _split_chunk_windows(self, samples: np.ndarray,
                             sr: int) -> list[np.ndarray]:
        """Contiguous non-overlapping windows of ``window_sec`` seconds.

        Unlike the MERT windowing (which shifts the final window back to keep
        it full), FFT statistics need every sample counted exactly once: the
        final window may be shorter and stays where it is.  Windows shorter
        than two samples cannot be transformed and are dropped.
        """
        window_len = max(2, int(round(self.window_sec * sr)))
        if samples.size <= window_len:
            return [samples]
        return [samples[start:start + window_len]
                for start in range(0, samples.size, window_len)
                if samples.size - start >= 2]

    def _chunk_vector(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """One chunk -> one feature vector (mean over windows, then compress).

        Each window yields :func:`spectral_features`; the per-window feature
        vectors are averaged into the chunk's raw statistics, then clipped
        and L2-normalized so cosine similarity behaves like it does for the
        CLAP/MERT vectors.
        """
        if samples.size == 0 or sr <= 0:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        windows = self._split_chunk_windows(samples, sr)
        raw = (np.mean([spectral_features(window, sr) for window in windows],
                       axis=0) if windows else np.zeros(FEATURE_DIM))
        return compress_and_normalize(raw)

    def raw_chunk_vector(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Raw (unnormalized) per-chunk feature vector — exposed for tests
        and inspection: the exact statistics before range-clipping."""
        if samples.size == 0 or sr <= 0:
            return np.zeros(FEATURE_DIM, dtype=np.float64)
        windows = self._split_chunk_windows(samples, sr)
        if not windows:
            return np.zeros(FEATURE_DIM, dtype=np.float64)
        return np.asarray(
            np.mean([spectral_features(w, sr) for w in windows], axis=0),
            dtype=np.float64)