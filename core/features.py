"""Chiron audio front-end -- THE single source of truth.

Every stage here must be portable to CMSIS-DSP fixed point on the EFR32MG26.
That constraint drives three rules, and breaking any of them silently destroys
on-device accuracy (see docs/06_MODEL_DESIGN.md section 2.2):

  1. The band-pass is a CAUSAL cascaded-biquad (SOS) filter applied with
     ``sosfilt``.  NEVER ``sosfiltfilt`` -- zero-phase filtering is non-causal
     and has no real-time equivalent.  The same SOS coefficients are exported
     to ``arm_biquad_cascade_df1_q15``.
  2. Framing is explicit and non-centred (no implicit padding), matching a
     ring-buffer walked one hop at a time in C.
  3. Nothing depends on librosa at inference time; the mel filterbank is built
     once and exported as a constant table.

KAUH audio is natively 4 kHz / int16 / mono, so the lung path runs at 4 kHz
with no resampling: Nyquist (2000 Hz) already matches the analysis band, and
it halves the on-device sample count versus upsampling to 8 kHz.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.io import wavfile
from math import gcd

from scipy.signal import butter, sosfilt, resample_poly
from scipy.fftpack import dct as _dct


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Front-end parameters. Mirrored by ``chiron_dsp_config.h`` on device."""

    name: str = "pulmo"
    sample_rate: int = 4000        # KAUH native rate; no resampling
    window_s: float = 5.0
    band_low_hz: float = 80.0
    band_high_hz: float = 1900.0   # below the 2000 Hz Nyquist, with margin
    filter_order: int = 4          # -> 2 biquad sections
    n_fft: int = 256               # 64 ms at 4 kHz
    hop: int = 128                 # 32 ms
    n_mels: int = 64
    n_frames: int = 160            # frames produced (155) padded to 160
    log_floor: float = 1e-6
    feature: str = "logmel"        # "logmel" | "mfcc"
    n_mfcc: int = 13               # used when feature == "mfcc"

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.sample_rate))

    @property
    def n_frames_raw(self) -> int:
        return 1 + (self.window_samples - self.n_fft) // self.hop

    @property
    def n_feature_rows(self) -> int:
        return self.n_mfcc if self.feature == "mfcc" else self.n_mels

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_feature_rows, self.n_frames)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            window_samples=self.window_samples,
            n_frames_raw=self.n_frames_raw,
            shape=list(self.shape),
        )
        return d


PULMO_CONFIG = FeatureConfig(name="pulmo")

# Yaseen 2018 heart sounds are natively 8 kHz mono, 1.16-3.99 s per clip.
# Murmur energy sits roughly 20-600 Hz, so the analysis band is lower and the
# frame is shorter than the lung path (heart events are far more transient).
CARDIO_CONFIG = FeatureConfig(
    name="cardio",
    sample_rate=8000,
    window_s=3.0,
    band_low_hz=20.0,
    band_high_hz=800.0,
    n_fft=256,        # 32 ms at 8 kHz
    hop=128,          # 16 ms
    n_mels=40,
    n_frames=192,     # 186 raw, padded
)


# ---------------------------------------------------------------------------
# Filterbank / filter construction (constants, computed once)
# ---------------------------------------------------------------------------
def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig) -> np.ndarray:
    """Slaney-style triangular mel filterbank, shape (n_mels, n_fft//2 + 1).

    Written out explicitly rather than calling librosa so the identical table
    can be dumped to a C header.
    """
    n_bins = cfg.n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, cfg.sample_rate / 2.0, n_bins)

    mel_pts = np.linspace(
        hz_to_mel(cfg.band_low_hz), hz_to_mel(cfg.band_high_hz), cfg.n_mels + 2
    )
    hz_pts = mel_to_hz(mel_pts)

    fb = np.zeros((cfg.n_mels, n_bins), dtype=np.float64)
    for i in range(cfg.n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
    # Area-normalise so wide high-frequency filters do not dominate.
    enorm = 2.0 / (hz_pts[2 : cfg.n_mels + 2] - hz_pts[: cfg.n_mels])
    fb *= enorm[:, None]
    return fb


def bandpass_sos(cfg: FeatureConfig) -> np.ndarray:
    """Causal Butterworth band-pass as second-order sections."""
    nyq = cfg.sample_rate / 2.0
    return butter(
        cfg.filter_order // 2,
        [cfg.band_low_hz / nyq, cfg.band_high_hz / nyq],
        btype="bandpass",
        output="sos",
    )


def hamming(n: int) -> np.ndarray:
    """Symmetric Hamming window, matching ``arm_hamming_f32``/our C table."""
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


class _Tables:
    """Lazily built, cached constant tables keyed by config."""

    _cache: dict = {}

    @classmethod
    def get(cls, cfg: FeatureConfig):
        key = (cfg.sample_rate, cfg.n_fft, cfg.n_mels,
               cfg.band_low_hz, cfg.band_high_hz, cfg.filter_order)
        if key not in cls._cache:
            cls._cache[key] = {
                "sos": bandpass_sos(cfg),
                "melfb": mel_filterbank(cfg),
                "win": hamming(cfg.n_fft),
            }
        return cls._cache[key]


# ---------------------------------------------------------------------------
# Signal path
# ---------------------------------------------------------------------------
def load_wav(path: str, cfg: FeatureConfig = PULMO_CONFIG,
             allow_resample: bool = True) -> np.ndarray:
    """Load a wav as float64 in [-1, 1), resampling to cfg.sample_rate.

    KAUH and Yaseen are already at their target rates, so resampling is a no-op
    there. ICBHI is heterogeneous (44.1 kHz / 10 kHz / 4 kHz across four
    recording devices) and MUST be brought to one rate before feature
    extraction, or sample rate becomes a device fingerprint the model can learn.

    ``resample_poly`` is used rather than naive decimation because it applies
    the required anti-aliasing filter.
    """
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    else:
        x = data.astype(np.float64)

    if sr != cfg.sample_rate:
        if not allow_resample:
            raise ValueError(f"{path}: sample rate {sr} != expected {cfg.sample_rate}")
        g = gcd(int(sr), int(cfg.sample_rate))
        x = resample_poly(x, int(cfg.sample_rate) // g, int(sr) // g)
    return x


def preprocess(x: np.ndarray, cfg: FeatureConfig = PULMO_CONFIG) -> np.ndarray:
    """DC removal -> causal band-pass -> per-recording RMS normalisation.

    RMS normalisation is what removes absolute device gain as a class cue.
    """
    t = _Tables.get(cfg)
    x = x - np.mean(x)
    x = sosfilt(t["sos"], x)          # CAUSAL. Not sosfiltfilt. See module docstring.
    rms = np.sqrt(np.mean(x**2))
    if rms > 1e-9:
        x = x / rms * 0.1             # target RMS 0.1 leaves headroom for Q15
    return x


def frame_signal(x: np.ndarray, cfg: FeatureConfig = PULMO_CONFIG) -> np.ndarray:
    """Explicit non-centred framing -> (n_frames_raw, n_fft)."""
    n = cfg.n_frames_raw
    idx = np.arange(cfg.n_fft)[None, :] + cfg.hop * np.arange(n)[:, None]
    return x[idx]


def log_mel(x: np.ndarray, cfg: FeatureConfig = PULMO_CONFIG) -> np.ndarray:
    """Windowed signal -> log-mel matrix, shape (n_mels, n_frames)."""
    t = _Tables.get(cfg)
    frames = frame_signal(x, cfg) * t["win"][None, :]
    spec = np.abs(np.fft.rfft(frames, n=cfg.n_fft, axis=1))       # magnitude
    mel = spec @ t["melfb"].T                                     # (frames, n_mels)
    out = np.log(mel + cfg.log_floor).T                           # (n_mels, frames)
    return _pad_frames(out, cfg, fill=float(np.log(cfg.log_floor)))


def _pad_frames(m: np.ndarray, cfg: FeatureConfig, fill: float) -> np.ndarray:
    """Pad/crop the time axis to exactly cfg.n_frames."""
    have = m.shape[1]
    if have == cfg.n_frames:
        return m
    if have > cfg.n_frames:
        start = (have - cfg.n_frames) // 2
        return m[:, start : start + cfg.n_frames]
    pad = np.full((m.shape[0], cfg.n_frames - have), fill, dtype=m.dtype)
    return np.concatenate([m, pad], axis=1)


def mfcc(x: np.ndarray, cfg: FeatureConfig = PULMO_CONFIG) -> np.ndarray:
    """MFCC via DCT-II of the log-mel matrix, shape (n_mfcc, n_frames).

    Included because Sen et al. (arXiv:2606.10972) found MFCC-13 outperformed
    log-mel spectrograms for asthma-vs-COPD discrimination. On device this is
    one extra (n_mfcc x n_mels) matrix multiply -- cheap.
    """
    lm = log_mel(x, cfg)
    coeffs = _dct(lm, type=2, axis=0, norm="ortho")[: cfg.n_mfcc]
    return coeffs


def extract(x: np.ndarray, cfg: FeatureConfig = PULMO_CONFIG) -> np.ndarray:
    """Full front-end on one already-preprocessed window."""
    return mfcc(x, cfg) if cfg.feature == "mfcc" else log_mel(x, cfg)


# ---------------------------------------------------------------------------
# Windowing a whole recording
# ---------------------------------------------------------------------------
def iter_windows(
    x: np.ndarray,
    cfg: FeatureConfig = PULMO_CONFIG,
    overlap: float = 0.5,
    min_tail_frac: float = 0.5,
):
    """Yield fixed-length analysis windows from a preprocessed recording.

    A trailing partial window is kept (zero-padded) only if it is at least
    ``min_tail_frac`` of a full window, so we never fabricate mostly-silent
    examples.
    """
    w = cfg.window_samples
    step = max(1, int(round(w * (1.0 - overlap))))
    if len(x) < w:
        if len(x) >= min_tail_frac * w:
            yield np.pad(x, (0, w - len(x)))
        return
    start = 0
    while start + w <= len(x):
        yield x[start : start + w]
        start += step
    tail = len(x) - start
    if tail >= min_tail_frac * w:
        yield np.pad(x[start:], (0, w - tail))


def features_for_file(
    path: str, cfg: FeatureConfig = PULMO_CONFIG, overlap: float = 0.5
) -> np.ndarray:
    """wav path -> (n_windows, rows, frames) feature stack."""
    x = preprocess(load_wav(path, cfg), cfg)
    feats = [extract(w, cfg) for w in iter_windows(x, cfg, overlap)]
    if not feats:
        return np.empty((0, *cfg.shape))
    return np.stack(feats).astype(np.float32)


if __name__ == "__main__":
    cfg = PULMO_CONFIG
    print("Chiron pulmo front-end")
    for k, v in cfg.to_dict().items():
        print(f"  {k:16s} {v}")
    t = _Tables.get(cfg)
    print(f"  sos sections     {t['sos'].shape[0]}")
    print(f"  melfb            {t['melfb'].shape}")
    print(f"  frames raw->pad  {cfg.n_frames_raw} -> {cfg.n_frames}")
