"""The heart result for the dashboard (roadmap S5; frontend guide, Section 3).

One call takes a heart recording and the patient's details and returns the objects
of core/contracts.py, so the dashboard never touches the model or its weights:

    from cardio.predict import analyse_heart
    from core.contracts import PatientMetadata

    result = analyse_heart(uploaded_file, PatientMetadata(sex="female", weight_kg=54))
    if result.withheld:              # "unreadable", "too_short" or "quality_check"
        ...                          # say why, and show no heart result
    result.model_output              # ModelOutput: calibrated class probabilities, 17 characteristics
    result.fused                     # FusedOutput: p_h, each vote and its weight
    result.systole_or_diastole       # "systole", "diastole", "neither", or None if not measured

The recording can be a path, raw bytes, or a file-like object such as a Streamlit
upload. Nothing is written to disk.

Stages, each timed in milliseconds; the edge budget reports these times:

  load_and_resample          any WAV, channel 0, converted to 8 kHz
  quality_check              at least 1.5 s of sound, which gives one 3-second window,
                             and the energy test of core/ood.py; a failure withholds
                             the heart result
  front_end                  band-pass, 3-second windows with 50% overlap, log-mel
  cnn_8bit                   the shipped 8-bit model on every window, the mean
                             softmax, then temperature T
  secondary_characteristics  the 17 numbers of cardio.model.features
  systole_or_diastole        which phase holds the extra sound, from the measured
                             systolic and diastolic energy (the rule audio_descriptors
                             uses for murmur_phase)
  votes_and_fusion           heart_fusion() from cardio/fusion/late_fusion.py once it
                             and reports/heart_fusion_params.json exist. Until then the
                             CNN vote counts alone with weight 1: the proposal's
                             missing-vote rule

Temperature T is read from reports/heart_fusion_params.json, key "T", and failing
that from reports/heart_calibration.json, key "T". Until either exists T = 1 and
result.temperature is None, so the dashboard can say the probabilities are not yet
calibrated. heart_fusion(p_cnn, features, metadata) receives p_cnn already
calibrated, as 1 - p(Normal) after T, and the 17 characteristics as a dict.

Systole or diastole is measured from the sound, not by occluding the model's input:
in v1 occlusion pointed to systole for every class, Normal included
(core/descriptors.py, audio_phase_attribution).

Usage (public recordings only, never a clinic one):
    python -m cardio.predict data/raw/circor/training_data/13918_AV.wav
"""

from __future__ import annotations

import argparse
import io
import json
import math
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from cardio.model.final_model import CLASSES, REPO_ROOT, FinalModel
from core.contracts import FusedOutput, ModelOutput, PatientMetadata
from core.features import CARDIO_CONFIG, extract, iter_windows, load_wav, preprocess
from core.ood import signal_provenance

REPORTS = REPO_ROOT / "reports"
FUSION_PARAMS = REPORTS / "heart_fusion_params.json"
CALIBRATION = REPORTS / "heart_calibration.json"
FUSION_CODE = REPO_ROOT / "cardio" / "fusion" / "late_fusion.py"
FS = CARDIO_CONFIG.sample_rate
MIN_SAMPLES = CARDIO_CONFIG.window_samples // 2   # 1.5 s: iter_windows keeps a half window
CLOSE_CALL = 0.05                                 # top two classes closer than this
STAGES = ("load_and_resample", "quality_check", "front_end", "cnn_8bit",
          "secondary_characteristics", "systole_or_diastole", "votes_and_fusion")

_LOCK = threading.Lock()        # one 8-bit interpreter, shared by the dashboard's sessions


@dataclass
class HeartResult:
    """Everything the dashboard shows for one heart recording."""

    withheld: str | None = None                 # None, "unreadable", "too_short" or "quality_check"
    seconds_of_audio: float = 0.0
    quality: dict = field(default_factory=dict)  # core.ood: in_distribution, hi_band, threshold
    windows: int = 0
    temperature: float | None = None            # None: not calibrated yet, T = 1 was used
    model_output: ModelOutput | None = None
    fused: FusedOutput | None = None
    fusion: str = "cnn_only"                    # "heart_fusion" once it is in the branch
    characteristics: dict | None = None         # all 17, None where not measured
    systole_or_diastole: str | None = None
    timings_ms: dict = field(default_factory=dict)
    audio_8k: np.ndarray | None = field(default=None, repr=False)   # for the waveform view

    @property
    def predicted_class(self) -> str | None:
        if self.model_output is None:
            return None
        p = self.model_output.probabilities
        return max(p, key=p.get)

    @property
    def p_h(self) -> float | None:
        """The fused probability that the heart sound is abnormal."""
        return None if self.fused is None else self.fused.p_abnormal

    @property
    def close_call(self) -> tuple[str, str] | None:
        """The top two classes, when they are within CLOSE_CALL of each other."""
        if self.model_output is None:
            return None
        (a, pa), (b, pb) = sorted(self.model_output.probabilities.items(), key=lambda kv: -kv[1])[:2]
        return (a, b) if pa - pb < CLOSE_CALL else None

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.get(s, 0.0) for s in STAGES))

    def pcm16(self) -> bytes | None:
        """The 8 kHz audio as 16-bit PCM, the input core.spectrogram.spectrogram_png takes."""
        if self.audio_8k is None:
            return None
        return (np.clip(self.audio_8k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


@lru_cache(maxsize=1)
def load_model() -> FinalModel:
    """The shipped model, loaded once per process."""
    return FinalModel()


def temperature() -> float | None:
    """T from the fusion parameters, else from the calibration report; None before either exists."""
    for path in (FUSION_PARAMS, CALIBRATION):
        if not path.exists():
            continue
        T = json.loads(path.read_text(encoding="utf-8")).get("T")
        if T is None:
            continue
        T = float(T)
        if not (math.isfinite(T) and T > 0):
            raise ValueError(f"{path.name}: T must be a positive number, got {T}")
        return T
    return None


def calibrate(p, T: float | None) -> np.ndarray:
    """Temperature scaling on log-probabilities, softmax(log p / T); T = None leaves p as it is."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    p = p / p.sum()
    if T is None:
        return p
    z = np.log(p) / T
    z = np.exp(z - z.max())
    return z / z.sum()


def fusion_function():
    """heart_fusion(p_cnn, features, metadata), once it and its parameter file exist; else None."""
    if not (FUSION_CODE.exists() and FUSION_PARAMS.exists()):
        return None
    from cardio.fusion import late_fusion          # errors inside it are real errors: let them show

    return getattr(late_fusion, "heart_fusion", None)


def systole_or_diastole(ch: dict) -> str | None:
    """Which phase holds the extra sound, by the rule audio_descriptors uses for murmur_phase.

    None when the heart cycles could not be found; "neither" when no murmur was measured.
    """
    if not ch.get("segmentation_ok"):
        return None
    if not ch.get("murmur_detected"):
        return "neither"
    s, d = ch.get("systolic_ratio") or 0.0, ch.get("diastolic_ratio") or 0.0
    return "systole" if s >= d else "diastole"


def _wav_source(recording):
    """A path, or an in-memory file that scipy's WAV reader accepts."""
    if isinstance(recording, (str, Path)):
        return str(recording)
    if isinstance(recording, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(recording))
    if hasattr(recording, "getvalue"):            # io.BytesIO, Streamlit's UploadedFile
        return io.BytesIO(recording.getvalue())
    if hasattr(recording, "read"):
        if hasattr(recording, "seek"):
            recording.seek(0)
        return io.BytesIO(recording.read())
    raise TypeError(f"expected a path, bytes or a file-like object, got {type(recording).__name__}")


def analyse_heart(recording, metadata: PatientMetadata | None = None, *,
                  model: FinalModel | None = None, complete: bool = False) -> HeartResult:
    """Run one recording through every stage and return what the dashboard shows.

    complete=True carries on after a failed quality check, so the edge budget can time
    a whole case; the result is still marked as withheld.
    """
    from cardio.model.features import characteristics_from_audio
    from core.quantize import tflite_predict

    res, clock = HeartResult(), time.perf_counter
    t = res.timings_ms

    t0 = clock()
    try:
        x8 = load_wav(_wav_source(recording), CARDIO_CONFIG)
    except (ValueError, EOFError):                 # not a WAV file, or a damaged one
        res.withheld = "unreadable"
        return res
    t["load_and_resample"] = (clock() - t0) * 1000
    res.audio_8k, res.seconds_of_audio = x8, len(x8) / FS

    t0 = clock()
    res.quality = signal_provenance(res.pcm16())
    if len(x8) < MIN_SAMPLES:
        res.withheld = "too_short"
    elif res.quality["in_distribution"] is not True:
        res.withheld = "quality_check"
    t["quality_check"] = (clock() - t0) * 1000
    if res.withheld == "too_short" or (res.withheld and not complete):
        return res

    t0 = clock()
    windows = [extract(w, CARDIO_CONFIG) for w in iter_windows(preprocess(x8, CARDIO_CONFIG), CARDIO_CONFIG, 0.5)]
    feats = np.stack(windows).astype(np.float32)   # exactly as core.features.features_for_file
    t["front_end"] = (clock() - t0) * 1000
    res.windows = len(feats)

    model = model or load_model()
    t0 = clock()
    with _LOCK:
        p_windows = tflite_predict(model.interp, model.normalise(feats))
    res.temperature = temperature()
    probs = calibrate(p_windows.mean(axis=0), res.temperature)
    t["cnn_8bit"] = (clock() - t0) * 1000

    t0 = clock()
    ch = characteristics_from_audio(x8)
    t["secondary_characteristics"] = (clock() - t0) * 1000
    res.characteristics = ch

    t0 = clock()
    res.systole_or_diastole = systole_or_diastole(ch)
    t["systole_or_diastole"] = (clock() - t0) * 1000

    res.model_output = ModelOutput(
        "heart", {c: float(p) for c, p in zip(CLASSES, probs)},
        secondary={k: float(v) for k, v in ch.items() if v is not None and math.isfinite(v)},
        quality_ok=res.withheld is None)

    t0 = clock()
    p_cnn = float(np.clip(1.0 - res.model_output.probabilities["Normal"], 0.0, 1.0))
    fuse = fusion_function()
    if fuse is None:
        res.fused = FusedOutput("heart", p_abnormal=p_cnn, votes={"cnn": p_cnn}, weights={"cnn": 1.0})
    else:
        res.fused, res.fusion = fuse(p_cnn, ch, metadata), "heart_fusion"
        if not (isinstance(res.fused, FusedOutput) and res.fused.modality == "heart"):
            raise TypeError("heart_fusion must return FusedOutput('heart', ...)")
    t["votes_and_fusion"] = (clock() - t0) * 1000
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="The heart result for one recording, as the dashboard shows it.")
    ap.add_argument("recording", type=Path, help="a WAV file at any sample rate (public data only)")
    args = ap.parse_args()

    res = analyse_heart(args.recording)
    if res.withheld == "unreadable":
        print("heart result withheld: not a readable WAV file")
        return 0
    print(f"{res.seconds_of_audio:.1f} s of sound; energy above 1 kHz {res.quality.get('hi_band')} "
          f"(limit {res.quality.get('threshold')})")
    if res.withheld:
        print(f"heart result withheld: {res.withheld}")
        return 0
    print(f"{res.windows} windows of 3 s")
    probs = res.model_output.probabilities
    print("class probabilities:", ", ".join(f"{c} {probs[c]:.3f}" for c in CLASSES),
          f"(T = {res.temperature})" if res.temperature else "(not calibrated yet)")
    if res.close_call:
        print("uncertain between %s and %s" % res.close_call)
    print(f"predicted {res.predicted_class}; p_h = {res.p_h:.3f}; votes {res.fused.votes}, "
          f"weights {res.fused.weights} ({res.fusion})")
    print(f"systole or diastole: {res.systole_or_diastole}")
    print("ms per stage:", ", ".join(f"{s} {res.timings_ms[s]:.1f}" for s in STAGES),
          f"| total {res.total_ms:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
