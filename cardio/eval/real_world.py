"""First look at the clinic recordings, without labels (roadmap S2, proposal 4.6).

The clinic recordings are 50 files: 10 patients x 5 chest positions, 5 s each,
48 kHz, captured with a MEMS microphone. They carry no diagnosis and no patient
details, so nothing here can be an accuracy. What can be measured is how the
Yaseen-trained model behaves on a new device:

  * how many recordings pass the quality check (core/ood.py)
  * what the model predicts, overall and by chest position
  * whether the 5 positions of the same patient agree on refer-or-not
  * whether heart-cycle segmentation works and gives a plausible heart rate
  * whether the 8-bit model agrees with the float model on this data

Data rules (from the folder's README): the recordings never leave the laptop
that holds them, and only totals and averages may be reported. This script
therefore takes the folder as an argument, reads it in place, and writes
nothing per recording or per patient: the output file holds aggregates only.

Usage:
    python -m core.quantize --dataset yaseen --holdout-fold 0   # final model, once
    python -m cardio.eval.real_world --data-dir <path to Real_World_Data>

Writes:
    reports/heart_real_world.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from cardio.model.final_model import CLASSES, FinalModel
from core.descriptors import audio_descriptors
from core.features import CARDIO_CONFIG, load_wav
from core.ood import signal_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r"^(p\d+)_(aortic_area|pulmonic_area|tricuspid_area|mitral_area|erbs_point)\.wav$")
POSITIONS = ["aortic_area", "pulmonic_area", "tricuspid_area", "mitral_area", "erbs_point"]
HR_RANGE = (40.0, 180.0)                    # beats per minute counted as plausible


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def measure(path: Path, model: FinalModel) -> dict:
    """Everything this analysis needs from one recording. Kept in memory only."""
    x8 = load_wav(str(path), CARDIO_CONFIG)              # channel 0, resampled to 8 kHz
    pcm = to_pcm16(x8)
    gate = signal_provenance(pcm.tobytes())
    desc = audio_descriptors(pcm, fs=CARDIO_CONFIG.sample_rate)
    hr = None
    if desc.get("segmentation_ok") and desc.get("systole_ms") and desc.get("diastole_ms"):
        hr = 60000.0 / (desc["systole_ms"] + desc["diastole_ms"])   # one cycle = systole + diastole
    p_float, p_int8 = model.predict_file(str(path))
    _, raw = wavfile.read(str(path))
    raw0 = raw[:, 0] if raw.ndim > 1 else raw
    return {
        "passes_check": bool(gate["in_distribution"]),
        "segmented": bool(desc.get("segmentation_ok")),
        "heart_rate": hr,
        "p_float": p_float, "p_int8": p_int8,
        "peak": float(np.max(np.abs(raw0.astype(np.float64)))),
    }


def summarise(rows: list[dict]) -> dict:
    """Totals and averages over a set of recordings."""
    if not rows:
        return {"recordings": 0}
    p = np.array([r["p_float"] for r in rows])
    abnormal = (1.0 - p[:, 0]) > 0.5
    pred = p.argmax(1)
    hr = [r["heart_rate"] for r in rows if r["heart_rate"] is not None]
    by_pos = {}
    for pos in POSITIONS:
        m = np.array([r["position"] == pos for r in rows])
        if m.any():
            by_pos[pos] = {"recordings": int(m.sum()), "share_called_abnormal": float(abnormal[m].mean())}
    return {
        "recordings": len(rows),
        "predicted_class_counts": {c: int(n) for c, n in zip(CLASSES, np.bincount(pred, minlength=len(CLASSES)))},
        "share_called_abnormal": float(abnormal.mean()),
        "mean_p_abnormal": float((1.0 - p[:, 0]).mean()),
        "mean_top_probability": float(p.max(1).mean()),
        "share_called_abnormal_by_position": by_pos,
        "share_segmented": float(np.mean([r["segmented"] for r in rows])),
        "share_heart_rate_plausible": float(np.mean(
            [r["heart_rate"] is not None and HR_RANGE[0] <= r["heart_rate"] <= HR_RANGE[1] for r in rows])),
        "median_heart_rate_bpm": float(np.median(hr)) if hr else None,
        "int8_float_agreement": float(np.mean(
            [np.argmax(r["p_int8"]) == np.argmax(r["p_float"]) for r in rows])),
    }


def position_agreement(rows: list[dict]) -> dict:
    """Do the 5 positions of one patient agree on refer-or-not? (aggregates only)"""
    by_patient: dict[str, list[bool]] = {}
    for r in rows:
        by_patient.setdefault(r["patient"], []).append((1.0 - r["p_float"][0]) > 0.5)
    all_agree, majority_share = [], []
    for calls in by_patient.values():
        n_abn = sum(calls)
        all_agree.append(n_abn in (0, len(calls)))
        majority_share.append(max(n_abn, len(calls) - n_abn) / len(calls))
    return {
        "patients": len(by_patient),
        "share_of_patients_all_positions_agree": float(np.mean(all_agree)),
        "mean_share_of_positions_with_patient_majority": float(np.mean(majority_share)),
    }


def shift_against_yaseen(files: list[Path], model: FinalModel) -> dict:
    """How far the clinic input sits from Yaseen fold 0 (which the final model never
    trained on), in the model's own normalised input units. Aggregates only."""
    import pandas as pd
    from core.features import features_for_file

    cache = np.load(REPO_ROOT / "data" / "cache" / "yaseen_logmel.npz", allow_pickle=True)
    z_ref = model.normalise(cache["X"][cache["fold"] == 0])
    z_new = model.normalise(np.concatenate([features_for_file(str(f), CARDIO_CONFIG) for f in files]))
    ref = pd.read_csv(REPO_ROOT / "data" / "manifests" / "yaseen_manifest.csv").query("fold == 0")
    seg_ref = [audio_descriptors(to_pcm16(load_wav(str(REPO_ROOT / p), CARDIO_CONFIG))).get("segmentation_ok")
               for p in ref.path]
    band_shift = (z_new.mean(axis=(0, 2, 3)) - z_ref.mean(axis=(0, 2, 3)))
    return {
        "normalised_input_mean": {"yaseen_fold0": float(z_ref.mean()), "clinic": float(z_new.mean())},
        "normalised_input_std": {"yaseen_fold0": float(z_ref.std()), "clinic": float(z_new.std())},
        "largest_mean_shift_in_lowest_10_mel_bands": float(np.abs(band_shift[:10]).max()),
        "largest_mean_shift_in_other_mel_bands": float(np.abs(band_shift[10:]).max()),
        "share_segmented_yaseen_fold0": float(np.mean(seg_ref)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="the Real_World_Data folder, read in place and never copied")
    args = ap.parse_args()

    files = sorted(f for f in args.data_dir.glob("p*_*.wav") if NAME.match(f.name))
    if len(files) != 50:
        print(f"note: expected 50 recordings, found {len(files)}")
    model = FinalModel()

    rows = []
    for f in files:
        patient, position = NAME.match(f.name).groups()
        r = measure(f, model)
        r.update(patient=patient, position=position)
        rows.append(r)

    peaks = np.array([r["peak"] for r in rows])
    passing = [r for r in rows if r["passes_check"]]
    out = {
        "what": "Clinic recordings (10 patients x 5 positions), measured without labels with the "
                "Yaseen-trained heart CNN. Aggregates only, per the data's handling rules.",
        "run_on": dt.date.today().isoformat(),
        "model": model.files,
        "normalisation": "Yaseen training folds (mean %.4f, std %.4f)" % (model.mu, model.sd),
        "recordings": len(rows),
        "patients": len({r["patient"] for r in rows}),
        "quality_check": {"passed": len(passing), "failed": len(rows) - len(passing),
                          "share_passed": len(passing) / len(rows)},
        "recordings_with_handling_spike": int(np.sum(peaks > 2.0 * np.median(peaks))),
        "all_recordings": summarise(rows),
        "recordings_passing_quality_check": summarise(passing),
        "position_agreement_all_recordings": position_agreement(rows),
        "shift_against_yaseen": shift_against_yaseen(files, model),
        "metadata": "None recorded for these patients, so the metadata vote is missing; late fusion "
                    "drops it and divides its weight between the CNN and feature votes (proposal 4.2).",
        "heart_rate_method": "60000 / (systole_ms + diastole_ms) from core.descriptors segmentation; "
                             "plausible means 40 to 180 bpm",
        "refer_or_not_rule": "p(abnormal) = 1 - p(Normal) > 0.5, uncalibrated float model",
    }
    path = REPO_ROOT / "reports" / "heart_real_world.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))

    a = out["all_recordings"]
    print(f"{out['recordings']} recordings from {out['patients']} patients")
    print(f"quality check passed: {out['quality_check']['passed']} of {out['recordings']}")
    print(f"called abnormal: {a['share_called_abnormal']:.0%}   predicted classes: {a['predicted_class_counts']}")
    print(f"patients whose 5 positions all agree: "
          f"{out['position_agreement_all_recordings']['share_of_patients_all_positions_agree']:.0%}")
    print(f"segmented: {a['share_segmented']:.0%}   plausible heart rate: {a['share_heart_rate_plausible']:.0%}")
    print(f"8-bit vs float agreement: {a['int8_float_agreement']:.0%}")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
