"""Does the 8-bit heart model give the same answer as the float model?
(roadmap S2, proposal 4.5 and 4.7)

Agreement has a known correct answer, which accuracy on new data does not: the
8-bit model should predict the same class as the float model on identical input.
Measured on Yaseen fold 0, the fold the final model never trained on, at window
and clip level; and on PhysioNet 2022 or BMD-HS through their manifests, which
need no labels for this. Every agreement carries a 95% bootstrap interval over
patients (cardio.eval.intervals, built on cardio.eval.metrics).

Usage:
    python -m cardio.eval.agreement
    python -m cardio.data.build_unseen --dataset circor          # manifest and caches, once
    python -m cardio.eval.agreement --manifest data/manifests/circor_manifest.csv --name physionet_2022
    python -m cardio.eval.agreement --wav-dir <folder> --name <label>   # any folder, by file

Writes (merging with earlier runs):
    reports/heart_agreement.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cardio.eval.intervals import agreement_interval
from cardio.model.final_model import FinalModel
from core.features import CARDIO_CONFIG, features_for_file

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "reports" / "heart_agreement.json"


def agreement(p_float: np.ndarray, p_int8: np.ndarray) -> float:
    return float(np.mean(p_float.argmax(1) == p_int8.argmax(1)))


def yaseen_fold0(model: FinalModel) -> dict:
    cache = np.load(REPO_ROOT / "data" / "cache" / "yaseen_logmel.npz", allow_pickle=True)
    te = cache["fold"] == 0
    p_float, p_int8 = model.predict_windows(cache["X"][te])
    pid = cache["patient_id"][te]
    clip_f = pd.DataFrame(p_float).groupby(pid, sort=False).mean().to_numpy()
    clip_q = pd.DataFrame(p_int8).groupby(pid, sort=False).mean().to_numpy()
    clips = pd.Series(pid).drop_duplicates().to_numpy()
    y = pd.Series(cache["y"][te]).groupby(pid, sort=False).first().to_numpy()
    return {
        "windows": int(te.sum()), "clips": int(len(clip_f)),
        "window_agreement": agreement(p_float, p_int8),
        "clip_agreement": agreement(clip_f, clip_q),
        "clip_agreement_interval": agreement_interval(clip_q.argmax(1), clip_f, clips),
        "resampling_unit": "clip (Yaseen has clip IDs, not patient IDs)",
        "float_clip_accuracy": float(np.mean(clip_f.argmax(1) == y)),
        "int8_clip_accuracy": float(np.mean(clip_q.argmax(1) == y)),
        "max_abs_probability_difference": float(np.abs(p_float - p_int8).max()),
    }


def recordings(rows: list[dict], model: FinalModel) -> dict:
    """rows: dicts with 'patient_id' and a 'load' function returning the recording's
    windows. Loaded one recording at a time; only probabilities are kept."""
    pf, pq, groups, n_win, n_same = [], [], [], 0, 0
    for r in rows:
        feats = r["load"]()
        if len(feats) == 0:                      # shorter than one 1.5-second window
            continue
        a, b = model.predict_windows(feats)
        n_win += len(a); n_same += int(np.sum(a.argmax(1) == b.argmax(1)))
        pf.append(a.mean(0)); pq.append(b.mean(0)); groups.append(r["patient_id"])
    rf, rq = np.array(pf), np.array(pq)
    return {"recordings": len(pf), "patients": len(set(groups)), "windows": n_win,
            "window_agreement": n_same / n_win, "recording_agreement": agreement(rf, rq),
            "recording_agreement_interval": agreement_interval(rq.argmax(1), rf, np.array(groups)),
            "resampling_unit": "patient"}


def load_cached(row: dict):
    """Windows of one manifest row: from its cache if built, else from the WAV."""
    cache = REPO_ROOT / row["cache_path"] if row.get("cache_path") else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as c:
            return c["features"]
    return features_for_file(str(REPO_ROOT / row["path"]), CARDIO_CONFIG)


def manifest_rows(manifest: Path) -> list[dict]:
    """Rows of a cardio.data.build_unseen manifest. Patient IDs are prefixed with the
    dataset name, as cardio.eval.metrics recommends."""
    with manifest.open(encoding="utf-8", newline="") as fh:
        return [{"patient_id": f"{row['dataset']}_{row['patient_id']}",
                 "load": (lambda row=row: load_cached(row))} for row in csv.DictReader(fh)]


def folder_rows(folder: Path) -> list[dict]:
    """Every WAV under a folder; the patient is the file name up to its first underscore."""
    return [{"patient_id": f.stem.split("_")[0],
             "load": (lambda f=f: features_for_file(str(f), CARDIO_CONFIG))}
            for f in sorted(folder.rglob("*.wav"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--manifest", type=Path, help="a cardio.data.build_unseen manifest")
    src.add_argument("--wav-dir", type=Path, help="any folder of recordings (searched recursively)")
    ap.add_argument("--name", default=None, help="label for the extra dataset in the output")
    args = ap.parse_args()

    model = FinalModel()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["model"] = model.files
    out["yaseen_fold0"] = r = yaseen_fold0(model)
    ci = r["clip_agreement_interval"]
    print(f"Yaseen fold 0: window agreement {r['window_agreement']:.2%}, clip agreement "
          f"{r['clip_agreement']:.2%} [{ci['lower_95']:.2%}, {ci['upper_95']:.2%}]")
    if args.manifest or args.wav_dir:
        rows = manifest_rows(args.manifest) if args.manifest else folder_rows(args.wav_dir)
        name = args.name or (args.manifest or args.wav_dir).stem
        out[name] = r = recordings(rows, model)
        ci = r["recording_agreement_interval"]
        print(f"{name}: {r['recordings']} recordings from {r['patients']} patients, recording agreement "
              f"{r['recording_agreement']:.2%} [{ci['lower_95']:.2%}, {ci['upper_95']:.2%}]")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
