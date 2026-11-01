"""Does the 8-bit heart model give the same answer as the float model?
(roadmap S2, proposal 4.5 and 4.7)

Agreement has a known correct answer, which accuracy on new data does not: the
8-bit model should predict the same class as the float model on identical input.
Measured on Yaseen fold 0, the fold the final model never trained on, at window
and clip level; and optionally on any folder of recordings, such as PhysioNet
2022, which needs no labels for this.

Usage:
    python -m cardio.eval.agreement
    python -m cardio.eval.agreement --wav-dir data/raw/circor --name physionet_2022

Writes (merging with earlier runs):
    reports/heart_agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    y = pd.Series(cache["y"][te]).groupby(pid, sort=False).first().to_numpy()
    return {
        "windows": int(te.sum()), "clips": int(len(clip_f)),
        "window_agreement": agreement(p_float, p_int8),
        "clip_agreement": agreement(clip_f, clip_q),
        "float_clip_accuracy": float(np.mean(clip_f.argmax(1) == y)),
        "int8_clip_accuracy": float(np.mean(clip_q.argmax(1) == y)),
        "max_abs_probability_difference": float(np.abs(p_float - p_int8).max()),
    }


def wav_folder(model: FinalModel, folder: Path) -> dict:
    files = sorted(folder.rglob("*.wav"))
    pf, pq, n_win = [], [], 0
    for f in files:
        feats = features_for_file(str(f), CARDIO_CONFIG)
        if len(feats) == 0:
            continue
        a, b = model.predict_windows(feats)
        pf.append(a); pq.append(b); n_win += len(a)
    wf, wq = np.concatenate(pf), np.concatenate(pq)
    rf = np.array([a.mean(0) for a in pf]); rq = np.array([b.mean(0) for b in pq])
    return {"recordings": len(pf), "windows": n_win,
            "window_agreement": agreement(wf, wq), "recording_agreement": agreement(rf, rq)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav-dir", type=Path, help="optional folder of recordings (searched recursively)")
    ap.add_argument("--name", default=None, help="label for --wav-dir in the output")
    args = ap.parse_args()

    model = FinalModel()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["model"] = model.files
    out["yaseen_fold0"] = r = yaseen_fold0(model)
    print(f"Yaseen fold 0: window agreement {r['window_agreement']:.2%}, clip agreement "
          f"{r['clip_agreement']:.2%} (float {r['float_clip_accuracy']:.2%}, 8-bit {r['int8_clip_accuracy']:.2%})")
    if args.wav_dir:
        name = args.name or args.wav_dir.name
        out[name] = r = wav_folder(model, args.wav_dir)
        print(f"{name}: {r['recordings']} recordings, recording agreement {r['recording_agreement']:.2%}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
