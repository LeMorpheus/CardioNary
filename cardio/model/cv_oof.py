"""5-fold cross-validation of the heart-sound CNN on Yaseen, keeping the
out-of-fold probabilities (roadmap S1, proposal 4.1).

Same protocol and settings as the baseline run of core/train.py: the compact
depthwise-separable CNN, no augmentation, the folds in the committed manifest,
normalisation fitted on the training folds only, seed = fold number. The one
addition is that each fold's model is run on that fold's test clips and the
probabilities are kept. Calibration and late fusion need exactly these:
predictions from models that never saw the clips they score.

Usage:
    python -m core.preprocess --dataset yaseen     # once, builds the cache
    python -m cardio.model.cv_oof

Writes:
    reports/heart_oof_yaseen.csv   one row per clip: patient_id, fold, label,
                                   diagnosis, n_windows, p_Normal ... p_MVP
                                   (mean of the softmax over the clip's windows)
    reports/heart_cv_yaseen.json   accuracy and macro-F1 per fold (mean, std),
                                   per-class recall, confusion matrix, and the
                                   refer-or-not AUROC of p(abnormal) = 1 - p(Normal)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)

import core.train as train

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data" / "cache" / "yaseen_logmel.npz"
MANIFEST = REPO_ROOT / "data" / "manifests" / "yaseen_manifest.csv"
REPORT_DIR = REPO_ROOT / "reports"
CLASSES = train.DATASET_CLASSES["yaseen"]          # Normal, AS, MS, MR, MVP

# The baseline settings (core/train.py defaults with --no-augment).
SETTINGS = argparse.Namespace(width=16, blocks=4, dropout=0.3, arch="dwsep",
                              lr=2e-3, batch=32, epochs=120, augment=False)


def clip_table(prob: np.ndarray, pid: np.ndarray, y: np.ndarray, f: int) -> pd.DataFrame:
    """Windows -> one row per clip, with the mean softmax over its windows."""
    cols = ["p_" + c for c in CLASSES]
    df = pd.DataFrame(prob, columns=cols)
    df["patient_id"], df["label"] = pid, y
    out = df.groupby("patient_id", sort=False).agg(
        label=("label", "first"), n_windows=("label", "size"), **{c: (c, "mean") for c in cols})
    out.insert(0, "fold", f)
    return out.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=SETTINGS.epochs,
                    help="training epochs per fold (baseline: 120)")
    args = ap.parse_args()
    SETTINGS.epochs = args.epochs

    cache = np.load(CACHE, allow_pickle=True)
    X, y, pid, fold = cache["X"], cache["y"], cache["patient_id"], cache["fold"]
    folds = sorted(set(fold.tolist()))
    # run_fold labels its columns from the module-level class list, which
    # defaults to another dataset's classes; point it at Yaseen's.
    train.CLASSES = CLASSES
    print(f"Yaseen cache: {X.shape[0]} windows from {len(set(pid))} clips, {len(folds)} folds")

    tables, per_fold = [], []
    for f in folds:
        tr, te = fold != f, fold == f
        r = train.run_fold(X[tr], y[tr], X[te], y[te], pid[te], SETTINGS, len(CLASSES), seed=f)
        mu, sd = r["norm"]
        xte = ((X[te] - mu) / sd)[..., None].astype(np.float32)
        prob = r["model"].predict(xte, verbose=0).reshape(int(te.sum()), -1)
        t = clip_table(prob, pid[te], y[te], f)
        tables.append(t)
        pred = t[["p_" + c for c in CLASSES]].to_numpy().argmax(1)
        per_fold.append({
            "fold": f, "clips": len(t),
            "accuracy": accuracy_score(t.label, pred),
            "macro_f1": f1_score(t.label, pred, average="macro"),
            "window_accuracy": accuracy_score(y[te], prob.argmax(1)),
        })
        print(f"  fold {f}: {len(t)} clips, accuracy {per_fold[-1]['accuracy']:.4f}, "
              f"macro-F1 {per_fold[-1]['macro_f1']:.4f}")

    oof = pd.concat(tables, ignore_index=True)
    diag = pd.read_csv(MANIFEST).set_index("patient_id")["diagnosis"]
    oof.insert(3, "diagnosis", oof.patient_id.map(diag))
    REPORT_DIR.mkdir(exist_ok=True)
    oof.to_csv(REPORT_DIR / "heart_oof_yaseen.csv", index=False, float_format="%.6f")

    p = oof[["p_" + c for c in CLASSES]].to_numpy()
    pred, true = p.argmax(1), oof.label.to_numpy()
    pf = pd.DataFrame(per_fold)
    result = {
        "dataset": "yaseen",
        "protocol": "5-fold cross-validation over the committed folds; clip-level scores are "
                    "the mean softmax over each clip's 3-second windows",
        "settings": vars(SETTINGS),
        "clips": int(len(oof)), "windows": int(X.shape[0]),
        "accuracy_mean": float(pf.accuracy.mean()), "accuracy_std": float(pf.accuracy.std()),
        "macro_f1_mean": float(pf.macro_f1.mean()), "macro_f1_std": float(pf.macro_f1.std()),
        "window_accuracy_mean": float(pf.window_accuracy.mean()),
        "pooled_accuracy": float(accuracy_score(true, pred)),
        "pooled_macro_f1": float(f1_score(true, pred, average="macro")),
        "per_class_recall": dict(zip(CLASSES, map(float, recall_score(true, pred, average=None)))),
        "refer_or_not_auroc": float(roc_auc_score(true != 0, 1.0 - p[:, 0])),
        "confusion_matrix": {"classes": CLASSES, "rows_true_cols_pred":
                             confusion_matrix(true, pred, labels=range(len(CLASSES))).tolist()},
        "per_fold": per_fold,
        "target": "accuracy_mean >= 0.95 (proposal 4.1)",
        "meets_target": bool(pf.accuracy.mean() >= 0.95),
    }
    (REPORT_DIR / "heart_cv_yaseen.json").write_text(json.dumps(result, indent=2))

    print(f"\naccuracy {result['accuracy_mean']:.4f} +/- {result['accuracy_std']:.4f}   "
          f"macro-F1 {result['macro_f1_mean']:.4f} +/- {result['macro_f1_std']:.4f}")
    print("per-class recall:", {k: round(v, 3) for k, v in result["per_class_recall"].items()})
    print(f"refer-or-not AUROC {result['refer_or_not_auroc']:.4f}")
    print("wrote reports/heart_oof_yaseen.csv and reports/heart_cv_yaseen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
