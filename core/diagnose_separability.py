"""Is the class signal actually present, or is the CNN pipeline at fault?

Decisive diagnostic: compute simple hand-crafted per-recording features and
classify them with classical models using the SAME patient-disjoint folds.

  * If a classical baseline clears the CNN comfortably, the CNN/training setup
    is the problem.
  * If it lands in the same place, the task is genuinely this hard at
    patient level and the published KAUH numbers rely on leaky splits.

Also reports the majority-class floor and checks for a duration confound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from core.features import PULMO_CONFIG, load_wav, preprocess, log_mel, iter_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "Asthma", "Heart Failure", "COPD"]


def recording_features(path: str) -> np.ndarray:
    """Compact summary statistics of the log-mel surface for one recording."""
    cfg = PULMO_CONFIG
    x = preprocess(load_wav(path, cfg), cfg)
    mats = [log_mel(w, cfg) for w in iter_windows(x, cfg, overlap=0.5)]
    if not mats:
        return np.zeros(64 * 4 + 6)
    M = np.concatenate(mats, axis=1)           # (n_mels, total_frames)

    feats = [M.mean(1), M.std(1), np.percentile(M, 10, axis=1), np.percentile(M, 90, axis=1)]

    # a few global descriptors
    band = M.mean(1)
    tot = np.exp(band).sum() + 1e-9
    centroid = float((np.arange(len(band)) * np.exp(band)).sum() / tot)
    lo = float(np.exp(band[:16]).sum() / tot)
    mid = float(np.exp(band[16:40]).sum() / tot)
    hi = float(np.exp(band[40:]).sum() / tot)
    flux = float(np.mean(np.abs(np.diff(M, axis=1))))
    dyn = float(M.max() - M.min())
    feats.append(np.array([centroid, lo, mid, hi, flux, dyn]))
    return np.concatenate(feats)


def main() -> int:
    df = pd.read_csv(REPO_ROOT / "data" / "manifests" / "kauh_manifest.csv")
    print(f"{len(df)} recordings, {df.patient_id.nunique()} patients")

    # ---- duration confound check -------------------------------------------
    print("\n--- duration by class (confound check) ---")
    print(df.groupby("diagnosis")["duration_s"].describe()[["count", "mean", "std", "min", "max"]].to_string())

    # ---- majority-class floor ----------------------------------------------
    counts = df["diagnosis"].value_counts()
    floor = counts.max() / counts.sum()
    print(f"\nmajority-class floor: {floor:.4f} (always predict '{counts.idxmax()}')")

    # ---- classical baselines on the same folds -----------------------------
    print("\ncomputing hand-crafted features ...")
    X = np.stack([recording_features(str(REPO_ROOT / p)) for p in df.path])
    y = df.label.values
    fold = df.fold.values
    print(f"feature matrix {X.shape}")

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=0.5,
                                                   class_weight="balanced")),
        "rf": RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                     class_weight="balanced", random_state=0, n_jobs=-1),
    }

    print("\n--- patient-level accuracy, patient-disjoint 5-fold ---")
    for name, mdl in models.items():
        preds = np.zeros_like(y)
        for f in sorted(set(fold)):
            tr, te = fold != f, fold == f
            m = mdl.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        print(f"  {name:8s} accuracy {acc:.4f}   macro-F1 {f1:.4f}")

    # ---- binary sub-tasks: where does the signal actually live? -------------
    print("\n--- binary sub-tasks (logreg, same folds) ---")
    pairs = [(0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (2, 3)]
    for a, b in pairs:
        mask = np.isin(y, [a, b])
        Xa, ya, fa = X[mask], y[mask], fold[mask]
        preds = np.zeros_like(ya)
        for f in sorted(set(fa)):
            tr, te = fa != f, fa == f
            if len(set(ya[tr])) < 2 or te.sum() == 0:
                preds[te] = ya[tr][0] if len(ya[tr]) else a
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=3000, C=0.5,
                                                 class_weight="balanced"))
            m.fit(Xa[tr], ya[tr])
            preds[te] = m.predict(Xa[te])
        acc = accuracy_score(ya, preds)
        base = max(np.mean(ya == a), np.mean(ya == b))
        print(f"  {CLASSES[a]:14s} vs {CLASSES[b]:14s} n={mask.sum():3d}  "
              f"acc {acc:.3f}  (floor {base:.3f})")

    # ---- normal vs abnormal -------------------------------------------------
    yb = (y != 0).astype(int)
    preds = np.zeros_like(yb)
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=0.5,
                                             class_weight="balanced"))
        m.fit(X[tr], yb[tr])
        preds[te] = m.predict(X[te])
    print(f"\n  Normal vs Abnormal      n={len(yb)}  acc {accuracy_score(yb, preds):.3f}  "
          f"(floor {max(yb.mean(), 1-yb.mean()):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
