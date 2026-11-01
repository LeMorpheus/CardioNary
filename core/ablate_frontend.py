"""Ablate front-end design choices using a fast classical baseline.

The CNN and a RandomForest on hand-crafted features land within ~1 pp of each
other on this corpus, so the RF is used as a cheap proxy to search front-end
configurations in seconds instead of minutes. Whatever wins here is then
re-checked with the CNN.

Every configuration is evaluated with the SAME patient-disjoint folds.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import sosfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from core.features import PULMO_CONFIG, load_wav, log_mel, iter_windows, _Tables

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "Asthma", "Heart Failure", "COPD"]


def preprocess_variant(x, cfg, norm: str):
    """features.preprocess with selectable amplitude handling."""
    t = _Tables.get(cfg)
    x = x - np.mean(x)
    x = sosfilt(t["sos"], x)
    if norm == "rms":
        r = np.sqrt(np.mean(x**2))
        if r > 1e-9:
            x = x / r * 0.1
    elif norm == "peak":
        p = np.max(np.abs(x))
        if p > 1e-9:
            x = x / p * 0.5
    return x


def summary_features(path, cfg, norm) -> np.ndarray:
    x = preprocess_variant(load_wav(path, cfg), cfg, norm)
    mats = [log_mel(w, cfg) for w in iter_windows(x, cfg, overlap=0.5)]
    if not mats:
        return np.zeros(cfg.n_mels * 4)
    M = np.concatenate(mats, axis=1)
    return np.concatenate(
        [M.mean(1), M.std(1), np.percentile(M, 10, axis=1), np.percentile(M, 90, axis=1)]
    )


def evaluate(df, cfg, norm) -> tuple[float, float]:
    X = np.stack([summary_features(str(REPO_ROOT / p), cfg, norm) for p in df.path])
    y, fold = df.label.values, df.fold.values
    preds = np.zeros_like(y)
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        m = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=0, n_jobs=-1,
        ).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return accuracy_score(y, preds), f1_score(y, preds, average="macro", zero_division=0)


def main() -> int:
    df = pd.read_csv(REPO_ROOT / "data" / "manifests" / "kauh_manifest.csv")
    floor = df.diagnosis.value_counts().max() / len(df)
    print(f"{len(df)} patients, majority-class floor {floor:.3f}\n")

    variants = [
        ("baseline  80-1900 rms",    replace(PULMO_CONFIG), "rms"),
        ("no-norm   80-1900",        replace(PULMO_CONFIG), "none"),
        ("peak-norm 80-1900",        replace(PULMO_CONFIG), "peak"),
        ("wide      20-1900 rms",    replace(PULMO_CONFIG, band_low_hz=20.0), "rms"),
        ("wide      20-1900 none",   replace(PULMO_CONFIG, band_low_hz=20.0), "none"),
        ("low       20-800  rms",    replace(PULMO_CONFIG, band_low_hz=20.0,
                                              band_high_hz=800.0), "rms"),
        ("hi-res    128 mel",        replace(PULMO_CONFIG, n_mels=128), "rms"),
        ("fine-fft  512/256",        replace(PULMO_CONFIG, n_fft=512, hop=256,
                                              n_frames=80), "rms"),
        ("long-win  10 s",           replace(PULMO_CONFIG, window_s=10.0,
                                              n_frames=320), "rms"),
        ("short-win 2.5 s",          replace(PULMO_CONFIG, window_s=2.5,
                                              n_frames=80), "rms"),
    ]

    rows = []
    for name, cfg, norm in variants:
        try:
            acc, f1 = evaluate(df, cfg, norm)
            rows.append((name, acc, f1))
            print(f"  {name:26s} acc {acc:.4f}   macro-F1 {f1:.4f}")
        except Exception as e:  # noqa: BLE001 - report and continue the sweep
            print(f"  {name:26s} FAILED: {type(e).__name__}: {e}")

    print("\nbest by accuracy:")
    for name, acc, f1 in sorted(rows, key=lambda r: -r[1])[:3]:
        print(f"  {name:26s} acc {acc:.4f}  F1 {f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
