"""Baselines every Chiron model must beat before it can be deployed.

A model that does not beat these is not learning pathology, whatever its
headline accuracy says. Three baselines, evaluated on the SAME patient-disjoint
folds as the CNN:

  1. majority class      -- the floor any classifier must clear
  2. stratified random   -- guessing with the right class priors
  3. DEVICE-ONLY         -- ICBHI only, and the important one.
                            Predicts each test patient's class from the
                            recording device alone, with the device->class map
                            fitted on the training fold. ICBHI's devices are
                            almost perfectly confounded with diagnosis
                            (646/646 AKGC417L recordings are COPD), so this
                            quantifies exactly how much "accuracy" is available
                            without listening to the audio at all.

Usage:
    python -m core.baselines --dataset icbhi
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
REPORT_DIR = REPO_ROOT / "reports"


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse recordings to one row per patient (the evaluation unit)."""
    dx = "binary_class" if "binary_class" in df.columns else "diagnosis"
    agg = {dx: "first", "label": "first", "fold": "first"}
    if "device" in df.columns:
        # a patient's dominant device
        agg["device"] = lambda s: s.value_counts().index[0]
    out = df.groupby("patient_id").agg(agg).reset_index()
    return out.rename(columns={dx: "diagnosis"})


def majority_baseline(pt: pd.DataFrame) -> tuple[float, float]:
    preds = np.empty(len(pt), dtype=int)
    for f in sorted(pt.fold.unique()):
        tr, te = pt.fold != f, pt.fold == f
        preds[te.values] = pt.loc[tr, "label"].mode()[0]
    return accuracy_score(pt.label, preds), f1_score(pt.label, preds,
                                                     average="macro", zero_division=0)


def stratified_random_baseline(pt: pd.DataFrame, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(200):                       # average over draws
        preds = np.empty(len(pt), dtype=int)
        for f in sorted(pt.fold.unique()):
            tr, te = pt.fold != f, pt.fold == f
            p = pt.loc[tr, "label"].value_counts(normalize=True)
            preds[te.values] = rng.choice(p.index, size=int(te.sum()), p=p.values)
        accs.append(accuracy_score(pt.label, preds))
        f1s.append(f1_score(pt.label, preds, average="macro", zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def device_baseline(pt: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Predict class from recording device alone. ICBHI only."""
    preds = np.empty(len(pt), dtype=int)
    for f in sorted(pt.fold.unique()):
        tr, te = pt.fold != f, pt.fold == f
        lut = pt[tr].groupby("device")["label"].agg(lambda s: s.mode()[0])
        fallback = pt.loc[tr, "label"].mode()[0]
        preds[te.values] = [lut.get(d, fallback) for d in pt.loc[te, "device"]]
    acc = accuracy_score(pt.label, preds)
    f1 = f1_score(pt.label, preds, average="macro", zero_division=0)
    tab = pd.crosstab(pt.device, pt.diagnosis)
    return acc, f1, tab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="icbhi", choices=["kauh", "icbhi", "icbhi_meditron", "merged", "yaseen", "kermany"])
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST_DIR / f"{args.dataset}_manifest.csv")
    pt = patient_table(df)
    n_classes = pt.label.nunique()

    print(f"=== baselines: {args.dataset} ===")
    print(f"{len(pt)} evaluation units, {n_classes} classes, "
          f"{pt.fold.nunique()} patient-disjoint folds\n")
    print("class distribution (patients):")
    print(pt.diagnosis.value_counts().to_string())
    print()

    results = {}
    acc, f1 = majority_baseline(pt)
    results["majority_class"] = {"accuracy": acc, "macro_f1": f1}
    print(f"  majority class      accuracy {acc:.4f}   macro-F1 {f1:.4f}")

    acc, f1 = stratified_random_baseline(pt)
    results["stratified_random"] = {"accuracy": acc, "macro_f1": f1}
    print(f"  stratified random   accuracy {acc:.4f}   macro-F1 {f1:.4f}")

    if "device" in pt.columns and pt.device.nunique() > 1:
        acc, f1, tab = device_baseline(pt)
        results["device_only"] = {"accuracy": acc, "macro_f1": f1}
        print(f"  DEVICE ONLY         accuracy {acc:.4f}   macro-F1 {f1:.4f}   "
              f"<-- audio never touched")
        print("\n  patient device x diagnosis:")
        print("   " + tab.to_string().replace("\n", "\n   "))

    hurdle = max(v["accuracy"] for v in results.values())
    print(f"\n>>> HURDLE: any deployable model must beat {hurdle:.4f} accuracy")
    print(f">>> and should beat the best macro-F1 "
          f"({max(v['macro_f1'] for v in results.values()):.4f})")

    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / f"baselines_{args.dataset}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset,
        "n_units": int(len(pt)),
        "n_classes": int(n_classes),
        "baselines": results,
        "hurdle_accuracy": float(hurdle),
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
