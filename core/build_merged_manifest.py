"""Option 2: the merged KAUH + ICBHI 6-class task (Fraiwan et al.).

Fraiwan et al. report 99.62 % on this task with subject-level 10-fold CV
(PMC8019351). This script reproduces the corpus so that claim can be tested --
and, more importantly, so the DEVICE-ONLY baseline for it can be measured.

Class mapping (Fraiwan 6-class):
    Normal        <- KAUH "Normal"        + ICBHI "Healthy"
    Asthma        <- KAUH "Asthma"
    COPD          <- KAUH "COPD"          + ICBHI "COPD"
    Pneumonia     <- KAUH "Pneumonia"     + ICBHI "Pneumonia"
    BRON          <- KAUH "BRON"          + ICBHI "Bronchiectasis"/"Bronchiolitis"
    Heart Failure <- KAUH "Heart Failure"

Note which sources feed which class: Heart Failure and Asthma come ONLY from
KAUH (Littmann 3200), and most COPD comes only from ICBHI. That is the shape of
a device shortcut, which is why the device baseline matters more than the CNN.

Usage:
    python -m core.build_merged_manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

KAUH_MAP = {
    "Normal": "Normal", "Asthma": "Asthma", "COPD": "COPD",
    "Heart Failure": "Heart Failure",
}
ICBHI_MAP = {
    "Healthy": "Normal", "COPD": "COPD", "Pneumonia": "Pneumonia",
    "Bronchiectasis": "BRON", "Bronchiolitis": "BRON",
}
CLASSES = ["Normal", "Asthma", "COPD", "Pneumonia", "BRON", "Heart Failure"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    kauh = pd.read_csv(MANIFEST_DIR / "kauh_manifest.csv")
    kauh = kauh[kauh.diagnosis.isin(KAUH_MAP)].copy()
    kauh["merged_class"] = kauh.diagnosis.map(KAUH_MAP)
    kauh["source"] = "KAUH"
    kauh["device"] = "Litt3200_KAUH"
    kauh["patient_id"] = "K" + kauh.patient_id.astype(str)

    icbhi_full = pd.read_csv(MANIFEST_DIR / "icbhi_manifest.csv")
    icbhi = icbhi_full[icbhi_full.diagnosis.isin(ICBHI_MAP)].copy()
    icbhi["merged_class"] = icbhi.diagnosis.map(ICBHI_MAP)
    icbhi["source"] = "ICBHI"
    icbhi["patient_id"] = "I" + icbhi.patient_id.astype(str)

    cols = ["path", "filename", "patient_id", "merged_class", "source",
            "device", "sample_rate", "n_samples", "duration_s", "pcm_md5"]
    df = pd.concat([kauh[cols], icbhi[cols]], ignore_index=True)
    df = df.rename(columns={"merged_class": "diagnosis"})
    df["label"] = df.diagnosis.map({c: i for i, c in enumerate(CLASSES)})
    df = df[df.label.notna()].copy()
    df["label"] = df.label.astype(int)

    print(f"merged: {len(df)} recordings, {df.patient_id.nunique()} patients\n")

    pat = df.groupby("patient_id").agg(
        dx=("diagnosis", "first"), src=("source", "first"),
        dev=("device", lambda s: s.value_counts().index[0]),
    )
    print("--- PATIENTS: class x source ---")
    print(pd.crosstab(pat.dx, pat.src).to_string())
    print("\n--- PATIENTS: class x device ---")
    print(pd.crosstab(pat.dx, pat.dev).to_string())

    # patient-disjoint stratified folds
    pats = pat.reset_index()[["patient_id", "dx"]]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pats["fold"] = -1
    for f, (_, te) in enumerate(skf.split(pats.patient_id, pats.dx)):
        pats.loc[pats.index[te], "fold"] = f
    df = df.merge(pats[["patient_id", "fold"]], on="patient_id", how="left")

    out = MANIFEST_DIR / "merged_manifest.csv"
    df.sort_values(["patient_id", "filename"]).to_csv(out, index=False)

    print(f"\n=== merged task: {len(df)} recordings, {df.patient_id.nunique()} patients, "
          f"{df.diagnosis.nunique()} classes ===")
    print(pat.dx.value_counts().to_string())
    print(f"\ntotal audio: {df.duration_s.sum()/60:.1f} min")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
