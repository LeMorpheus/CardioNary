"""Parse the KAUH lung-sound corpus into a patient-level manifest with CV folds.

KAUH filename convention (Fraiwan et al., Data in Brief):
    [Filter][P][PatientNo]_[Diagnosis],[SoundType],[ChestZone],[Age],[Gender].wav
e.g. BP60_heart failure,Crep,P L L,83,F.wav

Filter renderings (SAME acoustic event, three stethoscope filters):
    B = Bell        20-1000 Hz, emphasis 20-200 Hz
    D = Diaphragm   20-2000 Hz, emphasis 100-500 Hz   <-- we use this one
    E = Extended    20-1000 Hz, emphasis 50-500 Hz

Splitting B/D/E of one patient across a train/test boundary is data leakage; see
docs/04_DATASETS_RESEARCH.md section 6.2. We keep D only.

Outputs data/manifests/kauh_manifest.csv
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "data" / "raw" / "kauh" / "audio"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

FNAME_RE = re.compile(r"^([BDE])P(\d+)_(.*)$")

# Raw diagnosis strings vary in case and spelling across the corpus.
DIAGNOSIS_MAP = {
    "n": "Normal",
    "asthma": "Asthma",
    "copd": "COPD",
    "heart failure": "Heart Failure",
    "lung fibrosis": "Lung Fibrosis",
    "pneumonia": "Pneumonia",
    "bron": "BRON",
    "plueral effusion": "Pleural Effusion",
    "pleural effusion": "Pleural Effusion",
    "asthma and lung fibrosis": "Asthma+Lung Fibrosis",
    "heart failure + copd": "Heart Failure+COPD",
    "heart failure + lung fibrosis": "Heart Failure+Lung Fibrosis",
}

# The Track A task (docs/05_DATASET_SELECTION.md section 3.1).
TASK_4CLASS = ["Normal", "Asthma", "Heart Failure", "COPD"]

# ---------------------------------------------------------------------------
# Corpus defects found during pipeline bring-up (2026-08-13).
#
# A byte-identical-PCM scan of the corpus found three filing errors upstream:
#
#   1. DP100 "N,N,P R M,70,F"  and DP101 "Asthma,E W,P L M,12,F"
#      are BYTE-IDENTICAL in both the B and D renderings (Pearson r = 1.0000),
#      yet are labelled Normal and Asthma, for patients of different age (70 vs
#      12) and different auscultation sites. One of the two is misfiled and
#      there is no way to tell which, so BOTH are excluded: keeping either
#      injects a known-wrong training example, and keeping both puts identical
#      audio under two labels.
#
#   2. EP76 / EP77 are byte-identical in the E (Extended) rendering only.
#      The D renderings of P076 and P077 are genuinely distinct (max pairwise
#      correlation across all D files is 1.0000 for the P100/P101 pair and
#      below 0.90 for every other pair), so these two patients are RETAINED.
#
#   3. BP9 / DP9 are byte-identical, i.e. the Bell and Diaphragm renderings of
#      P009 are the same file. Harmless here: we keep only the D rendering.
#
# Rerun core/audit_duplicates.py to reproduce this analysis.
# ---------------------------------------------------------------------------
EXCLUDE_PATIENTS = {
    "P100": "byte-identical audio to P101 with a conflicting label",
    "P101": "byte-identical audio to P100 with a conflicting label",
}


def parse_filename(path: Path) -> dict:
    """Return the metadata fields encoded in one KAUH filename."""
    stem = path.stem
    m = FNAME_RE.match(stem)
    if not m:
        raise ValueError(f"unparsable filename: {path.name}")
    filter_code, patient_no, rest = m.groups()

    parts = [p.strip() for p in rest.split(",")]
    # Expected: diagnosis, sound_type, chest_zone, age, gender
    if len(parts) < 5:
        parts += [""] * (5 - len(parts))
    diagnosis_raw, sound_type, chest_zone, age, gender = parts[:5]

    key = diagnosis_raw.strip().lower()
    key = re.sub(r"\s+", " ", key)
    diagnosis = DIAGNOSIS_MAP.get(key)
    if diagnosis is None:
        raise ValueError(f"unmapped diagnosis {diagnosis_raw!r} in {path.name}")

    try:
        age_val = int(age)
    except ValueError:
        age_val = -1

    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "filename": path.name,
        "filter_code": filter_code,
        "patient_id": f"P{int(patient_no):03d}",
        "diagnosis_raw": diagnosis_raw,
        "diagnosis": diagnosis,
        "comorbid": "+" in diagnosis,
        "sound_type": sound_type,
        "chest_zone": chest_zone,
        "age": age_val,
        "sex": gender.upper()[:1],
    }


def audio_stats(path: Path) -> dict:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    return {
        "sample_rate": int(sr),
        "n_samples": int(len(data)),
        "duration_s": round(len(data) / sr, 3),
        # Hash the decoded PCM, not the file: catches re-encoded duplicates.
        "pcm_md5": hashlib.md5(np.ascontiguousarray(data).tobytes()).hexdigest(),
    }


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Patient-disjoint, class-stratified CV folds.

    Folds are assigned at PATIENT level, then broadcast to rows, so every
    recording and (later) every window of a patient lands in the same fold.
    """
    patients = (
        df.groupby("patient_id")["diagnosis"].first().reset_index().sort_values("patient_id")
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    patients["fold"] = -1
    for fold, (_, test_idx) in enumerate(
        skf.split(patients["patient_id"], patients["diagnosis"])
    ):
        patients.loc[patients.index[test_idx], "fold"] = fold
    assert (patients["fold"] >= 0).all()
    return df.merge(patients[["patient_id", "fold"]], on="patient_id", how="left")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--filter-code", default="D", choices=["B", "D", "E"],
                    help="stethoscope filter rendering to keep (default D=Diaphragm)")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="COPD has only 9 patients; 5 folds keeps ~2 per fold")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs under {AUDIO_DIR}; run core/download_datasets.py first")

    rows = [{**parse_filename(p), **audio_stats(p)} for p in wavs]
    full = pd.DataFrame(rows)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(MANIFEST_DIR / "kauh_all_renderings.csv", index=False)

    print(f"parsed {len(full)} files, {full.patient_id.nunique()} patients")
    print("\n--- patients per diagnosis (all renderings collapsed) ---")
    per_patient = full.groupby("patient_id")["diagnosis"].first()
    print(per_patient.value_counts().to_string())

    # ---- leakage rule 5.2: keep exactly one filter rendering -----------------
    df = full[full.filter_code == args.filter_code].copy()
    print(f"\nkept filter rendering {args.filter_code!r}: {len(df)} files")

    # ---- Track A task: pure 4-class, comorbid + defective subjects excluded --
    df["excluded_reason"] = df["patient_id"].map(EXCLUDE_PATIENTS)
    df["in_task_4class"] = (
        df["diagnosis"].isin(TASK_4CLASS)
        & ~df["comorbid"]
        & df["excluded_reason"].isna()
    )
    task = df[df.in_task_4class].copy()

    excluded = df[~df.in_task_4class]
    print(f"\nexcluded {len(excluded)} recordings not in the 4-class task:")
    print(excluded["diagnosis"].value_counts().to_string())

    defects = df[df["excluded_reason"].notna()]
    if len(defects):
        print(f"\nexcluded for corpus defects ({len(defects)}):")
        for _, r in defects.iterrows():
            print(f"  {r.patient_id} {r.diagnosis:14s} -- {r.excluded_reason}")

    task = assign_folds(task, n_folds=args.n_folds, seed=args.seed)
    task["label"] = task["diagnosis"].map({c: i for i, c in enumerate(TASK_4CLASS)})

    out = MANIFEST_DIR / "kauh_manifest.csv"
    task.sort_values(["patient_id"]).to_csv(out, index=False)

    print(f"\n=== Track A 4-class task: {len(task)} recordings, "
          f"{task.patient_id.nunique()} patients ===")
    print(task["diagnosis"].value_counts().to_string())
    print(f"\n--- patients per fold ({args.n_folds}-fold, patient-disjoint) ---")
    print(pd.crosstab(task["fold"], task["diagnosis"]).to_string())
    print(f"\ntotal audio: {task.duration_s.sum()/60:.1f} min")
    print(f"sample rates: {sorted(task.sample_rate.unique())}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
