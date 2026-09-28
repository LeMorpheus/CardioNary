"""Build the windowed feature cache for the KAUH 4-class task.

Enforces the anti-leakage rules from docs/05_DATASET_SELECTION.md section 5 as
hard assertions rather than as documentation. Every window carries its parent
recording id and patient id so the chain window -> recording -> patient -> fold
can be verified at any later stage.

Usage:
    python core/preprocess.py                       # log-mel (default)
    python core/preprocess.py --feature mfcc
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from core.features import PULMO_CONFIG, CARDIO_CONFIG, features_for_file
from core.images import CXR_CONFIG, features_for_image

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"

DATASETS = {
    "kauh": {
        "manifest": REPO_ROOT / "data" / "manifests" / "kauh_manifest.csv",
        "config": PULMO_CONFIG,
        "classes": ["Normal", "Asthma", "Heart Failure", "COPD"],
        "patient_disjoint": True,
    },
    "icbhi": {
        "manifest": REPO_ROOT / "data" / "manifests" / "icbhi_manifest.csv",
        "config": PULMO_CONFIG,
        "classes": ["Bronchiectasis", "Bronchiolitis", "COPD", "Healthy",
                    "Pneumonia", "URTI"],
        "patient_disjoint": True,
        "multi_recording_per_patient": True,
    },
    "icbhi_meditron": {
        "manifest": REPO_ROOT / "data" / "manifests" / "icbhi_meditron_manifest.csv",
        "config": PULMO_CONFIG,
        # Device held CONSTANT (Meditron only) so accuracy reflects pathology,
        # not stethoscope identity. See reports/A1_pipeline_findings.md 2b.
        "classes": ["Bronchiectasis", "Bronchiolitis", "COPD", "Healthy", "URTI"],
        "patient_disjoint": True,
        "multi_recording_per_patient": True,
    },
    "merged": {
        "manifest": REPO_ROOT / "data" / "manifests" / "merged_manifest.csv",
        "config": PULMO_CONFIG,
        # Option 2: Fraiwan et al. KAUH+ICBHI 6-class (published 99.62%).
        "classes": ["Normal", "Asthma", "COPD", "Pneumonia", "BRON", "Heart Failure"],
        "patient_disjoint": True,
        "multi_recording_per_patient": True,
    },
    "kermany": {
        "manifest": REPO_ROOT / "data" / "manifests" / "kermany_manifest.csv",
        "config": CXR_CONFIG,
        "classes": ["Normal", "Pneumonia"],
        "patient_disjoint": True,
        "multi_recording_per_patient": True,
        "modality": "image",
        "label_col": "label",
        "class_col": "binary_class",
    },
    "yaseen": {
        "manifest": REPO_ROOT / "data" / "manifests" / "yaseen_manifest.csv",
        "config": CARDIO_CONFIG,
        "classes": ["Normal", "AS", "MS", "MR", "MVP"],
        "patient_disjoint": False,   # clips are independent; no subject grouping published
    },
}

# Backwards-compatible defaults (KAUH was the first dataset wired up).
MANIFEST = DATASETS["kauh"]["manifest"]
CLASSES = DATASETS["kauh"]["classes"]


def check_manifest(df: pd.DataFrame, spec: dict) -> None:
    """Anti-leakage rules 5.1, 5.2 and duplicate detection, at manifest level."""
    classes = spec["classes"]

    if "filter_code" in df.columns:
        # 5.2 -- exactly one stethoscope filter rendering
        assert set(df.filter_code) == {"D"}, (
            f"multiple KAUH filter renderings present: {sorted(set(df.filter_code))}. "
            "B/D/E are the same acoustic event; mixing them across a split leaks."
        )
    # KAUH/Yaseen have one recording per unit; ICBHI has several per patient.
    if not spec.get("multi_recording_per_patient"):
        assert df.patient_id.is_unique, "duplicate patient_id rows in the manifest"

    # 5.1 -- a unit sits in exactly one fold
    assert (df.groupby("patient_id")["fold"].nunique() == 1).all(), \
        "a patient spans multiple folds"

    # Duplicate content check. Hash the DECODED payload (PCM samples / image
    # pixels), not the file, so re-encoded copies are caught too. This is the
    # rule that found the DP100/DP101 label contradiction in KAUH.
    #
    # For image corpora an exact-duplicate pair is common and harmless (the same
    # film filed twice under one subject), so only CONFLICTING labels are fatal
    # there; for audio any duplicate at all is treated as fatal.
    hashcol = "pcm_md5" if "pcm_md5" in df.columns else "md5"
    class_col = spec.get("class_col", "diagnosis")
    conflict = df.groupby(hashcol)[class_col].nunique()
    bad = conflict[conflict > 1]
    assert bad.empty, (
        f"{len(bad)} identical items carry conflicting labels: "
        f"{df[df[hashcol].isin(bad.index)].filename.tolist()[:6]}"
    )
    if spec.get("modality") != "image":
        dupes = df[df[hashcol].duplicated(keep=False)]
        assert dupes.empty, f"duplicate PCM content:\n{dupes[['filename', hashcol]]}"

    col = spec.get("class_col", "diagnosis")
    assert set(df[col]) <= set(classes), f"unexpected classes: {set(df[col])}"
    if "comorbid" in df.columns:
        assert not df.comorbid.any(), "comorbid subjects must be excluded from the task"


def build(dataset: str, feature: str, overlap: float) -> dict:
    spec = DATASETS[dataset]
    classes = spec["classes"]
    df = pd.read_csv(spec["manifest"])
    check_manifest(df, spec)

    is_image = spec.get("modality") == "image"
    cfg = spec["config"] if is_image else replace(spec["config"], feature=feature)
    rows, frames = cfg.shape

    X, y, pid, fold, rec = [], [], [], [], []
    for i, r in df.iterrows():
        feats = (features_for_image(str(REPO_ROOT / r.path), cfg) if is_image
                 else features_for_file(str(REPO_ROOT / r.path), cfg, overlap=overlap))
        if len(feats) == 0:
            print(f"  WARNING: no windows from {r.filename}")
            continue
        X.append(feats)
        n = len(feats)
        y.append(np.full(n, r[spec.get("label_col", "label")], dtype=np.int64))
        pid.append(np.full(n, r.patient_id, dtype=object))
        fold.append(np.full(n, r.fold, dtype=np.int64))
        rec.append(np.full(n, i, dtype=np.int64))

    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y)
    pid = np.concatenate(pid)
    fold = np.concatenate(fold)
    rec = np.concatenate(rec)

    # 5.4 -- windows of one recording never straddle a split
    tmp = pd.DataFrame({"rec": rec, "fold": fold})
    assert (tmp.groupby("rec")["fold"].nunique() == 1).all(), \
        "windows from one recording landed in different folds"

    # 5.1 again, now at window level
    for f in sorted(set(fold)):
        tr = set(pid[fold != f])
        te = set(pid[fold == f])
        assert not (tr & te), f"fold {f} leaks patients: {sorted(tr & te)}"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{dataset}_{feature}.npz"
    np.savez_compressed(out, X=X, y=y, patient_id=pid.astype(str), fold=fold, rec=rec)
    (CACHE_DIR / f"{dataset}_{feature}_config.json").write_text(
        json.dumps({**cfg.to_dict(), "overlap": overlap, "classes": classes}, indent=2)
    )

    print(f"\n=== feature cache: {out.name} ===")
    print(f"X {X.shape}  {X.nbytes/1e6:.1f} MB  dtype {X.dtype}")
    print(f"value range [{X.min():.2f}, {X.max():.2f}]  mean {X.mean():.2f}  std {X.std():.2f}")
    print(f"\nwindows per class:")
    for i, c in enumerate(classes):
        print(f"  {i} {c:15s} {int((y == i).sum()):5d} windows "
              f"from {len(set(pid[y == i]))} patients")
    print(f"\nwindows per fold:")
    print(pd.crosstab(fold, [classes[v] for v in y]).to_string())
    return {"X": X, "y": y, "pid": pid, "fold": fold}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="kauh", choices=list(DATASETS))
    ap.add_argument("--feature", default="logmel", choices=["logmel", "mfcc"])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--both", action="store_true", help="build both representations")
    args = ap.parse_args()

    feats = ["logmel", "mfcc"] if args.both else [args.feature]
    for f in feats:
        build(args.dataset, f, args.overlap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
