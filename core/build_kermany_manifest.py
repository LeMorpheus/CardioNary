"""Build a patient-disjoint manifest for the Kermany paediatric chest X-ray corpus.

Why this corpus and not a COVID one: both classes come from the SAME hospital
(Guangzhou Women and Children's Medical Center), the same paediatric cohort
(1-5 years) and the same imaging pipeline. The source confound that inflates
published COVID X-ray results -- paediatric Normal vs adult COVID, worth ~23 pp
per CoVScreen (arXiv:2405.07674) -- does not exist here. See
reports/A1_pipeline_findings.md.

Filename conventions:
    PNEUMONIA   person{PID}_{bacteria|virus}_{n}.jpeg
    NORMAL      IM-{study}-{n}.jpeg  or  NORMAL2-IM-{study}-{n}.jpeg

Both encode a subject/study id, and pneumonia subjects have SEVERAL images each,
so a random image-level split puts the same child in train and test. This script
recombines the official train/test dirs and assigns its own patient-disjoint
folds instead.

Usage:
    python -m core.build_kermany_manifest
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_ROOT = REPO_ROOT / "data" / "raw" / "kermany" / "img" / "chest_xray"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

PNEU_RE = re.compile(r"^person(\d+)_(bacteria|virus)_(\d+)", re.I)
NORM_RE = re.compile(r"^(?:NORMAL2-)?IM-(\d+)-(\d+)", re.I)

CLASSES_BINARY = ["Normal", "Pneumonia"]
CLASSES_3 = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]


def parse(p: Path, cls_dir: str, split_dir: str) -> dict | None:
    stem = p.stem
    if cls_dir.upper() == "PNEUMONIA":
        m = PNEU_RE.match(stem)
        if not m:
            print(f"  WARNING unparsable pneumonia name: {p.name}")
            return None
        pid, subtype = f"P{int(m.group(1)):05d}", m.group(2).lower()
        diagnosis = "Bacterial Pneumonia" if subtype == "bacteria" else "Viral Pneumonia"
        binary = "Pneumonia"
    else:
        m = NORM_RE.match(stem)
        if not m:
            print(f"  WARNING unparsable normal name: {p.name}")
            return None
        pid, subtype = f"N{int(m.group(1)):05d}", "normal"
        diagnosis, binary = "Normal", "Normal"

    return {
        "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
        "filename": p.name,
        "patient_id": pid,
        "subtype": subtype,
        "diagnosis": diagnosis,
        "binary_class": binary,
        "official_split": split_dir,
    }


def image_stats(p: Path) -> dict:
    with Image.open(p) as im:
        w, h = im.size
        mode = im.mode
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    return {
        "width": w, "height": h, "mode": mode,
        "md5": hashlib.md5(a.tobytes()).hexdigest(),
        "mean_intensity": float(a.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rows = []
    for split_dir in ("train", "test"):
        for cls_dir in ("NORMAL", "PNEUMONIA"):
            d = IMG_ROOT / split_dir / cls_dir
            if not d.exists():
                continue
            for p in sorted(d.glob("*.jpeg")):
                r = parse(p, cls_dir, split_dir)
                if r:
                    rows.append({**r, **image_stats(p)})
    df = pd.DataFrame(rows)
    print(f"parsed {len(df)} images, {df.patient_id.nunique()} subjects\n")

    print("--- images per class ---")
    print(df.diagnosis.value_counts().to_string())
    print("\n--- SUBJECTS per class ---")
    per_pat = df.groupby("patient_id")["diagnosis"].first()
    print(per_pat.value_counts().to_string())
    print(f"\nimages per subject: median {df.groupby('patient_id').size().median():.0f}, "
          f"max {df.groupby('patient_id').size().max()}")

    # ---- does the OFFICIAL split leak subjects? ------------------------------
    tr = set(df.loc[df.official_split == "train", "patient_id"])
    te = set(df.loc[df.official_split == "test", "patient_id"])
    overlap = tr & te
    print(f"\n--- official split audit ---")
    print(f"train subjects {len(tr)}, test subjects {len(te)}, "
          f"OVERLAP {len(overlap)}")
    if overlap:
        n_leaked = int(df[df.patient_id.isin(overlap)].shape[0])
        print(f"  *** {len(overlap)} subjects appear in BOTH official splits "
              f"({n_leaked} images). Published results using the official split "
              f"are subject-leaked. ***")
        print(f"  examples: {sorted(overlap)[:8]}")

    # ---- duplicate pixel content --------------------------------------------
    dupes = df[df.md5.duplicated(keep=False)]
    if len(dupes):
        conflict = dupes.groupby("md5")["diagnosis"].nunique()
        bad = conflict[conflict > 1]
        print(f"\n{len(dupes)} images share pixel content with another image; "
              f"{len(bad)} hash groups have CONFLICTING labels")
        if len(bad):
            df = df[~df.md5.isin(bad.index)].copy()
            print(f"  excluded them -> {len(df)} images remain")

    # ---- our own patient-disjoint folds -------------------------------------
    pats = df.groupby("patient_id")["diagnosis"].first().reset_index()
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pats["fold"] = -1
    for f, (_, idx) in enumerate(skf.split(pats.patient_id, pats.diagnosis)):
        pats.loc[pats.index[idx], "fold"] = f
    df = df.merge(pats[["patient_id", "fold"]], on="patient_id", how="left")

    df["label"] = df.binary_class.map({c: i for i, c in enumerate(CLASSES_BINARY)})
    df["label3"] = df.diagnosis.map({c: i for i, c in enumerate(CLASSES_3)})

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / "kermany_manifest.csv"
    df.sort_values(["patient_id", "filename"]).to_csv(out, index=False)

    print(f"\n=== task: {len(df)} images, {df.patient_id.nunique()} subjects ===")
    print("\nimages per fold x class:")
    print(pd.crosstab(df.fold, df.diagnosis).to_string())
    print("\nsubjects per fold x class:")
    print(df.groupby(["fold", "diagnosis"])["patient_id"].nunique().unstack(fill_value=0).to_string())
    print(f"\nimage size: {df.width.min()}x{df.height.min()} to {df.width.max()}x{df.height.max()}")
    print(f"modes: {df['mode'].value_counts().to_dict()}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
