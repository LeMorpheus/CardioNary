"""Build the Yaseen 2018 heart-sound manifest with stratified CV folds.

The corpus is 1000 clips, 200 per class, 8 kHz mono, organised one directory
per class. Clips are independent recordings with no published subject IDs, so
folds are assigned per clip with class stratification (docs/05 section 2.3).

Outputs data/manifests/yaseen_manifest.csv
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
AUDIO_DIR = REPO_ROOT / "data" / "raw" / "yaseen" / "audio"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

CLASSES = ["Normal", "AS", "MS", "MR", "MVP"]
# Directory names carry a Korean suffix ("_New_3주기"); match on the prefix.
DIR_TO_CLASS = {"N": "Normal", "AS": "AS", "MS": "MS", "MR": "MR", "MVP": "MVP"}


def class_from_path(p: Path) -> str:
    token = p.parent.name.split("_")[0].upper()
    if token not in DIR_TO_CLASS:
        m = re.match(r"New_([A-Z]+)_", p.stem)
        token = m.group(1).upper() if m else token
    if token not in DIR_TO_CLASS:
        raise ValueError(f"cannot infer class for {p}")
    return DIR_TO_CLASS[token]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    wavs = sorted(AUDIO_DIR.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"no wavs under {AUDIO_DIR}; run core/download_datasets.py first")

    rows = []
    for p in wavs:
        sr, d = wavfile.read(p)
        if d.ndim > 1:
            d = d[:, 0]
        rows.append({
            "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "filename": p.name,
            "diagnosis": class_from_path(p),
            # each clip is its own unit; no subject grouping is published
            "patient_id": p.stem,
            "sample_rate": int(sr),
            "n_samples": int(len(d)),
            "duration_s": round(len(d) / sr, 3),
            "pcm_md5": hashlib.md5(np.ascontiguousarray(d).tobytes()).hexdigest(),
        })
    df = pd.DataFrame(rows)

    dup = df[df.pcm_md5.duplicated(keep=False)]
    if len(dup):
        print(f"WARNING: {len(dup)} clips share identical PCM with another clip")
        conflict = dup.groupby("pcm_md5")["diagnosis"].nunique()
        bad = conflict[conflict > 1]
        if len(bad):
            print(f"  of which {len(bad)} groups have CONFLICTING labels")
            df = df[~df.pcm_md5.isin(bad.index)]
            print(f"  excluded them -> {len(df)} clips remain")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    df = df.reset_index(drop=True)
    df["fold"] = -1
    for f, (_, te) in enumerate(skf.split(df.index, df.diagnosis)):
        df.loc[te, "fold"] = f
    assert (df.fold >= 0).all()
    df["label"] = df["diagnosis"].map({c: i for i, c in enumerate(CLASSES)})

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / "yaseen_manifest.csv"
    df.to_csv(out, index=False)

    print(f"=== Yaseen: {len(df)} clips ===")
    print(df["diagnosis"].value_counts().to_string())
    print(f"\nper fold:\n{pd.crosstab(df.fold, df.diagnosis).to_string()}")
    print(f"\nsample rates: {sorted(df.sample_rate.unique())}")
    print(f"duration s: {df.duration_s.min():.2f} - {df.duration_s.max():.2f} "
          f"(median {df.duration_s.median():.2f})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
