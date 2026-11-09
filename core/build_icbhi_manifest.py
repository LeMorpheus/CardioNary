"""Extract the ICBHI 2017 corpus from HF parquet shards and build a manifest.

The mirror (Nacl666/icbhi) preserves the ORIGINAL ICBHI filenames inside the
Audio struct, which is what makes patient-disjoint evaluation possible:

    101_1b1_Al_sc_Meditron.wav
    |   |   |  |  |
    |   |   |  |  +-- recording equipment
    |   |   |  +----- acquisition mode (sc = single channel, mc = multi channel)
    |   |   +-------- chest location (Tc, Al, Ar, Pl, Pr, Ll, Lr)
    |   +------------ recording index within the patient
    +---------------- PATIENT ID

Chest location and device are both confounds worth reporting, so both are
carried into the manifest.

Usage:
    python -m core.build_icbhi_manifest
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.io import wavfile
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "icbhi"
AUDIO_DIR = RAW_DIR / "audio"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

# Label ids as published in the mirror's dataset_info.
LABEL_NAMES = {
    0: "Asthma", 1: "Bronchiectasis", 2: "Bronchiolitis", 3: "COPD",
    4: "Healthy", 5: "LRTI", 6: "Pneumonia", 7: "URTI",
}

FNAME_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<rec>[^_]+)_(?P<loc>[^_]+)_(?P<mode>[^_]+)_(?P<dev>.+)\.wav$"
)

LOCATION_NAMES = {
    "Tc": "Trachea", "Al": "Anterior left", "Ar": "Anterior right",
    "Pl": "Posterior left", "Pr": "Posterior right",
    "Ll": "Lateral left", "Lr": "Lateral right",
}


def extract_shards() -> int:
    """Write every embedded wav out to data/raw/icbhi/audio/ under its real name."""
    shards = sorted(RAW_DIR.glob("shard*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards in {RAW_DIR}")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    for sh in shards:
        table = pq.read_table(sh)
        audio = table.column("audio").to_pylist()
        labels = table.column("label").to_pylist()
        for a, lab in zip(audio, labels):
            name = Path(a["path"]).name
            dest = AUDIO_DIR / name
            if not dest.exists():
                dest.write_bytes(a["bytes"])
            n += 1
        print(f"  {sh.name}: {len(labels)} rows")
    print(f"extracted {n} recordings -> {AUDIO_DIR}")
    return n


def label_map_from_shards() -> dict[str, int]:
    out = {}
    for sh in sorted(RAW_DIR.glob("shard*.parquet")):
        t = pq.read_table(sh)
        for a, lab in zip(t.column("audio").to_pylist(), t.column("label").to_pylist()):
            out[Path(a["path"]).name] = int(lab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--min-patients", type=int, default=10,
                    help="drop diagnoses with fewer patients than this")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument(
        "--device", default=None,
        help="restrict to ONE recording device (e.g. Meditron). ICBHI's devices are "
             "almost perfectly confounded with diagnosis -- 646/646 AKGC417L recordings "
             "are COPD -- so a single-device subset is the only way to measure pathology "
             "signal rather than device signal. Writes icbhi_<device>_manifest.csv.",
    )
    args = ap.parse_args()

    if not args.skip_extract:
        extract_shards()
    labels = label_map_from_shards()

    rows = []
    for p in sorted(AUDIO_DIR.glob("*.wav")):
        m = FNAME_RE.match(p.name)
        if not m:
            print(f"  WARNING unparsable filename: {p.name}")
            continue
        sr, d = wavfile.read(p)
        if d.ndim > 1:
            d = d[:, 0]
        g = m.groupdict()
        rows.append({
            "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "filename": p.name,
            "patient_id": f"P{int(g['pid']):03d}",
            "recording_index": g["rec"],
            "chest_location": g["loc"],
            "chest_location_name": LOCATION_NAMES.get(g["loc"], g["loc"]),
            "acquisition_mode": g["mode"],
            "device": g["dev"],
            "diagnosis": LABEL_NAMES[labels[p.name]],
            "sample_rate": int(sr),
            "n_samples": int(len(d)),
            "duration_s": round(len(d) / sr, 3),
            "pcm_md5": hashlib.md5(np.ascontiguousarray(d).tobytes()).hexdigest(),
        })
    df = pd.DataFrame(rows)
    print(f"\nparsed {len(df)} recordings from {df.patient_id.nunique()} patients")

    if args.device:
        # Keep patients whose recordings are ALL on the requested device, so the
        # subset has no within-patient device mixing either.
        per_pat = df.groupby("patient_id")["device"].nunique()
        single = set(per_pat[per_pat == 1].index)
        df = df[(df.device == args.device) & (df.patient_id.isin(single))].copy()
        print(f"restricted to device {args.device!r}: {len(df)} recordings, "
              f"{df.patient_id.nunique()} patients")

    print("\n--- recordings per diagnosis ---")
    print(df["diagnosis"].value_counts().to_string())
    print("\n--- PATIENTS per diagnosis ---")
    pat_dx = df.groupby("patient_id")["diagnosis"].first()
    print(pat_dx.value_counts().to_string())
    print("\n--- devices ---")
    print(df["device"].value_counts().to_string())
    print("\n--- chest locations ---")
    print(df["chest_location"].value_counts().to_string())
    print("\n--- sample rates ---")
    print(df["sample_rate"].value_counts().to_string())

    # duplicate PCM check (same rule as the other corpora)
    dupes = df[df.pcm_md5.duplicated(keep=False)]
    if len(dupes):
        conflict = dupes.groupby("pcm_md5")["diagnosis"].nunique()
        bad = conflict[conflict > 1]
        print(f"\nWARNING: {len(dupes)} duplicate-PCM recordings, "
              f"{len(bad)} groups with conflicting labels")
        if len(bad):
            df = df[~df.pcm_md5.isin(bad.index)]

    # keep only sufficiently represented classes
    counts = pat_dx.value_counts()
    keep = counts[counts >= args.min_patients].index.tolist()
    dropped = counts[counts < args.min_patients]
    print(f"\nkeeping classes with >= {args.min_patients} patients: {keep}")
    if len(dropped):
        print(f"dropping: {dropped.to_dict()}")
    df = df[df.diagnosis.isin(keep)].copy()

    classes = sorted(keep)
    df["label"] = df["diagnosis"].map({c: i for i, c in enumerate(classes)})

    # patient-disjoint stratified folds
    pats = df.groupby("patient_id")["diagnosis"].first().reset_index().sort_values("patient_id")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    pats["fold"] = -1
    for f, (_, te) in enumerate(skf.split(pats.patient_id, pats.diagnosis)):
        pats.loc[pats.index[te], "fold"] = f
    df = df.merge(pats[["patient_id", "fold"]], on="patient_id", how="left")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.device.lower()}" if args.device else ""
    out = MANIFEST_DIR / f"icbhi{suffix}_manifest.csv"
    df.sort_values(["patient_id", "filename"]).to_csv(out, index=False)

    print(f"\n=== ICBHI task: {len(df)} recordings, {df.patient_id.nunique()} patients, "
          f"{len(classes)} classes ===")
    print(f"classes: {classes}")
    print("\npatients per fold:")
    pf = df.groupby(["fold", "diagnosis"])["patient_id"].nunique().unstack(fill_value=0)
    print(pf.to_string())
    print(f"\ntotal audio: {df.duration_s.sum()/60:.1f} min")
    print(f"recordings per patient: median {df.groupby('patient_id').size().median():.0f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
