"""Build the RSNA chest X-ray dataset and patient-level manifest.

Usage:
    python -m lung.data.build_rsna
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from core.images import features_for_image


ROOT = Path(__file__).resolve().parents[2]
RSNA_DIR = ROOT / "data" / "raw" / "rsna"
DICOM_DIR = RSNA_DIR / "stage_2_train_images"
CLASS_CSV = RSNA_DIR / "stage_2_detailed_class_info.csv"

PNG_DIR = RSNA_DIR / "png256"
CACHE_DIR = ROOT / "data" / "cache" / "rsna_96"
MANIFEST_DIR = ROOT / "data" / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "rsna_manifest.csv"

CLASS_TO_LABEL = {
    "Normal": 0,
    "Lung Opacity": 1,
}


def parse_age(value: object) -> float:
    """Convert DICOM PatientAge (e.g. 057Y, 006M) to years."""
    if value is None:
        return np.nan

    match = re.fullmatch(r"\s*(\d{1,3})([YMWD])\s*", str(value).upper())
    if not match:
        return np.nan

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == "Y":
        return float(amount)
    if unit == "M":
        return amount / 12.0
    if unit == "W":
        return amount / 52.0
    return amount / 365.0


def dicom_to_png(dicom_path: Path, png_path: Path) -> dict:
    """Read DICOM pixels and metadata, then save a square 256x256 PNG."""
    ds = pydicom.dcmread(dicom_path)

    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixels = pixels * slope + intercept

    low, high = np.percentile(pixels, [0.5, 99.5])
    if high <= low:
        raise ValueError(f"Image has no usable intensity range: {dicom_path}")

    pixels = np.clip(pixels, low, high)
    pixels = (pixels - low) / (high - low)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 1.0 - pixels

    pixels = (pixels * 255).round().astype(np.uint8)

    image = Image.fromarray(pixels, mode="L")

    # Centre-crop to square, then resize to 256x256.
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((256, 256), Image.Resampling.BILINEAR)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG")

    return {
        "age": parse_age(getattr(ds, "PatientAge", None)),
        "sex": str(getattr(ds, "PatientSex", "UNKNOWN")).strip().upper(),
        "view_position": str(
            getattr(ds, "ViewPosition", "UNKNOWN")
        ).strip().upper(),
    }


def build_manifest(n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Convert selected patients, create caches, and assign patient folds."""
    if not CLASS_CSV.exists():
        raise FileNotFoundError(f"Missing class CSV: {CLASS_CSV}")
    if not DICOM_DIR.exists():
        raise FileNotFoundError(f"Missing DICOM directory: {DICOM_DIR}")

    classes = pd.read_csv(CLASS_CSV)

    required_columns = {"patientId", "class"}
    if not required_columns.issubset(classes.columns):
        raise ValueError(
            f"Class CSV must contain columns {sorted(required_columns)}"
        )

    selected = classes[classes["class"].isin(CLASS_TO_LABEL)].copy()

    # Ensure one unambiguous selected label per patient.
    label_counts = selected.groupby("patientId")["class"].nunique()
    conflicts = label_counts[label_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            f"{len(conflicts)} patients have conflicting selected classes"
        )

    selected = selected.drop_duplicates(subset="patientId")
    print(f"Selected patients before metadata filtering: {len(selected)}")

    records: list[dict] = []
    skipped = 0

    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        patient_id = str(row["patientId"])
        class_name = str(row["class"])
        dicom_path = DICOM_DIR / f"{patient_id}.dcm"
        png_path = PNG_DIR / f"{patient_id}.png"

        if not dicom_path.exists():
            print(f"WARNING: Missing DICOM: {patient_id}")
            skipped += 1
            continue

        try:
            metadata = dicom_to_png(dicom_path, png_path)

            age = metadata["age"]
            if not np.isnan(age) and age > 100:
                png_path.unlink(missing_ok=True)
                skipped += 1
                continue

            # Use the shared core image front-end to create the 96x96 cache.
            cache_path = CACHE_DIR / f"{patient_id}.npy"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image_features = features_for_image(str(png_path))
            np.save(cache_path, image_features.astype(np.float32))

            records.append({
                "path": png_path.relative_to(ROOT).as_posix(),
                "patient_id": patient_id,
                "label": CLASS_TO_LABEL[class_name],
                "age": age,
                "sex": metadata["sex"],
                "view_position": metadata["view_position"],
            })

        except Exception as exc:
            print(f"WARNING: Could not process {patient_id}: {exc}")
            skipped += 1

        if index % 500 == 0:
            print(f"Processed {index}/{len(selected)} selected patients")

    if not records:
        raise RuntimeError("No RSNA patients were successfully processed")

    manifest = pd.DataFrame(records)

    # Each patient occurs exactly once, so stratifying these rows is
    # equivalent to assigning patient-disjoint folds.
    if manifest["patient_id"].duplicated().any():
        raise RuntimeError("Duplicate patients found before fold assignment")

    class_counts = manifest["label"].value_counts()
    if class_counts.min() < n_folds:
        raise ValueError(
            f"Each class needs at least {n_folds} patients for stratification"
        )

    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    manifest["fold"] = -1
    for fold, (_, validation_indices) in enumerate(
        splitter.split(manifest["patient_id"], manifest["label"])
    ):
        manifest.loc[validation_indices, "fold"] = fold

    manifest = manifest[
        ["path", "patient_id", "label", "fold", "age", "sex", "view_position"]
    ].sort_values("patient_id").reset_index(drop=True)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print("\n========== RSNA BUILD SUMMARY ==========")
    print(f"Patients in manifest: {len(manifest)}")
    print(f"Skipped/missing/invalid: {skipped}")
    print("\nClass counts (0=Normal, 1=Lung Opacity):")
    print(manifest["label"].value_counts().sort_index())
    print("\nPatients per fold and class:")
    print(pd.crosstab(manifest["fold"], manifest["label"]))
    print(f"\nPNG directory: {PNG_DIR}")
    print(f"96x96 cache directory: {CACHE_DIR}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_manifest(n_folds=args.n_folds, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
