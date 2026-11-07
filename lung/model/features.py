"""Extract classical and CLAHE image features from the Kermany chest X-ray cache.

Computes numeric descriptors from core.descriptors.vision_descriptors plus
three CLAHE texture heterogeneity (standard deviation) features across clip
limits 1.0, 2.0, and 4.0. Text fields (pattern, distribution, predominant zone)
are excluded.

Output:
    reports/xray_features_kermany.csv

Usage:
    python -m lung.model.features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core.descriptors import vision_descriptors

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "cache" / "kermany_logmel.npz"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "kermany_manifest.csv"
OUTPUT_PATH = REPO_ROOT / "reports" / "xray_features_kermany.csv"


def extract_features(
    cache_path: Path = CACHE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Extract numeric vision and CLAHE descriptors for every Kermany image."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found at {cache_path}. Run core.preprocess first.")

    print(f"Loading image cache from {cache_path}...")
    cache = np.load(cache_path)
    X = cache["X"]  # shape (N, 96, 96) float32 in [0, 1]
    n_images = len(X)
    print(f"Loaded {n_images} images.")

    # Match filenames and metadata if manifest is available
    if manifest_path.exists():
        df_manifest = pd.read_csv(manifest_path)
        filenames = df_manifest["filename"].tolist() if "filename" in df_manifest.columns else [f"img_{i:05d}" for i in range(n_images)]
        diagnoses = df_manifest["diagnosis"].tolist() if "diagnosis" in df_manifest.columns else None
    else:
        filenames = [f"img_{i:05d}" for i in range(n_images)]
        diagnoses = None

    patient_ids = cache["patient_id"]
    labels = cache["y"]
    folds = cache["fold"]

    # Pre-instantiate CLAHE filters for clip limits 1.0, 2.0, and 4.0
    clahe_filters = {
        "clahe_std_1": cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)),
        "clahe_std_2": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)),
        "clahe_std_4": cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)),
    }

    # Pre-compute thoracic ROI mask for 96x96 image
    h, w = X[0].shape
    b = int(0.08 * h)
    roi = np.zeros((h, w), dtype=bool)
    roi[b : h - b, b : w - b] = True
    roi[:, int(0.44 * w) : int(0.56 * w)] = False

    rows = []
    print("Computing numeric vision descriptors and CLAHE numbers...")
    for i in range(n_images):
        img = X[i]

        # 1. Classical numeric descriptors from core.descriptors
        desc = vision_descriptors(img)
        zones = desc.get("zone_distribution_pct", {})

        # 2. CLAHE texture heterogeneity at clip limits 1.0, 2.0, 4.0
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        clahe_vals = {
            k: round(float(np.std(filt.apply(u8)[roi])), 1)
            for k, filt in clahe_filters.items()
        }

        row = {
            "filename": filenames[i],
            "patient_id": str(patient_ids[i]),
            "label": int(labels[i]),
            "fold": int(folds[i]),
            "dense_area_pct": desc["dense_area_pct"],
            "asymmetry_index": desc["asymmetry_index"],
            "zone_upper_pct": zones.get("upper", 0.0),
            "zone_middle_pct": zones.get("middle", 0.0),
            "zone_lower_pct": zones.get("lower", 0.0),
            "largest_region_fraction": desc["largest_region_fraction"],
            "regions_detected": desc["regions_detected"],
            "mean_intensity": desc["mean_intensity"],
            "texture_heterogeneity": desc["texture_heterogeneity"],
            "clahe_std_1": clahe_vals["clahe_std_1"],
            "clahe_std_2": clahe_vals["clahe_std_2"],
            "clahe_std_4": clahe_vals["clahe_std_4"],
        }
        if diagnoses is not None:
            row["diagnosis"] = diagnoses[i]

        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Successfully computed features for {len(out_df)} images.")
    print(f"Output saved to: {output_path}")

    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH, help="Path to kermany_logmel.npz")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to kermany_manifest.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    extract_features(cache_path=args.cache, manifest_path=args.manifest, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
