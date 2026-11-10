"""Metadata vote for RSNA chest X-ray dataset.

Trains a logistic regression model on patient metadata (age, sex) using
5-fold cross-validation on the RSNA dataset.  View position is deliberately
excluded: it describes the imaging equipment, not the patient.

Features used:
    a   = age / 100
    a²  = (age / 100) **2
    sex_male = 1 if sex == 'M', else 0

When age is missing (NaN) the two age features are imputed with the
training-fold mean (strict anti-leakage: imputation mean is computed on
each fold's training split and applied to the test split).

Output:
    reports/xray_pm_rsna.csv    -- image-level pM (out-of-fold)
    reports/xray_pm_rsna.json   -- AUROC summary

Usage:
    python -m lung.fusion.meta_vote
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "rsna_manifest.csv"
OUT_CSV = REPO_ROOT / "reports" / "xray_pm_rsna.csv"
OUT_JSON = REPO_ROOT / "reports" / "xray_pm_rsna.json"


def _build_features(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 3) feature matrix: [a, a^2, sex_male].

    ``age`` is expected to be in years; NaN values are carried through for
    the caller to impute per-fold.
    """
    a = df["age"].to_numpy(dtype=float) / 100.0
    a2 = a ** 2
    sex_male = (df["sex"].str.upper() == "M").astype(float).to_numpy()
    return np.column_stack([a, a2, sex_male])


def run_meta_vote(
    manifest_csv: Path | None = None,
    out_csv: Path | None = None,
    out_json: Path | None = None,
) -> pd.DataFrame:
    """Run 5-fold metadata logistic regression on RSNA and return OOF DataFrame.

    Parameters
    ----------
    manifest_csv : path to rsna_manifest.csv (defaults to repo default)
    out_csv      : where to write xray_pm_rsna.csv
    out_json     : where to write the AUROC summary JSON

    Returns
    -------
    DataFrame with columns [index, patient_id, fold, label, pM]
    """
    manifest_csv = Path(manifest_csv or MANIFEST_PATH)
    out_csv = Path(out_csv or OUT_CSV)
    out_json = Path(out_json or OUT_JSON)

    df = pd.read_csv(manifest_csv)

    required = {"patient_id", "label", "fold", "sex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rsna_manifest.csv is missing columns: {missing}")

    # 'age' may be absent entirely or all-NaN; either way treat as NaN
    if "age" not in df.columns:
        df["age"] = np.nan

    folds = sorted(df["fold"].unique())
    if len(folds) != 5:
        raise ValueError(f"Expected 5 folds in RSNA manifest, found {folds}")

    X_all = _build_features(df)  # (N, 3); age columns may be NaN
    y_all = df["label"].to_numpy(dtype=int)

    oof_rows = []

    for f in folds:
        tr_mask = df["fold"].to_numpy() != f
        te_mask = df["fold"].to_numpy() == f

        X_tr, X_te = X_all[tr_mask].copy(), X_all[te_mask].copy()
        y_tr = y_all[tr_mask]

        # --- Impute NaN age features using training-fold mean (anti-leakage) ---
        for col_idx in (0, 1):  # a and a^2 columns
            col_tr = X_tr[:, col_idx]
            nan_mask_tr = np.isnan(col_tr)
            if nan_mask_tr.all():
                # All NaN -> zero-fill (feature carries no information)
                X_tr[:, col_idx] = 0.0
                X_te[:, col_idx] = 0.0
            else:
                col_mean = col_tr[~nan_mask_tr].mean()
                X_tr[nan_mask_tr, col_idx] = col_mean
                X_te[np.isnan(X_te[:, col_idx]), col_idx] = col_mean

        # --- Standardise on training fold only (anti-leakage) ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        lr.fit(X_tr_s, y_tr)
        p_te = lr.predict_proba(X_te_s)[:, 1]  # P(abnormal)

        te_indices = np.where(te_mask)[0]
        for idx, row_i, pm in zip(te_indices, df[te_mask].itertuples(), p_te):
            oof_rows.append({
                "index": int(idx),
                "patient_id": str(row_i.patient_id),
                "fold": int(f),
                "label": int(row_i.label),
                "pM": float(pm),
            })

    df_oof = pd.DataFrame(oof_rows).sort_values("index").reset_index(drop=True)

    # --- Compute AUROC ---
    auroc_image = roc_auc_score(df_oof["label"], df_oof["pM"])

    # Patient-level: mean pM per patient
    patient_grp = df_oof.groupby("patient_id").agg(
        label=("label", "first"),
        pM=("pM", "mean"),
    )
    auroc_patient = roc_auc_score(patient_grp["label"], patient_grp["pM"])

    # --- Save ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_oof.to_csv(out_csv, index=False)

    summary = {
        "n_images": int(len(df_oof)),
        "n_patients": int(len(patient_grp)),
        "auroc_image_level": round(auroc_image, 4),
        "auroc_patient_level": round(auroc_patient, 4),
    }
    out_json.write_text(json.dumps(summary, indent=2))

    return df_oof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="RSNA manifest CSV")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        return 1

    print(f"Running metadata vote on {args.manifest} ...")
    df_oof = run_meta_vote(args.manifest, args.out_csv, args.out_json)

    summary = json.loads(args.out_json.read_text())
    print(f"  Images: {summary['n_images']}  |  Patients: {summary['n_patients']}")
    print(f"  Image-level AUROC:   {summary['auroc_image_level']:.4f}")
    print(f"  Patient-level AUROC: {summary['auroc_patient_level']:.4f}")
    print(f"  Wrote {args.out_csv}")
    print(f"  Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
