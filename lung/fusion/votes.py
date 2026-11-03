"""Feature vote: standardise and fit logistic regression on Kermany 5-fold CV.

Computes the classical 'features' vote for late fusion (core.contracts.VOTES):
standardises the classical + CLAHE image descriptors fold-by-fold on training
data only (preventing leakage), fits a logistic regression classifier, and
produces out-of-fold predicted probabilities pLR for every Kermany image.

Output:
    reports/xray_plr_kermany.csv
    reports/xray_plr_kermany.json

Usage:
    python -m lung.fusion.votes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = REPO_ROOT / "reports" / "xray_features_kermany.csv"
OUTPUT_CSV = REPO_ROOT / "reports" / "xray_plr_kermany.csv"
OUTPUT_JSON = REPO_ROOT / "reports" / "xray_plr_kermany.json"

FEATURE_COLUMNS = [
    "dense_area_pct",
    "asymmetry_index",
    "zone_upper_pct",
    "zone_middle_pct",
    "zone_lower_pct",
    "largest_region_fraction",
    "regions_detected",
    "mean_intensity",
    "texture_heterogeneity",
    "clahe_std_1",
    "clahe_std_2",
    "clahe_std_4",
]


def run_feature_vote(
    features_path: Path = FEATURES_CSV,
    output_csv: Path = OUTPUT_CSV,
    output_json: Path = OUTPUT_JSON,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Train out-of-fold logistic regression on standardised image features."""
    if not features_path.exists():
        print(f"Features CSV not found at {features_path}. Extracting features first...")
        from lung.model.features import extract_features
        df = extract_features(output_path=features_path)
    else:
        df = pd.read_csv(features_path)

    # Verify required feature columns are present
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Features file is missing required columns: {missing_cols}")

    folds = sorted(df["fold"].unique())
    n_samples = len(df)
    oof_pLR = np.zeros(n_samples, dtype=np.float64)
    per_fold_metrics = []

    print(f"=== Kermany Feature Vote (Logistic Regression) ===")
    print(f"Total samples: {n_samples}, Folds: {len(folds)}")
    print(f"Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}\n")

    for f in folds:
        train_mask = df["fold"] != f
        test_mask = df["fold"] == f

        X_train = df.loc[train_mask, FEATURE_COLUMNS].to_numpy()
        y_train = df.loc[train_mask, "label"].to_numpy()
        X_test = df.loc[test_mask, FEATURE_COLUMNS].to_numpy()
        y_test = df.loc[test_mask, "label"].to_numpy()

        # Rule 5.5: Standardisation statistics fitted on TRAINING fold only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit Logistic Regression on training fold
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        clf.fit(X_train_scaled, y_train)

        # Predict probability of class 1 (Pneumonia / abnormal)
        probs_test = clf.predict_proba(X_test_scaled)[:, 1]
        oof_pLR[test_mask] = probs_test

        fold_auc = roc_auc_score(y_test, probs_test)
        preds_binary = (probs_test >= 0.5).astype(int)
        fold_acc = accuracy_score(y_test, preds_binary)
        fold_f1 = f1_score(y_test, preds_binary, average="macro")

        per_fold_metrics.append({
            "fold": int(f),
            "n_test": int(test_mask.sum()),
            "auroc": float(fold_auc),
            "accuracy": float(fold_acc),
            "macro_f1": float(fold_f1),
        })
        print(f"  Fold {f}: AUROC = {fold_auc:.4f}, Accuracy = {fold_acc:.4f}, Macro-F1 = {fold_f1:.4f}")

    y_all = df["label"].to_numpy()
    pooled_auroc = roc_auc_score(y_all, oof_pLR)
    fold_auroc_mean = float(np.mean([m["auroc"] for m in per_fold_metrics]))
    fold_auroc_std = float(np.std([m["auroc"] for m in per_fold_metrics]))

    # Patient-level aggregation: clinician gets one assessment per patient
    pat_df = df.copy()
    pat_df["pLR"] = oof_pLR
    patient_table = pat_df.groupby("patient_id", sort=False).agg(
        label=("label", "first"),
        pLR_patient=("pLR", "mean"),
    ).reset_index()
    patient_auroc = roc_auc_score(patient_table["label"], patient_table["pLR_patient"])

    print(f"\n--- AUROC Summary ---")
    print(f"Image-Level Pooled AUROC: {pooled_auroc:.4f}")
    print(f"Per-Fold AUROC (mean ± std): {fold_auroc_mean:.4f} ± {fold_auroc_std:.4f}")
    print(f"Patient-Level Pooled AUROC: {patient_auroc:.4f}")

    # Build output DataFrame with out-of-fold pLR
    oof_df = pd.DataFrame({
        "filename": df["filename"] if "filename" in df.columns else [f"img_{i:05d}" for i in range(n_samples)],
        "patient_id": df["patient_id"],
        "fold": df["fold"],
        "label": df["label"],
        "pLR": np.round(oof_pLR, 6),
        "p_Normal": np.round(1.0 - oof_pLR, 6),
        "p_Pneumonia": np.round(oof_pLR, 6),
    })
    if "diagnosis" in df.columns:
        oof_df.insert(4, "diagnosis", df["diagnosis"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(output_csv, index=False)
    print(f"\nWrote out-of-fold predictions to: {output_csv}")

    # Write summary metrics JSON
    summary = {
        "model": "logistic_regression",
        "vote": "features",
        "modality": "xray",
        "dataset": "kermany",
        "features": FEATURE_COLUMNS,
        "n_samples": n_samples,
        "n_patients": len(patient_table),
        "image_level_auroc": float(pooled_auroc),
        "patient_level_auroc": float(patient_auroc),
        "fold_auroc_mean": fold_auroc_mean,
        "fold_auroc_std": fold_auroc_std,
        "per_fold": per_fold_metrics,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary metrics to: {output_json}")

    return oof_df, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_CSV, help="Path to xray_features_kermany.csv")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="Path to write xray_plr_kermany.csv")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON, help="Path to write xray_plr_kermany.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for logistic regression")
    args = parser.parse_args()

    run_feature_vote(
        features_path=args.features,
        output_csv=args.output_csv,
        output_json=args.output_json,
        random_state=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
