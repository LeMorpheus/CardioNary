"""Late fusion of heart sound CNN probabilities, feature vote, and metadata vote.

Task 5: Combines the votes into p_h = w1*p_CNN + w2*p_LR + w3*p_M (w3 <= 0.3).
Runs Comparison A (generalisation to PhysioNet) and Comparison B (metadata gain).
"""
import csv
import json
from pathlib import Path
from typing import Generator, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from core.contracts import FusedOutput
from cardio.eval.intervals import metric_intervals
from cardio.fusion.calibrate import calibrate_probabilities
from cardio.fusion.votes import new_vote
from cardio.model.final_model import CLASSES, FinalModel


def weight_grid(use_metadata: bool, step: float = 0.05, w3_max: float = 0.3) -> Generator[Tuple[float, float, float], None, None]:
    """Generate valid late fusion weight combinations summing to 1."""
    n = round(1 / step)
    for i in range(n + 1):
        for k in range(n + 1 - i):
            w1, w3 = round(i * step, 4), round(k * step, 4)
            if w3 > w3_max + 1e-9 or (k > 0 and not use_metadata):
                continue
            w2 = round(1.0 - w1 - w3, 4)
            if w2 < -1e-9:
                continue
            yield round(w1, 2), round(w2, 2), round(w3, 2)


def best_weights(
    y: np.ndarray,
    p_cnn: np.ndarray,
    p_lr: np.ndarray,
    p_m: Optional[np.ndarray] = None,
    step: float = 0.05,
    w3_max: float = 0.3,
) -> Tuple[float, float, float]:
    """Find weights minimizing log-loss."""
    use_meta = p_m is not None

    def loss(w: Tuple[float, float, float]) -> float:
        p = w[0] * p_cnn + w[1] * p_lr + (w[2] * p_m if use_meta else 0.0)
        return float(log_loss(y, np.clip(p, 1e-6, 1.0 - 1e-6)))

    return min(weight_grid(use_meta, step=step, w3_max=w3_max), key=loss)


def generate_physionet_cnn_probabilities() -> pd.DataFrame:
    """Run FinalModel over PhysioNet cached features if not already generated."""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "heart_cnn_physionet.csv"

    if out_csv.exists():
        return pd.read_csv(out_csv)

    print("Extracting CNN probabilities for PhysioNet from cached features...")
    model = FinalModel()
    rows = []
    with open("data/manifests/circor_manifest.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            with np.load(r["cache_path"]) as cache:
                feats = cache["features"]
                if len(feats) == 0:
                    continue
                p = np.asarray(model.float_model(model.normalise(feats), training=False))
                p = p.reshape(len(feats), -1).mean(axis=0)
                rows.append({
                    "patient_id": r["patient_id"],
                    "path": r["path"],
                    **{f"p_{k}": float(v) for k, v in zip(CLASSES, p)},
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} PhysioNet CNN probability rows to {out_csv}")
    return df


def run_late_fusion():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load calibration temperature T
    cal_file = reports_dir / "heart_calibration.json"
    if not cal_file.exists():
        raise FileNotFoundError("Missing heart_calibration.json. Run cardio.fusion.calibrate first.")
    with open(cal_file, "r", encoding="utf-8") as fh:
        T = json.load(fh)["final_T"]

    # 2. Fit weights on Yaseen
    oof_yaseen = pd.read_csv(reports_dir / "heart_oof_yaseen.csv")
    P_y = oof_yaseen.filter(like="p_").to_numpy(dtype=float)
    P_y = P_y / P_y.sum(axis=1, keepdims=True)
    P_y_cal = calibrate_probabilities(P_y, T)
    oof_yaseen["p_cnn"] = 1.0 - P_y_cal[:, 0]

    plr_yaseen = pd.read_csv(reports_dir / "heart_plr_yaseen.csv")

    # Join on patient_id (drops the 10 short clips, exactly per task guide)
    yaseen_merged = pd.merge(
        oof_yaseen[["patient_id", "fold", "label", "p_cnn"]],
        plr_yaseen[["patient_id", "p_lr"]],
        on="patient_id",
    )

    y_yaseen = (yaseen_merged["label"] != 0).astype(int).to_numpy()
    p_cnn_y = yaseen_merged["p_cnn"].to_numpy(dtype=float)
    p_lr_y = yaseen_merged["p_lr"].to_numpy(dtype=float)

    w_yaseen = best_weights(y_yaseen, p_cnn_y, p_lr_y)
    print(f"Fitted Yaseen weights (w1_cnn, w2_lr, w3_m): {w_yaseen}")

    # 3. Load PhysioNet data
    df_cnn_phys = generate_physionet_cnn_probabilities()
    P_phys = df_cnn_phys.filter(like="p_").to_numpy(dtype=float)
    P_phys = P_phys / P_phys.sum(axis=1, keepdims=True)
    P_phys_cal = calibrate_probabilities(P_phys, T)
    df_cnn_phys["p_cnn"] = 1.0 - P_phys_cal[:, 0]
    patient_cnn = df_cnn_phys.groupby("patient_id")["p_cnn"].mean().reset_index()

    # Fit feature model on all Yaseen and predict for PhysioNet
    feats_y = pd.read_csv(reports_dir / "heart_features_yaseen.csv")
    feature_cols = [c for c in feats_y.columns if c not in ["patient_id", "fold", "label"]]
    full_feat_model = new_vote().fit(
        feats_y[feature_cols].to_numpy(dtype=float),
        (feats_y["label"] != 0).astype(int).to_numpy(),
    )

    feats_phys = pd.read_csv(reports_dir / "heart_features_physionet.csv")
    patient_feats = feats_phys.groupby("patient_id")[feature_cols].mean().reset_index()
    patient_feats["p_lr"] = full_feat_model.predict_proba(
        patient_feats[feature_cols].to_numpy(dtype=float)
    )[:, 1]

    # Load metadata vote probabilities and patient folds
    df_pm_phys = pd.read_csv(reports_dir / "heart_pm_physionet.csv")

    # Merge on patient_id (816 patients)
    merged = pd.merge(df_pm_phys, patient_cnn[["patient_id", "p_cnn"]], on="patient_id")
    merged = pd.merge(merged, patient_feats[["patient_id", "p_lr"]], on="patient_id")

    y_phys = merged["label"].to_numpy(dtype=int)
    groups_phys = merged["patient_id"].to_numpy()
    p_cnn_phys = merged["p_cnn"].to_numpy(dtype=float)
    p_lr_phys = merged["p_lr"].to_numpy(dtype=float)
    p_m_phys = merged["p_m"].to_numpy(dtype=float)
    folds_phys = merged["fold"].to_numpy(dtype=int)

    # 4. Comparison A: Generalisation
    p_compA_plain = p_cnn_phys
    p_compA_fused = w_yaseen[0] * p_cnn_phys + w_yaseen[1] * p_lr_phys

    metrics_A_plain = metric_intervals(y_phys, p_compA_plain, groups_phys, ["macro_f1", "auroc"])
    metrics_A_fused = metric_intervals(y_phys, p_compA_fused, groups_phys, ["macro_f1", "auroc"])

    print("\n=== Comparison A: Generalisation (Yaseen -> PhysioNet) ===")
    print(f"Plain CNN AUROC: {metrics_A_plain['auroc']['estimate']:.4f} [{metrics_A_plain['auroc']['lower_95']:.4f}, {metrics_A_plain['auroc']['upper_95']:.4f}], F1: {metrics_A_plain['macro_f1']['estimate']:.4f}")
    print(f"Fused CNN+LR AUROC: {metrics_A_fused['auroc']['estimate']:.4f} [{metrics_A_fused['auroc']['lower_95']:.4f}, {metrics_A_fused['auroc']['upper_95']:.4f}], F1: {metrics_A_fused['macro_f1']['estimate']:.4f}")

    # 5. Comparison B: Metadata Gain (Internal to PhysioNet across 5 folds)
    p_oof_2vote = np.zeros(len(merged))
    p_oof_3vote = np.zeros(len(merged))
    w_folds_2vote = []
    w_folds_3vote = []

    for k in range(5):
        train = folds_phys != k
        test = folds_phys == k
        w2 = best_weights(y_phys[train], p_cnn_phys[train], p_lr_phys[train])
        w3 = best_weights(y_phys[train], p_cnn_phys[train], p_lr_phys[train], p_m_phys[train])
        w_folds_2vote.append(w2)
        w_folds_3vote.append(w3)
        p_oof_2vote[test] = w2[0] * p_cnn_phys[test] + w2[1] * p_lr_phys[test]
        p_oof_3vote[test] = w3[0] * p_cnn_phys[test] + w3[1] * p_lr_phys[test] + w3[2] * p_m_phys[test]

    metrics_B_2vote = metric_intervals(y_phys, p_oof_2vote, groups_phys, ["auroc"])
    metrics_B_3vote = metric_intervals(y_phys, p_oof_3vote, groups_phys, ["auroc"])
    gain_B = float(metrics_B_3vote["auroc"]["estimate"] - metrics_B_2vote["auroc"]["estimate"])

    print("\n=== Comparison B: Metadata Gain ===")
    print(f"CNN + Feature Vote AUROC: {metrics_B_2vote['auroc']['estimate']:.4f} [{metrics_B_2vote['auroc']['lower_95']:.4f}, {metrics_B_2vote['auroc']['upper_95']:.4f}]")
    print(f"CNN + Feature + Metadata Vote AUROC: {metrics_B_3vote['auroc']['estimate']:.4f} [{metrics_B_3vote['auroc']['lower_95']:.4f}, {metrics_B_3vote['auroc']['upper_95']:.4f}]")
    print(f"Metadata AUROC Gain: {gain_B:+.4f} (Target >= +0.02)")

    # Contract verification: ensure sample produces valid FusedOutput
    sample_fused = FusedOutput(
        modality="heart",
        p_abnormal=float(p_oof_3vote[0]),
        votes={
            "cnn": float(p_cnn_phys[0]),
            "features": float(p_lr_phys[0]),
            "metadata": float(p_m_phys[0]),
        },
        weights={
            "cnn": w_folds_3vote[0][0],
            "features": w_folds_3vote[0][1],
            "metadata": w_folds_3vote[0][2],
        },
    )

    out_data = {
        "yaseen_weights": list(w_yaseen),
        "comparison_A": {
            "plain_cnn": metrics_A_plain,
            "fused": metrics_A_fused,
        },
        "comparison_B": {
            "two_vote_weights_per_fold": [list(w) for w in w_folds_2vote],
            "three_vote_weights_per_fold": [list(w) for w in w_folds_3vote],
            "two_vote": metrics_B_2vote,
            "three_vote": metrics_B_3vote,
            "auroc_gain": gain_B,
        },
    }

    out_path = reports_dir / "heart_fusion.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"\nSaved late fusion report to {out_path}")

    return out_data


def main():
    run_late_fusion()


if __name__ == "__main__":
    main()

