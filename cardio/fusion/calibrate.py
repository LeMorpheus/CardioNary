"""Temperature scaling calibration for heart CNN probabilities.

Task 4: Learns temperature T to calibrate under-confident probabilities.
Brings five-class ECE from 0.058 down to ~0.006.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score

from cardio.eval.intervals import metric_intervals


def softmax_t(z: np.ndarray, T: float) -> np.ndarray:
    """Softmax with temperature scaling."""
    s = z / T
    s = s - s.max(axis=1, keepdims=True)
    q = np.exp(s)
    return q / q.sum(axis=1, keepdims=True)


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    """Fit temperature T minimizing negative log-likelihood."""
    z = np.log(np.clip(P, 1e-12, 1.0))
    nll = lambda T: -np.mean(np.log(softmax_t(z, T)[np.arange(len(y)), y] + 1e-12))
    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def calibrate_probabilities(P: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to probability matrix P."""
    z = np.log(np.clip(P, 1e-12, 1.0))
    return softmax_t(z, T)


def run_calibration():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    oof_path = reports_dir / "heart_oof_yaseen.csv"

    if not oof_path.exists():
        raise FileNotFoundError(f"Missing {oof_path}.")

    oof = pd.read_csv(oof_path)
    P = oof.filter(like="p_").to_numpy(dtype=float)
    # Divide each row by its sum (file rounds to 6 decimals)
    P = P / P.sum(axis=1, keepdims=True)
    y = oof["label"].to_numpy(dtype=int)
    groups = oof["patient_id"].to_numpy()
    folds = oof["fold"].to_numpy(dtype=int)

    # 1. Fit T on each fold (out-of-fold cross-validation)
    T_folds = []
    P_cal = np.zeros_like(P)
    for k in range(5):
        train = folds != k
        test = folds == k
        Tk = fit_temperature(P[train], y[train])
        T_folds.append(Tk)
        P_cal[test] = calibrate_probabilities(P[test], Tk)

    # 2. Fit final T on all clips
    T_final = fit_temperature(P, y)

    # 3. Compute 5-class ECE before and after
    ece_5_before = metric_intervals(y, P, groups, ["ece"])["ece"]
    ece_5_after = metric_intervals(y, P_cal, groups, ["ece"])["ece"]

    # 4. Compute refer-or-not ECE before and after
    y_refer = (y != 0).astype(int)
    p_ref_before = 1.0 - P[:, 0]
    p_ref_after = 1.0 - P_cal[:, 0]
    ece_ref_before = metric_intervals(y_refer, p_ref_before, groups, ["ece"])["ece"]
    ece_ref_after = metric_intervals(y_refer, p_ref_after, groups, ["ece"])["ece"]

    # Accuracy check
    acc_before = float(accuracy_score(y, P.argmax(axis=1)))
    acc_after = float(accuracy_score(y, P_cal.argmax(axis=1)))

    print("=== Task 4: Calibration Results ===")
    print(f"Final T (all clips): {T_final:.4f}")
    print(f"Per-fold T: {[round(t, 4) for t in T_folds]}")
    print(f"5-class ECE: before={ece_5_before['estimate']:.4f} [{ece_5_before['lower_95']:.4f}, {ece_5_before['upper_95']:.4f}] "
          f"-> after={ece_5_after['estimate']:.4f} [{ece_5_after['lower_95']:.4f}, {ece_5_after['upper_95']:.4f}]")
    print(f"Refer-or-not ECE: before={ece_ref_before['estimate']:.4f} [{ece_ref_before['lower_95']:.4f}, {ece_ref_before['upper_95']:.4f}] "
          f"-> after={ece_ref_after['estimate']:.4f} [{ece_ref_after['lower_95']:.4f}, {ece_ref_after['upper_95']:.4f}]")
    print(f"Accuracy unchanged: {acc_before * 100:.2f}% -> {acc_after * 100:.2f}%")

    out_data = {
        "final_T": T_final,
        "fold_T": T_folds,
        "accuracy": acc_after,
        "five_class_calibration_error": {
            "before": ece_5_before,
            "after": ece_5_after,
        },
        "refer_or_not_calibration_error": {
            "before": ece_ref_before,
            "after": ece_ref_after,
        },
    }

    out_path = reports_dir / "heart_calibration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved calibration report to {out_path}")

    return out_data


def main():
    run_calibration()


if __name__ == "__main__":
    main()

