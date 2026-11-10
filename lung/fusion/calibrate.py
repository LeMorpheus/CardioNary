"""Temperature scaling calibration for model probabilities.

Fits a single temperature scalar T by minimising the negative log-likelihood (NLL)
over log-probabilities (z = log p) using scipy.optimize.minimize_scalar.

Over-confident models (probabilities skewed too close to 0/1 relative to empirical accuracy)
produce T > 1.0, softening probabilities towards the uniform distribution.
Under-confident models produce T < 1.0, sharpening probabilities towards 0/1.

Usage:
    python -m lung.fusion.calibrate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLR_CSV = REPO_ROOT / "reports" / "xray_plr_kermany.csv"
DEFAULT_OUT_JSON = REPO_ROOT / "reports" / "xray_calibration_kermany.json"
DEFAULT_OOF_CSV = REPO_ROOT / "reports" / "xray_oof_kermany.csv"
DEFAULT_CNN_OUT_JSON = REPO_ROOT / "reports" / "xray_calibration_cnn_kermany.json"

EPS = 1e-12


def _to_class_probs(probs: np.ndarray) -> np.ndarray:
    """Ensure probs is a 2D array of class probabilities summing to 1."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        # Binary probabilities: treat as [p(y=0), p(y=1)]
        p = np.clip(p, EPS, 1.0 - EPS)
        return np.column_stack([1.0 - p, p])
    elif p.ndim == 2:
        p = np.clip(p, EPS, 1.0)
        return p / p.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Expected 1D or 2D array of probabilities, got shape {p.shape}")


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (0.01, 10.0),
) -> float:
    """Fit a single temperature T by minimising negative log-likelihood on z = log p.

    Parameters:
        probs: 1D array of P(y=1) or 2D array of class probabilities (N, K).
        labels: 1D array of integer class labels in {0, ..., K-1}.
        bounds: (min_T, max_T) optimization search range.

    Returns:
        Fitted temperature T > 0.
    """
    p = _to_class_probs(probs)
    y = np.asarray(labels, dtype=int)
    n_samples, n_classes = p.shape

    if len(y) != n_samples:
        raise ValueError(f"Mismatch: {n_samples} probability rows vs {len(y)} labels")

    # z = log p
    z = np.log(p)
    rows = np.arange(n_samples)

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        # Scaled logits: z / T
        logits = z / T
        # Log-softmax = logits - logsumexp(logits)
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        # NLL = - mean(log p_true)
        loss = -np.mean(log_probs[rows, y])
        return float(loss)

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    if not res.success:
        raise RuntimeError(f"Temperature optimization failed: {res.message}")

    return float(res.x)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrate probabilities using temperature T: p_cal = softmax(log(p) / T).

    Parameters:
        probs: 1D or 2D array of probabilities.
        temperature: Scalar temperature T > 0.

    Returns:
        Calibrated probabilities with same shape as input.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    p = np.asarray(probs, dtype=np.float64)
    was_1d = (p.ndim == 1)
    p_2d = _to_class_probs(p)

    # z = log p, scale by 1 / T, then softmax
    z = np.log(p_2d)
    logits = z / temperature
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    calibrated = np.exp(log_probs)

    # Return 1D if input was 1D binary probability
    if was_1d:
        return calibrated[:, 1]
    return calibrated


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute the multi-class negative log-likelihood."""
    p = _to_class_probs(probs)
    y = np.asarray(labels, dtype=int)
    log_p = np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))
    return float(-np.mean(log_p))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) for binary or multi-class predictions."""
    p = _to_class_probs(probs)
    y = np.asarray(labels, dtype=int)

    # Take maximum predicted probability and whether it was correct
    confidences = np.max(p, axis=1)
    predictions = np.argmax(p, axis=1)
    accuracies = (predictions == y).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def test_overconfident_probabilities() -> dict:
    """Verification test on synthetic data: over-confident probabilities must give T > 1."""
    # Synthetic scenario:
    # True labels have accuracy ~75% (25% errors).
    # But model outputs over-confident probabilities near 0.99 or 0.01.
    rng = np.random.default_rng(1337)
    n = 2000

    y_true = rng.integers(0, 2, size=n)

    # Flip 25% of labels to create ~75% accuracy
    is_correct = rng.random(size=n) < 0.75

    # Over-confident: when predicting 1, output 0.98; when predicting 0, output 0.02
    predicted_class = np.where(is_correct, y_true, 1 - y_true)
    overconfident_p = np.where(predicted_class == 1, 0.98, 0.02)

    T_overconfident = fit_temperature(overconfident_p, y_true)
    assert T_overconfident > 1.0, f"Expected T > 1 for over-confident model, got {T_overconfident:.4f}"

    # Also verify under-confident scenario:
    # 100% accurate, but outputs timid probabilities near 0.55 / 0.45
    underconfident_p = np.where(y_true == 1, 0.55, 0.45)
    T_underconfident = fit_temperature(underconfident_p, y_true)
    assert T_underconfident < 1.0, f"Expected T < 1 for under-confident model, got {T_underconfident:.4f}"

    # Well-calibrated scenario:
    # Probabilities match actual class frequencies (e.g. 0.8 accuracy with 0.8 confidence)
    well_calibrated_p = np.where(predicted_class == 1, 0.75, 0.25)
    T_wellcalibrated = fit_temperature(well_calibrated_p, y_true)
    assert 0.85 <= T_wellcalibrated <= 1.15, f"Expected T ~ 1 for well-calibrated model, got {T_wellcalibrated:.4f}"

    return {
        "overconfident_T": round(T_overconfident, 4),
        "underconfident_T": round(T_underconfident, 4),
        "well_calibrated_T": round(T_wellcalibrated, 4),
    }


def calibrate_cnn_oof(
    oof_csv: Path | None = None,
    out_json: Path | None = None,
) -> dict:
    """Fit T on CNN out-of-fold predictions (xray_oof_kermany.csv).

    Uses p_pneumonia as P(abnormal).  Returns a dict with before/after metrics.
    """
    oof_csv = Path(oof_csv or DEFAULT_OOF_CSV)
    out_json = Path(out_json or DEFAULT_CNN_OUT_JSON)

    df = pd.read_csv(oof_csv)
    y = df["label"].to_numpy()
    # p_pneumonia == P(abnormal)
    p_cnn = df["p_pneumonia"].to_numpy()

    nll_raw = negative_log_likelihood(p_cnn, y)
    ece_raw = expected_calibration_error(p_cnn, y)

    T = fit_temperature(p_cnn, y)
    p_cal = apply_temperature(p_cnn, T)

    nll_cal = negative_log_likelihood(p_cal, y)
    ece_cal = expected_calibration_error(p_cal, y)

    result = {
        "n_samples": int(len(df)),
        "temperature_T": round(T, 4),
        "nll_before": round(nll_raw, 4),
        "nll_after": round(nll_cal, 4),
        "ece_before_pct": round(ece_raw * 100, 2),
        "ece_after_pct": round(ece_cal * 100, 2),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plr-csv", type=Path, default=DEFAULT_PLR_CSV, help="Path to xray_plr_kermany.csv")
    parser.add_argument("--oof-csv", type=Path, default=DEFAULT_OOF_CSV, help="Path to CNN OOF CSV (xray_oof_kermany.csv)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_JSON, help="Path to output summary JSON")
    args = parser.parse_args()

    print("=== Testing Temperature Scaling on Synthetic Probabilities ===")
    test_results = test_overconfident_probabilities()
    print(f"  Over-confident probabilities:   T = {test_results['overconfident_T']}  (T > 1 confirmed)")
    print(f"  Under-confident probabilities:  T = {test_results['underconfident_T']}  (T < 1 confirmed)")
    print(f"  Well-calibrated probabilities:  T = {test_results['well_calibrated_T']}  (T ~ 1 confirmed)")
    print("All synthetic calibration assertions PASSED.\n")

    summary = {
        "synthetic_test": test_results,
    }

    # --- Task 1: Fit T on CNN OOF (xray_oof_kermany.csv) ---
    if args.oof_csv.exists():
        print(f"=== Calibrating CNN OOF ({args.oof_csv}) ===")
        cnn_result = calibrate_cnn_oof(args.oof_csv)
        print(f"  Fitted Temperature T:            {cnn_result['temperature_T']}")
        print(f"  NLL:                             {cnn_result['nll_before']:.4f} -> {cnn_result['nll_after']:.4f}")
        print(f"  ECE:                             {cnn_result['ece_before_pct']:.2f}% -> {cnn_result['ece_after_pct']:.2f}%")
        summary["cnn_oof_kermany"] = cnn_result
    else:
        print(f"  [SKIP] OOF CSV not found: {args.oof_csv}")

    # --- Fit T on feature-vote pLR ---
    if args.plr_csv.exists():
        print(f"\n=== Calibrating Kermany Feature Vote ({args.plr_csv}) ===")
        df = pd.read_csv(args.plr_csv)
        y = df["label"].to_numpy()
        pLR = df["pLR"].to_numpy()

        nll_raw = negative_log_likelihood(pLR, y)
        ece_raw = expected_calibration_error(pLR, y)

        T_kermany = fit_temperature(pLR, y)
        pLR_cal = apply_temperature(pLR, T_kermany)

        nll_cal = negative_log_likelihood(pLR_cal, y)
        ece_cal = expected_calibration_error(pLR_cal, y)

        print(f"  Fitted Temperature T: {T_kermany:.4f}")
        print(f"  Negative Log-Likelihood: {nll_raw:.4f} -> {nll_cal:.4f}")
        print(f"  Expected Calibration Error (ECE): {ece_raw * 100:.2f}% -> {ece_cal * 100:.2f}%")

        summary["kermany_plr"] = {
            "n_samples": len(df),
            "temperature_T": round(T_kermany, 4),
            "nll_before": round(nll_raw, 4),
            "nll_after": round(nll_cal, 4),
            "ece_before_pct": round(ece_raw * 100, 2),
            "ece_after_pct": round(ece_cal * 100, 2),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote calibration summary to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
