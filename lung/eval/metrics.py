"""Patient-level evaluation metrics for the RSNA chest X-ray model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def aggregate_patients(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average image probabilities per patient and ensure one label per patient."""
    patient_ids = np.asarray(patient_ids)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if not (len(patient_ids) == len(y_true) == len(y_prob)):
        raise ValueError("patient_ids, y_true and y_prob must have equal lengths")
    if len(patient_ids) == 0:
        raise ValueError("At least one prediction is required")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")

    unique_ids = np.unique(patient_ids)
    patient_labels = []
    patient_probs = []

    for patient_id in unique_ids:
        mask = patient_ids == patient_id
        labels = np.unique(y_true[mask])

        if len(labels) != 1:
            raise ValueError(f"Patient {patient_id} has conflicting labels")

        patient_labels.append(labels[0])
        patient_probs.append(y_prob[mask].mean())

    return (
        unique_ids,
        np.asarray(patient_labels, dtype=int),
        np.asarray(patient_probs, dtype=float),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Calculate binary ECE using equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("y_true and y_prob must have equal non-zero lengths")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (y_prob >= bin_edges[index]) & (y_prob <= bin_edges[index + 1])
        else:
            mask = (y_prob >= bin_edges[index]) & (y_prob < bin_edges[index + 1])

        if mask.any():
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            ece += mask.mean() * abs(bin_accuracy - bin_confidence)

    return float(ece)


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.90,
) -> float:
    """Return maximum sensitivity at or above the requested specificity."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if not 0 < target_specificity <= 1:
        raise ValueError("target_specificity must be in (0, 1]")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Sensitivity at fixed specificity requires both classes")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    eligible = np.flatnonzero(specificity >= target_specificity)

    if len(eligible) == 0:
        return 0.0

    return float(np.max(tpr[eligible]))


def bootstrap_auroc_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for patient-level AUROC."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("y_true and y_prob must have equal non-zero lengths")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUROC interval requires both classes")

    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sampled_labels = y_true[indices]

        # AUROC is undefined if a bootstrap sample contains only one class.
        if len(np.unique(sampled_labels)) < 2:
            continue

        scores.append(roc_auc_score(sampled_labels, y_prob[indices]))

    if not scores:
        raise RuntimeError("No valid bootstrap samples contained both classes")

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(scores, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def evaluate_predictions(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    """Compute all required metrics after aggregating predictions by patient."""
    patient_ids, labels, probabilities = aggregate_patients(
        patient_ids, y_true, y_prob
    )

    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation requires both classes")

    auroc = float(roc_auc_score(labels, probabilities))
    ci_lower, ci_upper = bootstrap_auroc_interval(
        labels, probabilities,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return {
        "n_patients": int(len(patient_ids)),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "auroc": auroc,
        "sensitivity_at_90_specificity": sensitivity_at_specificity(
            labels, probabilities, target_specificity=0.90
        ),
        "ece": expected_calibration_error(labels, probabilities, n_bins=n_bins),
        "auroc_ci_95": [ci_lower, ci_upper],
        "ece_bins": int(n_bins),
        "bootstrap_resamples": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }
