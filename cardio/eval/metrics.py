"""Binary and single-label multiclass metrics, plus patient-cluster bootstrap.

Labels are integer column indices. A 1-D score is P(class=1); an N x C
matrix contains class probabilities. For multilabel data, evaluate each label
as a separate binary task. Undefined ROC metrics return NaN.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve


def _inputs(y_true, y_score):
    y = np.asarray(y_true)
    p = np.asarray(y_score, dtype=float)
    if p.ndim == 1:
        p = np.column_stack((1 - p, p))
    if y.ndim != 1 or not y.size or p.ndim != 2 or p.shape[0] != y.size or p.shape[1] < 2:
        raise ValueError("Expected nonempty labels (N,) and probabilities (N,) or (N,C).")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Probabilities must be finite, in [0,1], and sum to one.")
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError("Labels must be integer probability-column indices.")
    return y, p


def expected_calibration_error(y_true, y_score, n_bins=15):
    """Top-label ECE with equal-width bins, including confidence zero and one."""
    y, p = _inputs(y_true, y_score)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    confidence = p.max(axis=1)
    predicted = (p[:, 1] >= 0.5).astype(int) if p.shape[1] == 2 else p.argmax(axis=1)
    correct = predicted == y
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    return float(sum(
        np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
        for b in range(n_bins) if np.any(bins == b)
    ))


def sensitivity_at_specificity(y_true, y_score, specificity=0.9):
    """Best empirical sensitivity with specificity >= target; no interpolation.

    Binary labels only. Tied scores are handled as a group by roc_curve.
    Returns NaN if positives or negatives are absent.
    """
    y, p = _inputs(y_true, y_score)
    if p.shape[1] != 2 or not 0 <= specificity <= 1:
        raise ValueError("Requires a binary task and specificity in [0,1].")
    if np.unique(y).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p[:, 1], drop_intermediate=False)
    return float(tpr[fpr <= 1 - specificity + 1e-12].max())


def classification_metrics(y_true, y_score):
    """Return flat metric names; multiclass ROC uses macro one-vs-rest.

    Binary ROC measures class 1. Multiclass ROC metrics are NaN if any
    class is absent. F1/recall use zero for an absent class, including it
    in the macro average. Binary decisions use p(class 1) >= 0.5.
    """
    y, p = _inputs(y_true, y_score)
    classes = np.arange(p.shape[1])
    pred = (p[:, 1] >= 0.5).astype(int) if len(classes) == 2 else p.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "ece": expected_calibration_error(y, p),
    }
    result.update({f"recall_{c}": float(v) for c, v in zip(
        classes, recall_score(y, pred, labels=classes, average=None, zero_division=0)
    )})
    targets = [1] if len(classes) == 2 else classes
    aucs, sensitivities = [], []
    for c in targets:
        binary = (y == c).astype(int)
        aucs.append(float(roc_auc_score(binary, p[:, c])) if np.unique(binary).size == 2 else float("nan"))
        sensitivities.append(sensitivity_at_specificity(binary, p[:, c]))
    result["auroc"] = float(np.mean(aucs))
    result["sensitivity_at_90_specificity"] = float(np.mean(sensitivities))
    return result


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    valid_resamples: int
    total_resamples: int


def patient_bootstrap(y_true, y_score, patient_ids, metric: Callable,
                      n_resamples=1000, seed=0, confidence=0.95):
    """Percentile interval from sampling whole patients with replacement.

    All recordings of each sampled patient are retained, with multiplicity.
    The point metric remains recording-weighted. Nonfinite resample metrics
    are excluded and counted; if all are undefined the interval is NaN.
    Other metric exceptions propagate. Use dataset-prefixed patient IDs
    when pooling datasets to avoid merging unrelated patients.
    """
    y, p = _inputs(y_true, y_score)
    ids = np.asarray(patient_ids)
    if ids.shape != y.shape or any(v is None or str(v).strip().lower() in ("", "nan") for v in ids):
        raise ValueError("Provide one nonmissing patient ID per recording.")
    if not isinstance(n_resamples, int) or n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Require positive integer resamples and confidence in (0,1).")
    # Dict preserves first occurrence and accepts string or integer patient IDs.
    groups = {}
    for i, patient in enumerate(ids.tolist()):
        groups.setdefault(patient, []).append(i)
    clusters = [np.asarray(indices) for indices in groups.values()]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        indices = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
        value = float(metric(y[indices], p[indices]))
        if np.isfinite(value):
            values.append(value)
    alpha = (1 - confidence) / 2
    bounds = np.quantile(values, [alpha, 1-alpha]) if values else [np.nan, np.nan]
    return BootstrapInterval(float(metric(y, p)), float(bounds[0]), float(bounds[1]), len(values), n_resamples)
