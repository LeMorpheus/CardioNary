"""95% bootstrap intervals for the heart branch's results.

Every reported number carries a 95% interval from 1,000 resamples of whole
patients (proposal 4.7). This module builds on cardio.eval.metrics: the metric
definitions and the patient bootstrap come from there, so all heart results use
one implementation. It adds three things:

  * metric_intervals    any of classification_metrics' outputs, with intervals
  * agreement_interval  share of recordings on which two models give the same class
  * statistic_interval  a label-free statistic of probabilities (clinic data)

Yaseen has clip IDs, not patient IDs, so its intervals describe uncertainty over
clips rather than over patients.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from cardio.eval.metrics import BootstrapInterval, classification_metrics, patient_bootstrap


def as_dict(b: BootstrapInterval) -> dict:
    return {"estimate": b.estimate, "lower_95": b.lower, "upper_95": b.upper,
            "valid_resamples": b.valid_resamples, "resamples": b.total_resamples}


def metric_intervals(y, p, groups, names: Iterable[str]) -> dict:
    """Point estimate and interval for each named output of classification_metrics."""
    return {n: as_dict(patient_bootstrap(y, p, groups, lambda yy, pp, n=n: classification_metrics(yy, pp)[n]))
            for n in names}


def agreement_interval(pred_a, prob_b, groups) -> dict:
    """Share of recordings where model A's class equals model B's top class.

    Model A's predictions stand in for labels, so classification accuracy against
    them is exactly the agreement. prob_b must be a proper probability matrix
    (the float model's softmax), since 8-bit outputs do not sum exactly to one.
    """
    y = np.asarray(pred_a, dtype=int)
    return as_dict(patient_bootstrap(y, prob_b, groups,
                                     lambda yy, pp: float(np.mean(pp.argmax(1) == yy))))


def statistic_interval(prob, groups, stat: Callable[[np.ndarray], float]) -> dict:
    """Interval for a statistic of probabilities alone, for data without labels.

    The bootstrap needs a label per row; these statistics never read it, so a
    placeholder of class 0 is passed.
    """
    p = np.asarray(prob, dtype=float)
    return as_dict(patient_bootstrap(np.zeros(len(p), dtype=int), p, groups, lambda yy, pp: float(stat(pp))))


def share_abnormal(p: np.ndarray) -> float:
    """Share of recordings with p(abnormal) = 1 - p(Normal) above 0.5."""
    return float(np.mean((1.0 - p[:, 0]) > 0.5))


def mean_p_abnormal(p: np.ndarray) -> float:
    return float(np.mean(1.0 - p[:, 0]))


def mean_top_probability(p: np.ndarray) -> float:
    return float(np.mean(p.max(1)))
