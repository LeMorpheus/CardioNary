"""Tests for temperature scaling calibration (lung/fusion/calibrate.py).

Validates that:
  - Over-confident probabilities produce fitted T > 1.0.
  - Under-confident probabilities produce fitted T < 1.0.
  - Temperature scaling preserves probability invariants (in [0, 1] and sum to 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from lung.fusion.calibrate import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    test_overconfident_probabilities as run_synthetic_test,
)


def test_overconfident_probabilities_produce_T_greater_than_one():
    # 75% empirical accuracy, but outputs extreme confidence (0.99 / 0.01)
    y = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    p = np.array([0.99, 0.99, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.99, 0.01, 0.01])

    T = fit_temperature(p, y)
    assert T > 1.0, f"Expected T > 1 for over-confident model, got {T}"


def test_synthetic_calibration_suite():
    results = run_synthetic_test()
    assert results["overconfident_T"] > 1.0
    assert results["underconfident_T"] < 1.0
    assert 0.85 <= results["well_calibrated_T"] <= 1.15


def test_temperature_scaling_preserves_valid_probabilities():
    probs = np.array([0.1, 0.5, 0.9])
    cal_over = apply_temperature(probs, temperature=2.5)
    cal_under = apply_temperature(probs, temperature=0.5)

    assert np.all((cal_over >= 0.0) & (cal_over <= 1.0))
    assert np.all((cal_under >= 0.0) & (cal_under <= 1.0))

    # T > 1 pulls towards 0.5 (so 0.9 decreases, 0.1 increases)
    assert cal_over[0] > probs[0]
    assert cal_over[2] < probs[2]

    # T < 1 pushes away from 0.5 (so 0.9 increases, 0.1 decreases)
    assert cal_under[0] < probs[0]
    assert cal_under[2] > probs[2]


def test_multiclass_temperature_scaling():
    # Multi-class 2D probabilities (N, 3)
    p = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
    ])
    y = np.array([0, 1, 0, 2])  # 50% accuracy with 80% confidence -> overconfident

    T = fit_temperature(p, y)
    assert T > 1.0

    p_cal = apply_temperature(p, T)
    assert p_cal.shape == p.shape
    np.testing.assert_allclose(p_cal.sum(axis=1), 1.0, atol=1e-6)
