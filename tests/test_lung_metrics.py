import numpy as np
import pytest

from lung.eval.metrics import (
    aggregate_patients,
    bootstrap_auroc_interval,
    evaluate_predictions,
    expected_calibration_error,
    sensitivity_at_specificity,
)


def test_aggregate_patients_averages_probabilities():
    ids = np.array(["A", "A", "B", "C"])
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.2, 0.4, 0.8, 0.9])

    patient_ids, y, p = aggregate_patients(ids, labels, probs)

    assert patient_ids.tolist() == ["A", "B", "C"]
    assert y.tolist() == [0, 1, 1]
    assert np.allclose(p, [0.3, 0.8, 0.9])


def test_aggregate_patients_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_patients(
            np.array(["A", "A"]),
            np.array([0, 1]),
            np.array([0.2, 0.8]),
        )


def test_ece_perfect_calibration():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(labels, probs, n_bins=15) == 0.0


def test_sensitivity_at_90_percent_specificity():
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.9, 0.8])

    result = sensitivity_at_specificity(labels, probs, 0.90)

    assert result == 1.0


def test_bootstrap_interval_is_reproducible():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    first = bootstrap_auroc_interval(labels, probs, n_bootstrap=100, seed=0)
    second = bootstrap_auroc_interval(labels, probs, n_bootstrap=100, seed=0)

    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_evaluate_predictions_returns_required_metrics():
    ids = np.array(["A", "B", "C", "D", "E", "F"])
    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    result = evaluate_predictions(
        ids, labels, probs, n_bootstrap=100, seed=0
    )

    required = {
        "n_patients",
        "accuracy",
        "auroc",
        "sensitivity_at_90_specificity",
        "ece",
        "auroc_ci_95",
        "ece_bins",
        "bootstrap_resamples",
        "bootstrap_seed",
    }

    assert required.issubset(result.keys())
    assert result["n_patients"] == 6
    assert result["accuracy"] == 1.0
    assert result["auroc"] == 1.0
    assert result["auroc_ci_95"] == [1.0, 1.0]
