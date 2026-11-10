import numpy as np
import pytest

from cardio.eval.run_robustness import accuracy_comparison


def test_adapted_accuracy_averages_draws_and_pairs_clip_resamples():
    plain = np.array([True, True, False, False])
    adapted = np.array([[True, False, True, False], [False, True, True, True]])
    weights = np.array([[1., 1., 1., 1.], [2., 0., 2., 0.], [0., 2., 0., 2.]])

    result = accuracy_comparison(plain, adapted, weights)

    assert result["plain"]["estimate"] == 0.5
    assert result["adapted"]["estimate"] == 0.625
    assert result["gain"]["estimate"] == 0.125
    assert result["gain"]["lower_95"] == pytest.approx(0.00625)
    assert result["gain"]["upper_95"] == pytest.approx(0.24375)
