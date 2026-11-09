"""Pure helpers behind the fusion comparisons: weight search, fusion, paired bootstrap."""
import numpy as np
import pytest

from lung.eval.common import choose_weights, fuse, meta_matrix, paired_auroc


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    cnn = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    return rng, y, cnn


def test_weights_sum_to_one_and_metadata_is_capped():
    rng, y, cnn = _data()
    w = choose_weights({"cnn": cnn, "features": rng.uniform(0.01, 0.99, len(y)),
                        "metadata": np.clip(0.3 + 0.4 * y + rng.normal(0, .2, len(y)), .01, .99)}, y)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["metadata"] <= 0.3 + 1e-9


def test_a_noise_vote_does_not_beat_the_informative_vote():
    rng, y, cnn = _data()
    w = choose_weights({"cnn": cnn, "features": rng.uniform(0.01, 0.99, len(y))}, y)
    assert w["cnn"] >= 0.6


def test_without_metadata_no_metadata_weight():
    _, y, cnn = _data()
    assert "metadata" not in choose_weights({"cnn": cnn}, y)


def test_fuse_renormalises_when_a_vote_is_missing():
    out = fuse({"cnn": np.array([0.8]), "features": np.array([0.4])},
               {"cnn": 0.5, "features": 0.3, "metadata": 0.2})
    assert out[0] == pytest.approx((0.5 * 0.8 + 0.3 * 0.4) / 0.8)


def test_paired_auroc_detects_a_better_score_and_is_reproducible():
    rng, y, cnn = _data()
    better = np.clip(0.1 + 0.8 * y + rng.normal(0, 0.1, len(y)), 0, 1)
    a = paired_auroc(y, cnn, better, n_boot=200)
    assert a["difference_b_minus_a"]["estimate"] > 0
    assert a["difference_b_minus_a"]["ci95"][0] > 0
    assert a == paired_auroc(y, cnn, better, n_boot=200)


def test_meta_matrix_handles_missing_age_and_sex():
    import pandas as pd
    m = meta_matrix(pd.DataFrame({"age": [50.0, np.nan], "sex": ["M", "UNKNOWN"]}), age_fill=40.0)
    assert m.shape == (2, 3)
    assert m[0].tolist() == pytest.approx([0.5, 0.25, 1.0])
    assert m[1].tolist() == pytest.approx([0.4, 0.16, 0.5])
