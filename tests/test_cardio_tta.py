from types import SimpleNamespace

import numpy as np
import pytest

from cardio.eval.metrics import classification_metrics
from cardio.eval.run_tta import binary_metrics, compare, draw_indices, split_patients
from cardio.model.final_model import FinalModel, STEM
from cardio.tta import bn_adapt


def test_bn_copy_updates_statistics_and_preserves_source_weights(tmp_path):
    keras = pytest.importorskip("tensorflow").keras
    inputs = keras.Input(shape=(2,))
    layer = keras.layers.BatchNormalization()
    model = keras.Model(inputs, layer(inputs))
    before = model.get_weights()
    windows = np.array([[1, 2], [3, 6], [5, 10]], dtype=np.float32)
    adapted = bn_adapt.reestimate_bn(model, windows)
    copy_bn = adapted.layers[-1]
    assert not copy_bn.trainable
    assert layer.trainable
    np.testing.assert_allclose(copy_bn.moving_mean, windows.mean(axis=0))
    np.testing.assert_allclose(copy_bn.moving_variance, windows.var(axis=0))
    for actual, original in zip(model.get_weights(), before):
        np.testing.assert_array_equal(actual, original)
    for updated, original in zip(adapted.get_weights()[:2], before[:2]):
        np.testing.assert_array_equal(updated, original)
    checkpoint = tmp_path / f"{STEM}_float.keras"
    adapted.save(checkpoint)
    (tmp_path / f"quantization_{STEM}.json").write_text('{"norm_mean": 0, "norm_std": 1}')
    loaded = FinalModel(tmp_path, tmp_path, load_int8=False).float_model
    assert not loaded.layers[-1].trainable
    np.testing.assert_allclose(loaded(windows, training=False), adapted(windows, training=False))


@pytest.mark.parametrize("already_collapsed", [False, True])
def test_guard_rejects_only_new_collapse(monkeypatch, already_collapsed):
    original, candidate = object(), object()
    monkeypatch.setattr(bn_adapt, "reestimate_bn", lambda model, windows: candidate)
    collapsed = np.tile([0.1, 0.9], (20, 1))
    mixed = np.vstack((collapsed[:10], collapsed[:10, ::-1]))
    batch = SimpleNamespace(windows=None, predict=lambda net: collapsed
                            if net is candidate or already_collapsed else mixed)
    selected, report = bn_adapt.adapt_bn(original, batch)
    assert selected is (candidate if already_collapsed else original)
    assert report["collapse_guard_triggered"] is (not already_collapsed)


def test_draws_exclude_all_recordings_of_scoring_patients():
    records = [SimpleNamespace(patient=str(i), label=i % 2) for i in range(48) for _ in range(3)]
    pool, test = split_patients(records)
    indices = draw_indices(pool, 32, 0.9, np.random.default_rng(0))
    assert len(set(indices)) == 32
    assert sum(pool[index].label for index in indices) == 29
    assert not ({pool[index].patient for index in indices} & {row.patient for row in test})


def test_weighted_bootstrap_scores_match_existing_metrics_with_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.8, 0.2, 0.5])
    weights = np.array([[1., 1., 1., 1.], [2., 1., 0., 1.], [2., 0., 2., 0.]])
    scores = binary_metrics(y, p, weights)
    for i, row in enumerate(weights.astype(int)):
        expected = classification_metrics(np.repeat(y, row), np.repeat(p, row))
        for name in scores:
            np.testing.assert_allclose(scores[name][i], expected[name])


def test_recovery_is_undefined_without_a_clean_to_shift_drop():
    clean = {"macro_f1": np.array([0.9, 0.9]), "auroc": np.array([0.5, 0.5])}
    plain = {"macro_f1": np.array([0.5, 0.5]), "auroc": np.array([0.6, 0.6])}
    after = {"macro_f1": np.array([0.7, 0.7]), "auroc": np.array([0.7, 0.7])}
    result = compare(clean, plain, [after])
    assert result["macro_f1"]["recovery"]["estimate"] == pytest.approx(0.5)
    assert result["auroc"]["recovery"]["estimate"] is None
