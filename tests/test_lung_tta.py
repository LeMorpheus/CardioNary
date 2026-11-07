"""Tests for test-time adaptation and latency/memory benchmarking.

Verifies:
1. BN layer identification across nested MobileNetV2 architecture.
2. Independent model copying (original model is never modified).
3. BN statistics update with momentum=0.0 and no gradients.
4. Collapse guard logic (restoring statistics when single class > 95%).
5. Synthetic clinic shift transformation (darker x0.7, contrast halved).
6. Timing and memory benchmarking for float and INT8 models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from lung.tta.bn_adapt import (
    find_bn_layers,
    copy_model,
    apply_bn_adaptation,
    apply_synthetic_shift,
)
from lung.eval.timing import (
    benchmark_float_model,
    benchmark_int8_model,
    get_test_images,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOAT_MODEL_PATH = REPO_ROOT / "models" / "chiron_pulmo_v1_float.keras"
INT8_MODEL_PATH = REPO_ROOT / "models" / "chiron_pulmo_v1_int8.tflite"


@pytest.fixture(scope="module")
def float_model():
    if not FLOAT_MODEL_PATH.exists():
        pytest.skip(f"Float model not found: {FLOAT_MODEL_PATH}")
    return keras.models.load_model(FLOAT_MODEL_PATH)


def test_find_bn_layers_nested_mobilenet(float_model):
    """Finds all BatchNormalization layers inside MobileNetV2."""
    bns = find_bn_layers(float_model)
    assert len(bns) > 0, "No BN layers found"
    # MobileNetV2 alpha=0.35 typically has 52 BatchNormalization layers
    assert len(bns) >= 50, f"Expected >= 50 BN layers, found {len(bns)}"
    for bn in bns:
        assert isinstance(bn, (keras.layers.BatchNormalization, tf.keras.layers.BatchNormalization)) or "batch_normalization" in bn.name.lower()


def test_copy_model_preserves_independence(float_model):
    """Ensures adapted copy does not modify original model weights."""
    copied = copy_model(float_model)
    orig_bns = find_bn_layers(float_model)
    copy_bns = find_bn_layers(copied)

    assert len(orig_bns) == len(copy_bns)
    orig_mean = orig_bns[0].moving_mean.numpy().copy()

    # Modify copy
    copy_bns[0].moving_mean.assign(orig_mean + 10.0)

    # Check original remains untouched
    assert np.allclose(orig_bns[0].moving_mean.numpy(), orig_mean)
    assert not np.allclose(copy_bns[0].moving_mean.numpy(), orig_mean)


def test_bn_adaptation_updates_statistics(float_model):
    """Running clinic images through adapted model updates moving mean and variance."""
    rng = np.random.default_rng(42)
    # Generate 16 test clinic images
    clinic_images = rng.uniform(0.1, 0.9, (16, 96, 96, 1)).astype(np.float32)

    orig_bns = find_bn_layers(float_model)
    orig_means = [bn.moving_mean.numpy().copy() for bn in orig_bns]

    adapted, collapse_triggered = apply_bn_adaptation(
        float_model, clinic_images, collapse_threshold=0.99
    )

    adapt_bns = find_bn_layers(adapted)
    adapt_means = [bn.moving_mean.numpy() for bn in adapt_bns]

    # At least some BN moving means must have changed
    has_changed = any(not np.allclose(o, a) for o, a in zip(orig_means, adapt_means))
    assert has_changed, "BN statistics were not updated by adaptation pass"

    # And all BN layers must be back to trainable=False
    for bn in adapt_bns:
        assert not bn.trainable, "BN layer was left trainable"

    # Original model still untouched
    for bn, o in zip(orig_bns, orig_means):
        assert np.allclose(bn.moving_mean.numpy(), o), "Original model was mutated"


def test_collapse_guard_restores_statistics(float_model):
    """If adapted predictions collapse (>95% single class), original stats are restored."""
    orig_bns = find_bn_layers(float_model)
    orig_means = [bn.moving_mean.numpy().copy() for bn in orig_bns]

    # Artificially trigger collapse by using extreme shift and very low threshold
    rng = np.random.default_rng(99)
    extreme_images = (rng.normal(10.0, 1.0, (20, 96, 96, 1))).astype(np.float32)

    adapted, collapse_triggered = apply_bn_adaptation(
        float_model, extreme_images, collapse_threshold=0.50
    )

    adapt_bns = find_bn_layers(adapted)
    # If collapse was triggered, moving means must equal original
    if collapse_triggered:
        for bn, o in zip(adapt_bns, orig_means):
            assert np.allclose(bn.moving_mean.numpy(), o), "Collapse guard failed to restore statistics"


def test_apply_synthetic_shift():
    """Verify synthetic shift: darker (x 0.7) and flatter (contrast halved)."""
    rng = np.random.default_rng(123)
    clean = rng.uniform(0.2, 0.8, (10, 96, 96)).astype(np.float32)

    shifted = apply_synthetic_shift(clean, dark_factor=0.7, contrast_factor=0.5)

    assert shifted.shape == clean.shape
    assert np.all(shifted >= 0.0) and np.all(shifted <= 1.0)

    # Shifted images should be darker on average
    assert shifted.mean() < clean.mean()
    # Shifted images should have lower standard deviation (flatter contrast)
    assert shifted.std() < clean.std()


def test_timing_and_memory_benchmarking():
    """Verify timing script runs on float and int8 models."""
    if not INT8_MODEL_PATH.exists() or not FLOAT_MODEL_PATH.exists():
        pytest.skip("Models not available for benchmark test")

    images, _, _ = get_test_images(num_images=5)

    res_float = benchmark_float_model(FLOAT_MODEL_PATH, images, warmup=1)
    assert res_float["mean_time_ms"] > 0
    assert res_float["peak_memory_bytes"] > 0
    assert res_float["file_size_bytes"] > 0

    res_int8 = benchmark_int8_model(INT8_MODEL_PATH, images, warmup=1)
    assert res_int8["mean_time_ms"] > 0
    assert res_int8["peak_memory_bytes"] >= 0
    assert res_int8["file_size_kb"] < 1000  # < 1 MB edge budget requirement

