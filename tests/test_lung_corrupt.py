import numpy as np
import pytest

from lung.eval.corrupt import (
    SEVERITIES,
    apply_corruption,
    generate_corruptions,
)


@pytest.fixture
def sample_image():
    return np.tile(
        np.linspace(0, 1, 96, dtype=np.float32),
        (96, 1),
    )


@pytest.mark.parametrize("corruption", ["blur", "noise", "contrast", "jpeg"])
@pytest.mark.parametrize("severity", [1, 2, 3, 4, 5])
def test_corruption_output_is_valid(sample_image, corruption, severity):
    result = apply_corruption(sample_image, corruption, severity, seed=0)

    assert result.shape == sample_image.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1


def test_noise_is_reproducible(sample_image):
    first = apply_corruption(sample_image, "noise", 3, seed=0)
    second = apply_corruption(sample_image, "noise", 3, seed=0)

    assert np.array_equal(first, second)


def test_generate_all_20_corruptions(sample_image):
    results = generate_corruptions(sample_image, seed=0)

    assert set(results) == set(SEVERITIES)
    assert sum(len(levels) for levels in results.values()) == 20


def test_invalid_corruption_raises(sample_image):
    with pytest.raises(ValueError):
        apply_corruption(sample_image, "rotation", 1)


def test_invalid_severity_raises(sample_image):
    with pytest.raises(ValueError):
        apply_corruption(sample_image, "blur", 6)


def test_invalid_image_range_raises():
    image = np.full((96, 96), 2.0, dtype=np.float32)

    with pytest.raises(ValueError):
        apply_corruption(image, "blur", 1)
