"""Re-estimate batch-normalisation statistics on unlabelled recordings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COLLAPSE_SHARE = 0.95


@dataclass(frozen=True)
class RecordingBatch:
    """Normalised windows and the boundaries of their source recordings."""

    windows: np.ndarray
    lengths: np.ndarray

    @classmethod
    def from_recordings(cls, recordings):
        lengths = np.array([len(windows) for windows in recordings], dtype=int)
        if not len(lengths) or np.any(lengths == 0):
            raise ValueError("Every recording must contribute at least one window")
        return cls(np.concatenate(recordings), lengths)

    def predict(self, model):
        """Mean softmax for each recording. Inference batches bound memory use."""
        windows = []
        for start in range(0, len(self.windows), 256):
            chunk = self.windows[start:start + 256]
            windows.append(np.asarray(model(chunk, training=False)).reshape(len(chunk), -1))
        probabilities = np.concatenate(windows)
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1.0)):
            raise ValueError("Expected finite softmax probabilities per window")
        starts = np.r_[0, self.lengths.cumsum()[:-1]]
        totals = np.add.reduceat(probabilities.astype(np.float64), starts, axis=0)
        return totals / self.lengths[:, None]


def reestimate_bn(model, windows):
    """Copy the model and replace its BN moving statistics in one forward pass."""
    from tensorflow import keras

    if len(windows) == 0:
        raise ValueError("Adaptation needs at least one window")
    adapted = keras.models.clone_model(model)
    adapted.set_weights(model.get_weights())
    layers = [layer for layer in adapted.layers if isinstance(layer, keras.layers.BatchNormalization)]
    if not layers:
        raise ValueError("The heart model has no batch-normalisation layers")
    momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.trainable = True
        layer.momentum = 0.0
    adapted(windows, training=True)
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
        layer.trainable = False
    return adapted


def top_class_share(probabilities):
    """Fraction of recordings assigned to the most frequently predicted class."""
    return float(np.bincount(probabilities.argmax(axis=1)).max() / len(probabilities))


def adapt_bn(model, batch):
    """Return a frozen copy and collapse diagnostics, using the original on collapse.

    The guard only rejects a newly collapsed model. Labels are never passed to
    either the statistics update or the guard.
    """
    candidate = reestimate_bn(model, batch.windows)
    before = top_class_share(batch.predict(model))
    after = top_class_share(batch.predict(candidate))
    collapsed = after > COLLAPSE_SHARE >= before
    return (model if collapsed else candidate), {
        "top_class_share_before": before,
        "top_class_share_after": after,
        "collapse_guard_triggered": collapsed,
    }
