"""Load the final heart-sound model and run it, in float and in 8-bit.

The final model is the one `python -m core.quantize --dataset yaseen
--holdout-fold 0` builds: trained on Yaseen folds 1 to 4, with fold 0 held out.
That command writes the float model, its 8-bit copy and a report holding the
normalisation mean and standard deviation of the training folds. Every
recording from any other source is normalised with those Yaseen statistics,
never its own.

Usage:
    from cardio.model.final_model import FinalModel
    m = FinalModel()
    p_float, p_int8 = m.predict_file("clip.wav")     # mean softmax over windows
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from core.features import CARDIO_CONFIG, features_for_file

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"
STEM = "chiron_cardio_v1"                     # file name used by core/quantize.py
CLASSES = ["Normal", "AS", "MS", "MR", "MVP"]


class FinalModel:
    def __init__(self, model_dir: Path = MODEL_DIR, report_dir: Path = REPO_ROOT / "reports", *, load_int8=True):
        import tensorflow as tf
        from tensorflow import keras

        float_path = model_dir / f"{STEM}_float.keras"
        int8_path = model_dir / f"{STEM}_int8.tflite"
        report = report_dir / f"quantization_{STEM}.json"
        required = (float_path, report, int8_path) if load_int8 else (float_path, report)
        for p in required:
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} is missing; build it with: python -m core.quantize --dataset yaseen --holdout-fold 0")
        rep = json.loads(report.read_text())
        self.mu, self.sd = float(rep["norm_mean"]), float(rep["norm_std"])
        self.float_model = keras.models.load_model(float_path, compile=load_int8)
        self.interp = None
        self.files = {"float": float_path.name}
        if load_int8:
            self.interp = tf.lite.Interpreter(model_path=str(int8_path))
            self.interp.allocate_tensors()
            self.files.update(int8=int8_path.name, int8_bytes=int8_path.stat().st_size)

    def normalise(self, feats: np.ndarray) -> np.ndarray:
        """(n, 40, 192) log-mel windows -> normalised (n, 40, 192, 1) float32."""
        return ((feats - self.mu) / self.sd)[..., None].astype(np.float32)

    def predict_windows(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Softmax per window from the float model and from the 8-bit model."""
        if self.interp is None:
            raise ValueError("Load the INT8 model to compare float and INT8 predictions")
        from core.quantize import tflite_predict

        x = self.normalise(feats)
        p_float = self.float_model.predict(x, verbose=0).reshape(len(x), -1)
        p_int8 = tflite_predict(self.interp, x)
        return p_float, p_int8

    def predict_file(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Mean softmax over a recording's 3-second windows, float and 8-bit."""
        feats = features_for_file(str(path), CARDIO_CONFIG)
        p_float, p_int8 = self.predict_windows(feats)
        return p_float.mean(0), p_int8.mean(0)
