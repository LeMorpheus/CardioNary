"""Shared helpers for the X-ray evaluation scripts (RSNA runs, fusion, robustness, tool).

Heavy libraries (tensorflow, cv2, core.*) are imported inside the functions that
need them, so the pure helpers (weights, bootstrap) can be tested in CI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"
CACHE_DIR = REPO_ROOT / "data" / "cache"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

FLOAT_MODEL = MODEL_DIR / "chiron_pulmo_v1_float.keras"
INT8_MODEL = MODEL_DIR / "chiron_pulmo_v1_int8.tflite"
QUANT_REPORT = REPORT_DIR / "quantization_chiron_pulmo_v1.json"
DEFAULT_NORM = (0.57182133, 0.17511542)      # Kermany mean/std, used only if the report is missing


# --------------------------------------------------------------------------- data
def load_norm() -> tuple[float, float]:
    """Kermany training-fold mean and std. Every other dataset is normalised with
    THESE numbers, never with its own statistics."""
    if QUANT_REPORT.exists():
        q = json.loads(QUANT_REPORT.read_text())
        return float(q["norm_mean"]), float(q["norm_std"])
    return DEFAULT_NORM


def normalise(X01: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """(N, 96, 96) in [0, 1] -> (N, 96, 96, 1) float32, as the model expects."""
    return ((np.asarray(X01, dtype=np.float32) - mu) / sd)[..., None].astype(np.float32)


def load_rsna() -> tuple[pd.DataFrame, np.ndarray]:
    """RSNA manifest (one row per patient) and the matching 96x96 images, same order."""
    man = pd.read_csv(MANIFEST_DIR / "rsna_manifest.csv", dtype={"patient_id": str})
    cache = CACHE_DIR / "rsna_96"
    X = np.stack([np.load(cache / f"{p}.npy")[0] for p in man.patient_id])
    return man, X.astype(np.float32)


def load_kermany():
    """X (N,96,96) in [0,1], labels, patient ids, folds."""
    c = np.load(CACHE_DIR / "kermany_logmel.npz", allow_pickle=True)
    return c["X"], c["y"].astype(int), c["patient_id"].astype(str), c["fold"].astype(int)


# --------------------------------------------------------------------------- models
def load_float_model():
    from tensorflow import keras
    return keras.models.load_model(FLOAT_MODEL)


def load_int8():
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(INT8_MODEL))
    it.allocate_tensors()
    return it


def predict_float(model, X01, mu, sd, batch: int = 256) -> np.ndarray:
    """P(pneumonia) per image from a float Keras model."""
    x = normalise(X01, mu, sd)
    return model.predict(x, batch_size=batch, verbose=0).reshape(len(x), -1)[:, 1]


def predict_int8(interp, X01, mu, sd) -> np.ndarray:
    """P(pneumonia) per image from the 8-bit TFLite model."""
    from core.quantize import tflite_predict
    return tflite_predict(interp, normalise(X01, mu, sd))[:, 1]


# --------------------------------------------------------------------------- X-ray characteristics
def _roi(h: int, w: int) -> np.ndarray:
    b = int(0.08 * h)
    roi = np.zeros((h, w), dtype=bool)
    roi[b:h - b, b:w - b] = True
    roi[:, int(0.44 * w):int(0.56 * w)] = False
    return roi


def clahe_view(img01: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Contrast-enhanced view (CLAHE, 8x8 tiles) as uint8, for display."""
    import cv2
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(u8)


def image_features(img01: np.ndarray) -> dict:
    """The 12 numbers lung.model.features writes per image. Must stay identical to it
    (tests/test_lung_app_features.py checks this against the committed CSV)."""
    import cv2
    from core.descriptors import vision_descriptors

    desc = vision_descriptors(img01)
    zones = desc.get("zone_distribution_pct", {})
    u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    roi = _roi(*img01.shape)
    row = {
        "dense_area_pct": desc["dense_area_pct"],
        "asymmetry_index": desc["asymmetry_index"],
        "zone_upper_pct": zones.get("upper", 0.0),
        "zone_middle_pct": zones.get("middle", 0.0),
        "zone_lower_pct": zones.get("lower", 0.0),
        "largest_region_fraction": desc["largest_region_fraction"],
        "regions_detected": desc["regions_detected"],
        "mean_intensity": desc["mean_intensity"],
        "texture_heterogeneity": desc["texture_heterogeneity"],
    }
    for name, clip in (("clahe_std_1", 1.0), ("clahe_std_2", 2.0), ("clahe_std_4", 4.0)):
        f = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        row[name] = round(float(np.std(f.apply(u8)[roi])), 1)
    return row


# --------------------------------------------------------------------------- metadata vote
def meta_matrix(df: pd.DataFrame, age_fill: float) -> np.ndarray:
    """Columns: age/100, (age/100)^2, male. View position is NEVER an input (it describes
    how the film was taken, not the patient). Missing age -> age_fill (training median);
    unknown sex -> 0.5."""
    age = pd.to_numeric(df["age"], errors="coerce").fillna(age_fill).to_numpy(float) / 100.0
    sex = df["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).fillna(0.5).to_numpy(float)
    return np.column_stack([age, age ** 2, sex])


# --------------------------------------------------------------------------- fusion helpers
def _logloss(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fuse(votes: dict, weights: dict) -> np.ndarray:
    """p = sum(w_k * p_k) over the votes present, weights renormalised to sum to 1."""
    names = [k for k in votes if weights.get(k, 0.0) > 0]
    total = sum(weights[k] for k in names)
    return sum(weights[k] * np.asarray(votes[k], dtype=float) for k in names) / total


def choose_weights(votes: dict, y, units: int = 20, metadata_cap: float = 0.3) -> dict:
    """Try every w1 + w2 + w3 = 1 in steps of 1/units (0.05), w3 <= metadata_cap, and keep
    the lowest log-loss. `votes` has key 'cnn' and optionally 'features', 'metadata'."""
    y = np.asarray(y, dtype=float)
    m_max = int(round(metadata_cap * units)) if "metadata" in votes else 0
    best_loss, best = np.inf, None
    for m in range(m_max + 1):
        for f in (range(units - m + 1) if "features" in votes else (0,)):
            c = units - m - f
            w = {"cnn": c / units}
            if "features" in votes:
                w["features"] = f / units
            if "metadata" in votes:
                w["metadata"] = m / units
            loss = _logloss(y, sum(w[k] * np.asarray(votes[k], float) for k in w))
            if loss < best_loss - 1e-12:
                best_loss, best = loss, w
    return best


def paired_auroc(y, p_a, p_b, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC of two score vectors on the same cases, with 95% bootstrap intervals for each
    and for their difference (b minus a), resampling the same cases for both."""
    y, p_a, p_b = np.asarray(y, int), np.asarray(p_a, float), np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    a_s, b_s, d_s = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        ra, rb = roc_auc_score(y[i], p_a[i]), roc_auc_score(y[i], p_b[i])
        a_s.append(ra); b_s.append(rb); d_s.append(rb - ra)
    ci = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]
    ea, eb = float(roc_auc_score(y, p_a)), float(roc_auc_score(y, p_b))
    return {
        "a": {"auroc": ea, "ci95": ci(a_s)},
        "b": {"auroc": eb, "ci95": ci(b_s)},
        "difference_b_minus_a": {"estimate": eb - ea, "ci95": ci(d_s)},
        "resamples": len(d_s),
    }
