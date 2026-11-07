"""Batch-normalisation test-time adaptation for chest X-ray models.

Protocol (Roadmap proposal 4.3):
1. Load the float model (the original is never changed).
2. Set every batch-normalisation layer, including those inside the nested
   MobileNetV2, to trainable = True and momentum = 0.0.
3. Run the N clinic images through it once with training=True. Each layer now
   holds the clinic's mean and variance; no gradients are computed.
4. Set the layers back to trainable = False and save the adapted copy.
5. Collapse guard: if the adapted copy labels more than 95% of the N images as
   one class while the original did not, keep the original statistics.

Synthetic shift experiment (Roadmap S1):
Kermany fold 0 made darker (x 0.7) and flatter (contrast halved). Adapt on 32 of
those images and score the rest.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"
CACHE_DIR = REPO_ROOT / "data" / "cache"


def find_bn_layers(model_or_layer) -> list[keras.layers.Layer]:
    """Recursively collect all BatchNormalization layers in a model,
    including those nested inside functional sub-models like MobileNetV2."""
    bns = []
    if hasattr(model_or_layer, "layers"):
        for sub in model_or_layer.layers:
            bns.extend(find_bn_layers(sub))
    elif (
        isinstance(model_or_layer, (keras.layers.BatchNormalization, tf.keras.layers.BatchNormalization))
        or "batch_normalization" in model_or_layer.name.lower()
        or model_or_layer.__class__.__name__ == "BatchNormalization"
    ):
        bns.append(model_or_layer)
    return bns


def copy_model(model: keras.Model) -> keras.Model:
    """Create an independent deep copy of a Keras model with identical weights."""
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    return model_copy


def apply_bn_adaptation(
    model: keras.Model,
    clinic_images: np.ndarray,
    collapse_threshold: float = 0.95,
) -> tuple[keras.Model, bool]:
    """Adapt batch-normalisation statistics using unlabelled clinic images.

    Args:
        model: Trained float Keras model (will not be modified).
        clinic_images: N clinic images, shape (N, H, W, C) or (N, H, W),
            already normalized to the model's expected input scale.
        collapse_threshold: Maximum allowed fraction for any single class
            before collapse guard triggers (default 0.95).

    Returns:
        (adapted_model, collapse_triggered): tuple containing the new adapted
        model and a boolean indicating whether the collapse guard had to
        revert the statistics.
    """
    if clinic_images.ndim == 3:
        clinic_images = clinic_images[..., None]

    n_images = len(clinic_images)
    if n_images == 0:
        raise ValueError("clinic_images must contain at least 1 image.")

    # 1. Independent copy so original is never changed
    adapted = copy_model(model)

    # Collect BN layers and backup original statistics/weights
    bn_layers = find_bn_layers(adapted)
    orig_stats = [layer.get_weights() for layer in bn_layers]

    # Pre-adaptation prediction on clinic_images for collapse check
    orig_probs = model.predict(clinic_images, verbose=0).reshape(n_images, -1)
    orig_preds = orig_probs.argmax(axis=-1)
    orig_n_classes = orig_probs.shape[1]
    orig_counts = np.bincount(orig_preds, minlength=orig_n_classes)
    orig_max_frac = float(orig_counts.max() / n_images)
    orig_was_collapsed = orig_max_frac > collapse_threshold

    # 2. Set every BN layer to trainable = True, momentum = 0.0
    for layer in bn_layers:
        layer.trainable = True
        layer.momentum = 0.0

    # 3. Run the N clinic images through once with training=True
    # No gradients computed, updates moving_mean and moving_variance
    x_tensor = tf.convert_to_tensor(clinic_images.astype(np.float32))
    _ = adapted(x_tensor, training=True)

    # 4. Set layers back to trainable = False
    for layer in bn_layers:
        layer.trainable = False

    # 5. Collapse guard
    adapted_probs = adapted.predict(clinic_images, verbose=0).reshape(n_images, -1)
    adapted_preds = adapted_probs.argmax(axis=-1)
    adapted_counts = np.bincount(adapted_preds, minlength=orig_n_classes)
    adapted_max_frac = float(adapted_counts.max() / n_images)
    adapted_is_collapsed = adapted_max_frac > collapse_threshold

    collapse_triggered = adapted_is_collapsed and not orig_was_collapsed

    if collapse_triggered:
        print(
            f"[Collapse Guard Triggered] Adapted copy labels {adapted_max_frac*100:.1f}% "
            f"as class {int(adapted_counts.argmax())} (original had {orig_max_frac*100:.1f}%). "
            f"Reverting batch-normalisation statistics."
        )
        for layer, weights in zip(bn_layers, orig_stats):
            layer.set_weights(weights)
    else:
        print(
            f"[Adaptation Succeeded] Updated {len(bn_layers)} BN layers on {n_images} images. "
            f"Class balance: {adapted_counts.tolist()} (max {adapted_max_frac*100:.1f}%)."
        )

    return adapted, collapse_triggered


def apply_synthetic_shift(
    images: np.ndarray,
    dark_factor: float = 0.7,
    contrast_factor: float = 0.5,
) -> np.ndarray:
    """Apply synthetic clinic shift: darker (x 0.7) and flatter (contrast halved).

    Operates on images with pixel values in [0, 1].
    Darker: x * 0.7
    Flatter: (x - mean) * 0.5 + mean
    Clipped to [0, 1].
    """
    # 1. Darken
    x = images.astype(np.float32) * dark_factor

    # 2. Flatten (halve contrast around each image's mean intensity)
    if x.ndim == 4:
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
    elif x.ndim == 3:
        mean = x.mean(axis=(1, 2), keepdims=True)
    else:
        mean = x.mean()

    x = (x - mean) * contrast_factor + mean
    return np.clip(x, 0.0, 1.0)


def evaluate_synthetic_shift(
    model_path: Path | str | None = None,
    n_adapt: int = 32,
    seed: int = 0,
) -> dict:
    """Run S1 synthetic shift verification on Kermany fold 0.

    1. Loads Kermany cache and extracts fold 0.
    2. Makes fold 0 darker (x 0.7) and flatter (contrast halved).
    3. Adapts on n_adapt (32) images.
    4. Evaluates clean, shifted plain, and shifted adapted on the remaining images.
    """
    if model_path is None:
        model_path = MODEL_DIR / "chiron_pulmo_v1_float.keras"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load normalisation parameters from quantization report if present, else defaults
    quant_rep = REPORT_DIR / "quantization_chiron_pulmo_v1.json"
    if quant_rep.exists():
        with open(quant_rep) as f:
            qr = json.load(f)
        mu, sd = float(qr["norm_mean"]), float(qr["norm_std"])
    else:
        mu, sd = 0.57182133, 0.17511542

    cache_file = CACHE_DIR / "kermany_logmel.npz"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}. Build Kermany cache first.")

    cache = np.load(cache_file, allow_pickle=True)
    X, y, pid, fold = cache["X"], cache["y"], cache["patient_id"], cache["fold"]

    # Select fold 0
    f0_mask = fold == 0
    X_f0 = X[f0_mask]  # values in [0, 1]
    y_f0 = y[f0_mask]
    pid_f0 = pid[f0_mask]

    n_total = len(X_f0)
    print(f"Loaded fold 0: {n_total} images across {len(set(pid_f0))} patients.")

    # Fixed seed for reproducibility (Rule: Seeds are fixed at 0)
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    adapt_idx = indices[:n_adapt]
    eval_idx = indices[n_adapt:]

    # The N images used to adapt are never scored (Rule: honesty)
    print(f"Adapting on {len(adapt_idx)} images, evaluating on {len(eval_idx)} images.")

    # 1. Clean data (normalised)
    X_clean_eval_norm = (((X_f0[eval_idx] - mu) / sd)[..., None]).astype(np.float32)

    # 2. Shifted data
    X_f0_shifted = apply_synthetic_shift(X_f0)
    X_shift_adapt_norm = (((X_f0_shifted[adapt_idx] - mu) / sd)[..., None]).astype(np.float32)
    X_shift_eval_norm = (((X_f0_shifted[eval_idx] - mu) / sd)[..., None]).astype(np.float32)
    y_eval = y_f0[eval_idx]
    pid_eval = pid_f0[eval_idx]

    # Load float model
    print(f"Loading float model: {model_path} ...")
    orig_model = keras.models.load_model(model_path)

    # Adapt
    adapted_model, collapse = apply_bn_adaptation(
        orig_model, X_shift_adapt_norm, collapse_threshold=0.95
    )

    # Prediction helper (item + per-patient)
    def score_model(m: keras.Model, x_norm: np.ndarray) -> tuple[float, float, float]:
        probs = m.predict(x_norm, verbose=0).reshape(len(x_norm), -1)
        preds = probs.argmax(axis=-1)
        item_acc = float(accuracy_score(y_eval, preds))

        # Per-patient accuracy (mean probability per patient)
        df = pd.DataFrame({"pid": pid_eval, "y": y_eval, "p0": probs[:, 0], "p1": probs[:, 1]})
        grp = df.groupby("pid").agg({"y": "first", "p0": "mean", "p1": "mean"})
        pat_preds = (grp["p1"] > grp["p0"]).astype(int)
        pat_acc = float((pat_preds == grp["y"]).mean())

        # AUROC
        try:
            auc = float(roc_auc_score(y_eval, probs[:, 1]))
        except Exception:
            auc = 0.0

        return item_acc, pat_acc, auc

    clean_acc, clean_pat, clean_auc = score_model(orig_model, X_clean_eval_norm)
    plain_acc, plain_pat, plain_auc = score_model(orig_model, X_shift_eval_norm)
    adapt_acc, adapt_pat, adapt_auc = score_model(adapted_model, X_shift_eval_norm)

    # Recovery formula: (adapted - plain) / (clean - plain)
    recovery_item = (
        float((adapt_acc - plain_acc) / (clean_acc - plain_acc))
        if abs(clean_acc - plain_acc) > 1e-6
        else 0.0
    )
    recovery_pat = (
        float((adapt_pat - plain_pat) / (clean_pat - plain_pat))
        if abs(clean_pat - plain_pat) > 1e-6
        else 0.0
    )

    results = {
        "n_adapt": int(n_adapt),
        "n_eval": int(len(eval_idx)),
        "seed": int(seed),
        "collapse_guard_triggered": bool(collapse),
        "clean": {
            "item_acc": clean_acc,
            "patient_acc": clean_pat,
            "auroc": clean_auc,
        },
        "shifted_plain": {
            "item_acc": plain_acc,
            "patient_acc": plain_pat,
            "auroc": plain_auc,
        },
        "shifted_adapted": {
            "item_acc": adapt_acc,
            "patient_acc": adapt_pat,
            "auroc": adapt_auc,
        },
        "recovery": {
            "item": recovery_item,
            "patient": recovery_pat,
        },
    }

    print("\n=== Synthetic Shift Results (Kermany Fold 0, Darker x0.7 & Flatter x0.5) ===")
    print(f"Clean:           Item Acc = {clean_acc*100:.2f}%, Patient Acc = {clean_pat*100:.2f}%, AUROC = {clean_auc:.4f}")
    print(f"Shifted Plain:   Item Acc = {plain_acc*100:.2f}%, Patient Acc = {plain_pat*100:.2f}%, AUROC = {plain_auc:.4f}")
    print(f"Shifted Adapted: Item Acc = {adapt_acc*100:.2f}%, Patient Acc = {adapt_pat*100:.2f}%, AUROC = {adapt_auc:.4f}")
    print(f"Recovery:        Item = {recovery_item*100:.1f}%, Patient = {recovery_pat*100:.1f}%")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Test-time batch norm adaptation.")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "chiron_pulmo_v1_float.keras"))
    parser.add_argument("--n-adapt", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    res = evaluate_synthetic_shift(args.model, n_adapt=args.n_adapt, seed=args.seed)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2))
        print(f"Saved results to {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

