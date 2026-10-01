"""Train and cross-validate the Chiron lung-sound classifier.

Protocol (docs/05_DATASET_SELECTION.md):
  * patient-disjoint stratified K-fold cross-validation
  * normalisation statistics fitted on the TRAINING fold only (rule 5.5)
  * augmentation applied inside the training generator only (rule 5.3)
  * metrics reported at BOTH window level and patient level

Patient-level is the number that matters: a clinician gets one answer per
patient, not one per 5-second window. Published lung-sound papers report
patient/subject-level figures, so that is what we compare against.

Usage:
    python -m core.train --feature logmel
    python -m core.train --feature mfcc --seeds 3
    python -m core.train --feature logmel --no-augment
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
REPORT_DIR = REPO_ROOT / "reports"
MODEL_DIR = REPO_ROOT / "models"
DATASET_CLASSES = {
    "kauh": ["Normal", "Asthma", "Heart Failure", "COPD"],
    "yaseen": ["Normal", "AS", "MS", "MR", "MVP"],
    "icbhi": ["Bronchiectasis", "Bronchiolitis", "COPD", "Healthy", "Pneumonia", "URTI"],
    "icbhi_meditron": ["Bronchiectasis", "Bronchiolitis", "COPD", "Healthy", "URTI"],
    "merged": ["Normal", "Asthma", "COPD", "Pneumonia", "BRON", "Heart Failure"],
    "kermany": ["Normal", "Pneumonia"],
}
CLASSES = DATASET_CLASSES["kauh"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_mobilenet(input_shape, n_classes: int, alpha: float = 0.35,
                    dropout: float = 0.3, trainable_from: int = 40) -> keras.Model:
    """ImageNet-pretrained MobileNetV2 backbone, fine-tuned.

    This is what the published chest X-ray work does, and it is the single
    biggest legitimate accuracy lever available: the backbone arrives already
    knowing edges, textures and shapes, so the 5,856 training images only have
    to learn the decision boundary.

    MVP compatibility: MobileNetV2 uses CONV_2D, DEPTHWISE_CONV_2D, ADD (the
    residual connections) and AVERAGE_POOL_2D -- every one of which has an MVP
    kernel. alpha=0.35 keeps the parameter count near 410 k, i.e. ~410 kB of
    INT8 weights, comfortable in the EFR32MG26's 3200 kB flash.

    ImageNet weights expect 3 channels, so the single grayscale plane is
    replicated with a fixed 1x1 convolution rather than by storing three copies
    of every image -- the device still ships one 96x96 uint8 plane per vector.
    """
    inp = keras.Input(shape=input_shape, name="image")

    x = layers.Conv2D(3, 1, use_bias=False, name="gray_to_rgb",
                      trainable=False,
                      kernel_initializer=keras.initializers.Constant(1.0))(inp)

    base = keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        alpha=alpha, include_top=False, weights="imagenet",
    )
    # Freeze the early generic-feature layers; fine-tune the rest.
    for layer in base.layers[:trainable_from]:
        layer.trainable = False

    x = base(x)
    x = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(n_classes, 1, padding="valid", use_bias=True)(x)
    out = layers.Softmax(name="probs")(x)
    return keras.Model(inp, out, name="chiron_mobilenet")


def build_model(input_shape, n_classes: int, width: int = 16,
                n_blocks: int = 4, dropout: float = 0.3,
                arch: str = "dwsep") -> keras.Model:
    """Small depthwise-separable CNN.

    Only operators with MVP-accelerated TFLM kernels are used:
    CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, SOFTMAX.
    BatchNorm folds into the preceding conv at TFLite conversion time.
    """
    if arch == "mobilenet":
        return build_mobilenet(input_shape, n_classes, dropout=dropout)

    inp = keras.Input(shape=input_shape, name="features")
    x = inp

    f = input_shape[0]
    x = layers.Conv2D(width, 3, strides=(2 if f >= 8 else 1, 2),
                      padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    ch = width
    for _ in range(n_blocks - 1):
        ch *= 2
        f = x.shape[1]
        s = (2 if f >= 4 else 1, 2)
        x = layers.SeparableConv2D(ch, 3, strides=s, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)

    # Full-size average pool rather than GlobalAveragePooling2D: AVERAGE_POOL_2D
    # has an MVP kernel, MEAN does not.
    x = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    x = layers.Dropout(dropout)(x)

    # Classifier head as a 1x1 convolution on the (1, 1, ch) tensor.
    #
    # The obvious `Reshape((ch,)) -> Dense` makes Keras emit SHAPE, STRIDED_SLICE
    # and PACK to compute the reshape target at runtime. None of those have MVP
    # kernels, and dynamic shapes are exactly what a static TFLM arena does not
    # want. A 1x1 CONV_2D is numerically identical to the Dense layer and keeps
    # the graph inside the accelerated operator set.
    x = layers.Conv2D(n_classes, 1, padding="valid", use_bias=True)(x)
    out = layers.Softmax(name="probs")(x)
    return keras.Model(inp, out, name="chiron_net")


# ---------------------------------------------------------------------------
# Augmentation (training fold only)
# ---------------------------------------------------------------------------
def augment_images(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Geometric + photometric augmentation for chest films.

    Unlike the lung-audio case (where Sen et al. found augmentation degraded
    results), image augmentation is a well-established win here: patient
    positioning, zoom and exposure genuinely vary between films, so these
    transforms generate plausible new examples rather than corrupting the
    signal.

    Deliberately NO horizontal flip: the heart sits on the left, and mirroring
    a chest film manufactures dextrocardia, which is a real but rare condition.
    Training on it would teach the model that laterality carries no information.
    """
    from scipy.ndimage import affine_transform

    out = np.empty_like(x)
    n, h, w, _ = x.shape
    centre = np.array([h / 2.0, w / 2.0])

    for i in range(n):
        ang = np.deg2rad(rng.uniform(-10, 10))
        zoom = rng.uniform(0.90, 1.10)
        c, s = np.cos(ang) / zoom, np.sin(ang) / zoom
        mat = np.array([[c, -s], [s, c]])
        shift = centre - mat @ centre + rng.uniform(-0.08, 0.08, 2) * np.array([h, w])

        img = affine_transform(x[i, :, :, 0], mat, offset=shift, order=1,
                               mode="nearest")
        img = img * rng.uniform(0.90, 1.10) + rng.uniform(-0.08, 0.08)   # exposure
        out[i, :, :, 0] = img
    return out


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light SpecAugment-style masking plus gain jitter, in feature space.

    Sen et al. (arXiv:2606.10972) found heavy augmentation DEGRADED pulmonary
    sound classification, so this is deliberately mild and is ablated by the
    --no-augment flag.
    """
    x = x.copy()
    n, rows, cols, _ = x.shape
    # gain jitter: an additive offset in the log domain is a multiplicative gain
    x += rng.normal(0.0, 0.10, size=(n, 1, 1, 1)).astype(x.dtype)
    for i in range(n):
        if rng.random() < 0.5:                       # frequency mask
            w = rng.integers(1, max(2, rows // 8))
            s = rng.integers(0, rows - w)
            x[i, s : s + w, :, 0] = 0.0
        if rng.random() < 0.5:                       # time mask
            w = rng.integers(1, max(2, cols // 8))
            s = rng.integers(0, cols - w)
            x[i, :, s : s + w, 0] = 0.0
    return x


def make_dataset(x, y, batch: int, rng, augment: bool, n_classes: int,
                 class_weight: dict | None = None, modality: str = "auto"):
    """Infinite batch generator.

    Keras 3 rejects ``class_weight`` for generator inputs, so class balancing is
    emitted as per-sample weights instead -- mathematically identical.
    """
    # The 1x1-conv head emits (batch, 1, 1, n_classes), so targets carry the
    # same rank -- Keras requires target and output ranks to match.
    y1h = keras.utils.to_categorical(y, n_classes).reshape(len(y), 1, 1, n_classes)
    w = np.ones(len(y), dtype=np.float32)
    if class_weight:
        w = np.array([class_weight[int(v)] for v in y], dtype=np.float32)

    if modality == "auto":
        # square feature maps are images; spectrograms are never square here
        modality = "image" if x.shape[1] == x.shape[2] else "audio"
    aug_fn = augment_images if modality == "image" else augment_batch

    def gen():
        idx = np.arange(len(x))
        while True:
            rng.shuffle(idx)
            for i in range(0, len(idx) - batch + 1, batch):
                sel = idx[i : i + batch]
                xb = x[sel]
                if augment:
                    xb = aug_fn(xb, rng)
                yield xb, y1h[sel], w[sel]

    return gen()


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def run_fold(Xtr, ytr, Xte, yte, pid_te, args, n_classes, seed):
    # rule 5.5 -- normalisation statistics from the training fold only
    mu, sd = float(Xtr.mean()), float(Xtr.std() + 1e-8)
    Xtr = ((Xtr - mu) / sd)[..., None].astype(np.float32)
    Xte = ((Xte - mu) / sd)[..., None].astype(np.float32)

    keras.utils.set_random_seed(seed)
    model = build_model(Xtr.shape[1:], n_classes, args.width, args.blocks,
                        args.dropout, arch=getattr(args, "arch", "dwsep"))

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    cw = {i: float(len(ytr) / (n_classes * max(c, 1))) for i, c in enumerate(counts)}

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    rng = np.random.default_rng(seed)
    steps = max(1, len(Xtr) // args.batch)
    ds = make_dataset(Xtr, ytr, args.batch, rng, args.augment, n_classes, cw)

    model.fit(
        ds,
        steps_per_epoch=steps,
        epochs=args.epochs,
        verbose=0,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=12, min_lr=1e-5, verbose=0
            )
        ],
    )

    prob = model.predict(Xte, verbose=0).reshape(len(Xte), -1)
    win_pred = prob.argmax(1)

    # patient-level: mean softmax across every window of that patient
    dfp = pd.DataFrame(prob, columns=CLASSES[:n_classes])
    dfp["pid"] = pid_te
    dfp["y"] = yte
    grp = dfp.groupby("pid")
    pat_true = grp["y"].first().values
    pat_pred = grp[CLASSES[:n_classes]].mean().values.argmax(1)

    return {
        "model": model,
        "norm": (mu, sd),
        "win_true": yte, "win_pred": win_pred,
        "pat_true": pat_true, "pat_pred": pat_pred,
    }


def summarize(tag, true, pred, n_classes) -> dict:
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro", zero_division=0)
    return {f"{tag}_acc": acc, f"{tag}_macro_f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="kauh", choices=list(DATASET_CLASSES))
    ap.add_argument("--feature", default="logmel", choices=["logmel", "mfcc"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--arch", default="dwsep", choices=["dwsep", "mobilenet"],
                    help="mobilenet = ImageNet-pretrained MobileNetV2 alpha=0.35")
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--split", default="patient", choices=["patient", "window"],
        help="'window' assigns folds randomly per WINDOW, letting windows of the "
             "same recording appear in train and test. That is data leakage and "
             "is provided ONLY to quantify how much published numbers are "
             "inflated by it. Never report a 'window' result as an accuracy.",
    )
    ap.add_argument("--classes", default=None,
                    help="comma-separated subset of classes, e.g. 'Normal,Asthma,Heart Failure'")
    args = ap.parse_args()

    cache = np.load(CACHE_DIR / f"{args.dataset}_{args.feature}.npz", allow_pickle=True)
    X, y = cache["X"], cache["y"]
    pid, fold = cache["patient_id"], cache["fold"]

    global CLASSES
    CLASSES = DATASET_CLASSES[args.dataset]
    if args.classes:
        keep = [c.strip() for c in args.classes.split(",")]
        keep_idx = [CLASSES.index(c) for c in keep]
        m = np.isin(y, keep_idx)
        remap = {old: new for new, old in enumerate(keep_idx)}
        X, pid, fold = X[m], pid[m], fold[m]
        y = np.array([remap[int(v)] for v in y[m]], dtype=np.int64)
        CLASSES = keep

    n_classes = len(CLASSES)

    if args.split == "window":
        rng = np.random.default_rng(0)
        fold = rng.integers(0, len(set(fold.tolist())), size=len(y))
        print("*** WARNING: window-level split -- LEAKY, diagnostic only ***")

    folds = sorted(set(fold.tolist()))

    tag = args.tag or (f"{args.dataset}_{args.feature}_{args.arch}_w{args.width}_b{args.blocks}" + ("" if args.augment else "_noaug") + ("" if args.split == "patient" else "_LEAKY"))
    print(f"=== {tag} ===")
    print(f"X {X.shape}  {len(set(pid))} patients  {len(folds)} folds  "
          f"augment={args.augment}  seeds={args.seeds}")

    n_params = build_model((*X.shape[1:], 1), n_classes, args.width, args.blocks,
                           args.dropout, arch=args.arch).count_params()
    print(f"model params: {n_params:,}\n")

    rows = []
    all_win_t, all_win_p, all_pat_t, all_pat_p = [], [], [], []

    for seed in range(args.seeds):
        for f in folds:
            tr, te = fold != f, fold == f
            r = run_fold(X[tr], y[tr], X[te], y[te], pid[te],
                         args, n_classes, seed * 100 + f)
            m = {"seed": seed, "fold": f}
            m.update(summarize("win", r["win_true"], r["win_pred"], n_classes))
            m.update(summarize("pat", r["pat_true"], r["pat_pred"], n_classes))
            rows.append(m)
            all_win_t.append(r["win_true"]); all_win_p.append(r["win_pred"])
            all_pat_t.append(r["pat_true"]); all_pat_p.append(r["pat_pred"])
            print(f"  seed {seed} fold {f}: "
                  f"window acc {m['win_acc']:.3f} F1 {m['win_macro_f1']:.3f} | "
                  f"patient acc {m['pat_acc']:.3f} F1 {m['pat_macro_f1']:.3f}")

    df = pd.DataFrame(rows)
    wt = np.concatenate(all_win_t); wp = np.concatenate(all_win_p)
    pt = np.concatenate(all_pat_t); pp = np.concatenate(all_pat_p)

    print("\n" + "=" * 64)
    print(f"RESULT  {tag}   ({args.seeds} seeds x {len(folds)} folds)")
    print("=" * 64)
    for lvl in ("win", "pat"):
        name = "WINDOW " if lvl == "win" else "PATIENT"
        print(f"{name} accuracy  {df[lvl+'_acc'].mean():.4f} +/- {df[lvl+'_acc'].std():.4f}"
              f"   macro-F1 {df[lvl+'_macro_f1'].mean():.4f} +/- {df[lvl+'_macro_f1'].std():.4f}")

    print("\n--- patient-level, pooled over all folds and seeds ---")
    print(classification_report(pt, pp, target_names=CLASSES, digits=3, zero_division=0))
    cm = confusion_matrix(pt, pp)
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    REPORT_DIR.mkdir(exist_ok=True)
    out = {
        "tag": tag, "feature": args.feature, "augment": args.augment,
        "params": int(n_params), "seeds": args.seeds, "folds": len(folds),
        "window_acc_mean": float(df.win_acc.mean()), "window_acc_std": float(df.win_acc.std()),
        "window_f1_mean": float(df.win_macro_f1.mean()),
        "patient_acc_mean": float(df.pat_acc.mean()), "patient_acc_std": float(df.pat_acc.std()),
        "patient_f1_mean": float(df.pat_macro_f1.mean()),
        "confusion_matrix_patient": cm.tolist(), "classes": CLASSES,
    }
    (REPORT_DIR / f"cv_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote reports/cv_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
