"""Train a final model on all data, convert to fully-integer INT8 TFLite, and
report the float -> int8 accuracy delta.

Quantisation rules (docs/06_MODEL_DESIGN.md section 4):
  * full-integer INT8, int8 input AND output (no float conversion on device)
  * the representative dataset comes from the TRAINING fold only -- test data
    must never reach the quantiser
  * it must span every class and the full loudness range
  * the float -> int8 drop is measured and reported, not assumed

Usage:
    python -m core.quantize --dataset yaseen --holdout-fold 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras

from core.train import DATASET_CLASSES, build_model, make_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"


def representative_dataset_factory(Xtr: np.ndarray, ytr: np.ndarray,
                                   n_classes: int, per_class: int = 60):
    """Class-balanced sample of TRAINING windows spanning the loudness range."""
    rng = np.random.default_rng(0)
    picks = []
    for c in range(n_classes):
        idx = np.flatnonzero(ytr == c)
        if len(idx) == 0:
            continue
        # sort by energy so the sample spans quiet -> loud rather than clustering
        energy = Xtr[idx].mean(axis=(1, 2, 3))
        order = idx[np.argsort(energy)]
        take = np.linspace(0, len(order) - 1, min(per_class, len(order))).astype(int)
        picks.append(order[take])
    sel = np.concatenate(picks)
    rng.shuffle(sel)

    def gen():
        for i in sel:
            yield [Xtr[i : i + 1].astype(np.float32)]

    return gen, len(sel)


def tflite_predict(interp: tf.lite.Interpreter, X: np.ndarray) -> np.ndarray:
    """Run the quantised model window by window, mirroring on-device execution."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    n_out = int(np.prod(out["shape"][1:]))
    preds = np.zeros((len(X), n_out), dtype=np.float32)
    for i in range(len(X)):
        q = np.round(X[i] / in_scale + in_zp).astype(inp["dtype"])
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        raw = interp.get_tensor(out["index"]).reshape(-1).astype(np.float32)
        preds[i] = (raw - out_zp) * out_scale
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="yaseen", choices=list(DATASET_CLASSES))
    ap.add_argument("--feature", default="logmel")
    ap.add_argument("--holdout-fold", type=int, default=0,
                    help="fold reserved for evaluation and golden-vector export")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="dwsep", choices=["dwsep", "mobilenet"])
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    classes = DATASET_CLASSES[args.dataset]
    n_classes = len(classes)

    cache = np.load(CACHE_DIR / f"{args.dataset}_{args.feature}.npz", allow_pickle=True)
    X, y, pid, fold = cache["X"], cache["y"], cache["patient_id"], cache["fold"]

    tr, te = fold != args.holdout_fold, fold == args.holdout_fold
    mu, sd = float(X[tr].mean()), float(X[tr].std() + 1e-8)
    Xtr = ((X[tr] - mu) / sd)[..., None].astype(np.float32)
    Xte = ((X[te] - mu) / sd)[..., None].astype(np.float32)
    ytr, yte = y[tr], y[te]

    print(f"=== {args.dataset} / {args.feature} ===")
    print(f"train {Xtr.shape}  holdout fold {args.holdout_fold}: {Xte.shape}")
    print(f"normalisation (training fold only): mean {mu:.4f} std {sd:.4f}")

    # ---- train -------------------------------------------------------------
    keras.utils.set_random_seed(args.seed)
    model = build_model(Xtr.shape[1:], n_classes, args.width, args.blocks,
                        args.dropout, arch=args.arch)
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    cw = {i: float(len(ytr) / (n_classes * max(c, 1))) for i, c in enumerate(counts)}
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    rng = np.random.default_rng(args.seed)
    model.fit(
        make_dataset(Xtr, ytr, args.batch, rng, args.augment, n_classes, cw),
        steps_per_epoch=max(1, len(Xtr) // args.batch),
        epochs=args.epochs, verbose=0,
        callbacks=[keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=12, min_lr=1e-5)],
    )

    pid_te = pid[te]

    def patient_level(prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate per-item probabilities to one prediction per subject.

        This is the metric the cross-validation headline uses, so the INT8
        number must be computed the same way or the two are not comparable.
        """
        d = pd.DataFrame(prob, columns=classes[: prob.shape[1]])
        d["pid"], d["y"] = pid_te, yte
        g = d.groupby("pid")
        return g["y"].first().values, g[classes[: prob.shape[1]]].mean().values.argmax(1)

    float_prob = model.predict(Xte, verbose=0).reshape(len(Xte), -1)
    float_pred = float_prob.argmax(1)
    float_acc = accuracy_score(yte, float_pred)
    fp_true, fp_pred = patient_level(float_prob)
    float_pat_acc = accuracy_score(fp_true, fp_pred)
    print(f"\nfloat32 holdout accuracy: {float_acc:.4f} (item) "
          f"{float_pat_acc:.4f} (patient)")

    # ---- convert -----------------------------------------------------------
    rep_gen, rep_n = representative_dataset_factory(Xtr, ytr, n_classes)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite = conv.convert()
    print(f"representative dataset: {rep_n} training windows")

    MODEL_DIR.mkdir(exist_ok=True)
    stem = f"chiron_{'cardio' if args.dataset == 'yaseen' else 'pulmo'}_v1"
    tfl_path = MODEL_DIR / f"{stem}_int8.tflite"
    tfl_path.write_bytes(tflite)
    model.save(MODEL_DIR / f"{stem}_float.keras")

    # ---- evaluate int8 ------------------------------------------------------
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    q_prob = tflite_predict(interp, Xte)
    q_pred = q_prob.argmax(1)
    q_acc = accuracy_score(yte, q_pred)
    agree = float((q_pred == float_pred).mean())

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    qp_true, qp_pred = patient_level(q_prob)
    q_pat_acc = accuracy_score(qp_true, qp_pred)

    print(f"int8    holdout accuracy: {q_acc:.4f} (item) {q_pat_acc:.4f} (patient)")
    print(f"float -> int8 delta:      {(q_acc - float_acc)*100:+.2f} pp (item)  "
          f"{(q_pat_acc - float_pat_acc)*100:+.2f} pp (patient)")
    print(f"prediction agreement:     {agree*100:.2f} %")
    print(f"*** DEPLOYED INT8 PATIENT-LEVEL ACCURACY: {q_pat_acc:.4f} ***")
    print(f"\nmodel size: {len(tflite)/1024:.1f} kB  (params {model.count_params():,})")
    print(f"input  {inp['shape']} {np.dtype(inp['dtype']).name} "
          f"scale {inp['quantization'][0]:.6f} zp {inp['quantization'][1]}")
    print(f"output {out['shape']} {np.dtype(out['dtype']).name} "
          f"scale {out['quantization'][0]:.6f} zp {out['quantization'][1]}")

    ops = {d["op_name"] for d in interp._get_ops_details()} if hasattr(interp, "_get_ops_details") else set()
    if ops:
        print(f"operators: {sorted(ops)}")

    print("\n--- int8 holdout report ---")
    print(classification_report(yte, q_pred, target_names=classes, digits=3, zero_division=0))
    print(pd.DataFrame(confusion_matrix(yte, q_pred), index=classes, columns=classes).to_string())

    REPORT_DIR.mkdir(exist_ok=True)
    rep = {
        "dataset": args.dataset, "feature": args.feature, "holdout_fold": args.holdout_fold,
        "params": int(model.count_params()), "tflite_bytes": len(tflite),
        "float_acc": float(float_acc), "int8_acc": float(q_acc),
        "float_patient_acc": float(float_pat_acc), "int8_patient_acc": float(q_pat_acc),
        "delta_pp": float((q_acc - float_acc) * 100),
        "delta_patient_pp": float((q_pat_acc - float_pat_acc) * 100),
        "prediction_agreement": agree,
        "representative_windows": int(rep_n),
        "norm_mean": mu, "norm_std": sd,
        "input_scale": float(inp["quantization"][0]), "input_zero_point": int(inp["quantization"][1]),
        "output_scale": float(out["quantization"][0]), "output_zero_point": int(out["quantization"][1]),
        "operators": sorted(ops), "classes": classes,
    }
    (REPORT_DIR / f"quantization_{stem}.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {tfl_path}")
    print(f"wrote reports/quantization_{stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
