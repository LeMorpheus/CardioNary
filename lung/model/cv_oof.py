
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from core.train import (
    CACHE_DIR,
    DATASET_CLASSES,
    REPORT_DIR,
    run_fold,
)

CLASSES = DATASET_CLASSES["kermany"]  # ["Normal", "Pneumonia"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Kermany OOF predictions.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()

   
    args.dataset = "kermany"
    args.feature = "logmel"
    args.arch = "mobilenet"
    args.augment = True

    cache_file = CACHE_DIR 
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cache: {cache_file}. Run core.preprocess first.")

    cache = np.load(cache_file, allow_pickle=True)
    X, y = cache["X"], cache["y"]
    pid, fold = cache["patient_id"], cache["fold"]
    n_classes = len(CLASSES)

    folds = sorted(set(fold.tolist()))
    print(f"Loaded Kermany cache: {X.shape[0]} images across {len(set(pid))} patients, {len(folds)} folds.")

    
    oof_rows = []

    for f in folds:
        print(f"\n--- Running Fold {f} ---")
        tr_mask = fold != f
        te_mask = fold == f

        
        result = run_fold(
            X[tr_mask], y[tr_mask],
            X[te_mask], y[te_mask],
            pid[te_mask],
            args, n_classes, seed=f
        )

        model = result["model"]
        mu, sd = result["norm"]

        
        te_indices = np.where(te_mask)[0]
        Xte_norm = ((X[te_mask] - mu) / sd)[..., None].astype(np.float32)

        
        probs = model.predict(Xte_norm, verbose=0).reshape(len(Xte_norm), -1)

        for idx, p_id, true_y, p in zip(te_indices, pid[te_mask], y[te_mask], probs):
            oof_rows.append({
                "index": int(idx),
                "patient_id": str(p_id),
                "fold": int(f),
                "label": int(true_y),
                "p_normal": float(p[0]),
                "p_pneumonia": float(p[1]),
            })

    
    df_oof = pd.DataFrame(oof_rows).sort_values("index").reset_index(drop=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORT_DIR / "xray_oof_kermany.csv"
    df_oof.to_csv(out_csv, index=False)
    print(f"\nWrote out-of-fold predictions to {out_csv}")

   
    patient_grp = df_oof.groupby("patient_id").agg({
        "label": "first",
        "p_normal": "mean",
        "p_pneumonia": "mean",
    })
    patient_pred = (patient_grp["p_pneumonia"] > patient_grp["p_normal"]).astype(int)
    patient_acc = (patient_pred == patient_grp["label"]).mean()

    print(f"Overall Per-Patient Accuracy: {patient_acc * 100:.2f}%")
    if patient_acc < 0.90:
        print("Accuracy below 0.9")
    else:
        print("PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
