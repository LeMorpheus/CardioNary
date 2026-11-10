"""Late fusion: combine CNN, feature-vote and metadata-vote probabilities.

Fusion rule
-----------
    p = w_cnn * p_CNN  +  w_lr * p_LR  +  w_m * p_M

where the weights for fold k are fitted by optimising log-loss on the
complementary folds (the "other-fold" strategy, fully respecting the
anti-leakage requirement).

Missing votes
~~~~~~~~~~~~~
If a vote is unavailable for an image (e.g. no metadata), its weight is
forced to 0 and the remaining weights are renormalised to sum to 1.

Contracts
~~~~~~~~~
Every fused result is returned as a ``FusedOutput("xray", ...)`` object
from ``core.contracts``, which asserts:

  * weights sum to 1
  * the metadata weight never exceeds METADATA_WEIGHT_CAP (0.3)
  * the CNN vote is always present

Kermany dataset
~~~~~~~~~~~~~~~
Kermany has no patient metadata, so fusion is CNN + feature vote only
(w_m = 0, automatically enforced by the missing-vote logic).

Outputs
-------
    reports/xray_fused_kermany.csv    -- image-level fused probabilities
    reports/xray_fused_rsna.csv       -- image-level fused probabilities
    reports/xray_fused_kermany.json   -- AUROC summary (Kermany)
    reports/xray_fused_rsna.json      -- AUROC summary (RSNA)

Usage
-----
    python -m lung.fusion.late_fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.contracts import FusedOutput, METADATA_WEIGHT_CAP
from lung.eval.common import choose_weights, fuse

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports"


def fuse_dataset(
    df_oof: pd.DataFrame,
    df_plr: Optional[pd.DataFrame],
    df_pm: Optional[pd.DataFrame],
    dataset_name: str,
    out_csv: Path,
    out_json: Path,
) -> list[FusedOutput]:
    """Perform out-of-fold late fusion and return list of FusedOutput objects.

    Parameters
    ----------
    df_oof  : CNN OOF predictions (needs: index, patient_id, fold, label, p_pneumonia)
    df_plr  : feature-vote OOF predictions (needs: pLR, same row order as df_oof);
              None = not available
    df_pm   : metadata-vote OOF predictions (needs: pM, same row order as df_oof);
              None = not available
    dataset_name : used only in progress messages
    out_csv : destination for per-image fused probabilities
    out_json: destination for AUROC summary

    Both ``df_plr`` and ``df_pm`` are assumed to be **row-aligned** with
    ``df_oof`` (same images in the same order), since all three are generated
    from the same underlying manifest.
    """
    # Add a contiguous positional key so we can splice per fold
    df_oof = df_oof.reset_index(drop=True).copy()
    df_oof["_pos_"] = np.arange(len(df_oof))

    if df_plr is not None:
        df_plr = df_plr.reset_index(drop=True).copy()
        df_plr["_pos_"] = np.arange(len(df_plr))

    if df_pm is not None:
        df_pm = df_pm.reset_index(drop=True).copy()
        df_pm["_pos_"] = np.arange(len(df_pm))
    folds = sorted(df_oof["fold"].unique())
    rows = []
    fused_outputs: list[FusedOutput] = []

    for f in folds:
        te_mask = df_oof["fold"] == f
        tr_mask = ~te_mask

        # --- Extract probabilities for train / test splits ---
        p_cnn_tr = df_oof.loc[tr_mask, "p_pneumonia"].to_numpy()
        p_cnn_te = df_oof.loc[te_mask, "p_pneumonia"].to_numpy()
        y_tr = df_oof.loc[tr_mask, "label"].to_numpy()

        # Align feature-vote using positional key (both CSVs are row-aligned)
        p_lr_tr = p_lr_te = None
        if df_plr is not None:
            pos_tr = df_oof.loc[tr_mask, "_pos_"].to_numpy()
            pos_te = df_oof.loc[te_mask, "_pos_"].to_numpy()
            p_lr_tr = df_plr.loc[pos_tr, "pLR"].to_numpy()
            p_lr_te = df_plr.loc[pos_te, "pLR"].to_numpy()

        # Align metadata vote using positional key
        p_m_tr = p_m_te = None
        if df_pm is not None:
            pos_tr = df_oof.loc[tr_mask, "_pos_"].to_numpy()
            pos_te = df_oof.loc[te_mask, "_pos_"].to_numpy()
            p_m_tr = df_pm.loc[pos_tr, "pM"].to_numpy()
            p_m_te = df_pm.loc[pos_te, "pM"].to_numpy()

        # --- Fit weights on other folds only (anti-leakage) ---
        votes_tr = {"cnn": p_cnn_tr}
        votes_te = {"cnn": p_cnn_te}
        if p_lr_tr is not None and p_lr_te is not None:
            votes_tr["features"] = p_lr_tr
            votes_te["features"] = p_lr_te
        if p_m_tr is not None and p_m_te is not None:
            votes_tr["metadata"] = p_m_tr
            votes_te["metadata"] = p_m_te

        weights_dict = choose_weights(
            votes=votes_tr,
            y=y_tr,
            units=20,
            metadata_cap=METADATA_WEIGHT_CAP,
        )

        # Available votes for this test fold
        available_names = list(votes_te.keys())

        # Missing votes are left out; present weights are normalised to sum to 1
        total_w = sum(weights_dict.get(k, 0.0) for k in available_names)
        if total_w > 0:
            norm_weights = {k: float(weights_dict.get(k, 0.0) / total_w) for k in available_names}
        else:
            # Fallback to equal weighting among available votes
            norm_weights = {k: float(1.0 / len(available_names)) for k in available_names}

        # --- Fuse predictions on test fold ---
        p_fused_te = fuse(votes_te, norm_weights)

        # --- Build FusedOutput objects ---
        te_df = df_oof[te_mask].reset_index(drop=True)
        for i, row in enumerate(te_df.itertuples()):
            p_i = float(p_fused_te[i])

            # Present votes dict matching present weights
            votes_dict = {k: float(votes_te[k][i]) for k in available_names}

            fo = FusedOutput(
                modality="xray",
                p_abnormal=float(np.clip(p_i, 0.0, 1.0)),
                votes=votes_dict,
                weights=norm_weights,
            )
            fused_outputs.append(fo)

            rows.append({
                "index": int(row.index),
                "patient_id": str(row.patient_id),
                "fold": int(f),
                "label": int(row.label),
                "p_fused": float(fo.p_abnormal),
                "w_cnn": norm_weights.get("cnn", 0.0),
                "w_features": norm_weights.get("features", 0.0),
                "w_metadata": norm_weights.get("metadata", 0.0),
            })

    df_out = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)

    # --- AUROC ---
    auroc_image = roc_auc_score(df_out["label"], df_out["p_fused"])

    patient_grp = df_out.groupby("patient_id").agg(
        label=("label", "first"),
        p_fused=("p_fused", "mean"),
    )
    auroc_patient = roc_auc_score(patient_grp["label"], patient_grp["p_fused"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    summary = {
        "dataset": dataset_name,
        "n_images": int(len(df_out)),
        "n_patients": int(len(patient_grp)),
        "votes_used": list(weights_dict.keys()),
        "auroc_image_level": round(auroc_image, 4),
        "auroc_patient_level": round(auroc_patient, 4),
    }
    out_json.write_text(json.dumps(summary, indent=2))

    return fused_outputs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["kermany", "rsna", "both"],
        default="both",
        help="Which dataset to fuse",
    )
    args = parser.parse_args()

    # ------- Kermany fusion (CNN + feature vote; no metadata) -------
    if args.dataset in ("kermany", "both"):
        oof_path = REPORT_DIR / "xray_oof_kermany.csv"
        plr_path = REPORT_DIR / "xray_plr_kermany.csv"

        if not oof_path.exists():
            print(f"[SKIP Kermany] CNN OOF not found: {oof_path}")
        else:
            df_oof = pd.read_csv(oof_path)
            df_plr = pd.read_csv(plr_path) if plr_path.exists() else None

            print(f"Fusing Kermany: CNN {'+ features' if df_plr is not None else 'only'} ...")
            fo_list = fuse_dataset(
                df_oof=df_oof,
                df_plr=df_plr,
                df_pm=None,  # no metadata in Kermany
                dataset_name="kermany",
                out_csv=REPORT_DIR / "xray_fused_kermany.csv",
                out_json=REPORT_DIR / "xray_fused_kermany.json",
            )
            summary = json.loads((REPORT_DIR / "xray_fused_kermany.json").read_text())
            print(f"  Votes: {summary['votes_used']}")
            print(f"  Image-level AUROC:   {summary['auroc_image_level']:.4f}")
            print(f"  Patient-level AUROC: {summary['auroc_patient_level']:.4f}")
            print(f"  FusedOutput objects: {len(fo_list)}")

    # ------- RSNA fusion (CNN + feature vote + metadata) -------
    if args.dataset in ("rsna", "both"):
        oof_path = REPORT_DIR / "xray_oof_rsna.csv"
        plr_path = REPORT_DIR / "xray_plr_rsna.csv"
        pm_path = REPORT_DIR / "xray_pm_rsna.csv"

        if not oof_path.exists():
            print(f"[SKIP RSNA] CNN OOF not found: {oof_path}")
        else:
            df_oof = pd.read_csv(oof_path)
            df_plr = pd.read_csv(plr_path) if plr_path.exists() else None
            df_pm = pd.read_csv(pm_path) if pm_path.exists() else None

            votes_desc = "CNN"
            if df_plr is not None:
                votes_desc += " + features"
            if df_pm is not None:
                votes_desc += " + metadata"
            print(f"Fusing RSNA: {votes_desc} ...")

            fo_list = fuse_dataset(
                df_oof=df_oof,
                df_plr=df_plr,
                df_pm=df_pm,
                dataset_name="rsna",
                out_csv=REPORT_DIR / "xray_fused_rsna.csv",
                out_json=REPORT_DIR / "xray_fused_rsna.json",
            )
            summary = json.loads((REPORT_DIR / "xray_fused_rsna.json").read_text())
            print(f"  Votes: {summary['votes_used']}")
            print(f"  Image-level AUROC:   {summary['auroc_image_level']:.4f}")
            print(f"  Patient-level AUROC: {summary['auroc_patient_level']:.4f}")
            print(f"  FusedOutput objects: {len(fo_list)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
