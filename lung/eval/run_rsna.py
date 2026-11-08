"""S2: run the final X-ray model on RSNA (a dataset it never saw).

The Kermany mean/std are used for normalisation, never RSNA's own.
Reports AUROC, sensitivity at 90% specificity, calibration error and accuracy, each with a
95% bootstrap interval, for the float and 8-bit models, plus the same on Kermany fold 0
(the held-out fold) so "how much was lost" can be read off directly.

Usage:
    python -m lung.eval.run_rsna
Writes:
    reports/xray_cnn_rsna.csv   per-image probabilities (used by the fusion script)
    reports/xray_rsna.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lung.eval.common import (REPORT_DIR, load_float_model, load_int8, load_kermany, load_norm,
                              load_rsna, predict_float, predict_int8)
from lung.eval.metrics import evaluate_predictions


def main() -> int:
    mu, sd = load_norm()
    model, interp = load_float_model(), load_int8()
    print(f"normalisation (Kermany): mean {mu:.4f}, std {sd:.4f}")

    man, X = load_rsna()
    y, pid = man.label.to_numpy(int), man.patient_id.to_numpy()
    p_float = predict_float(model, X, mu, sd)
    p_int8 = predict_int8(interp, X, mu, sd)

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"patient_id": pid, "fold": man.fold, "label": y, "age": man.age, "sex": man.sex,
                  "p_float": np.round(p_float, 6), "p_int8": np.round(p_int8, 6)}
                 ).to_csv(REPORT_DIR / "xray_cnn_rsna.csv", index=False)

    # Kermany fold 0: images the final model never trained on
    Xk, yk, pk, fk = load_kermany()
    m0 = fk == 0
    pk_float = predict_float(model, Xk[m0], mu, sd)

    rsna_f = evaluate_predictions(pid, y, p_float)
    rsna_q = evaluate_predictions(pid, y, p_int8)
    kerm_f = evaluate_predictions(pk[m0], yk[m0], pk_float)
    out = {
        "what": "Final X-ray CNN (trained on Kermany folds 1-4) run on RSNA Normal vs Lung Opacity, "
                "one image per patient, Kermany normalisation. No RSNA label was used for anything but scoring.",
        "n_images": int(len(y)),
        "pneumonia_share": float(y.mean()),
        "rsna_float": rsna_f,
        "rsna_int8": rsna_q,
        "class_agreement_float_vs_int8": float(np.mean((p_float >= 0.5) == (p_int8 >= 0.5))),
        "kermany_fold0_float": kerm_f,
        "auroc_lost_vs_kermany_fold0": float(kerm_f["auroc"] - rsna_f["auroc"]),
    }
    (REPORT_DIR / "xray_rsna.json").write_text(json.dumps(out, indent=2))

    def line(tag, r):
        lo, hi = r["auroc_ci_95"]
        print(f"{tag:22s} AUROC {r['auroc']:.4f} [{lo:.4f}, {hi:.4f}]  acc {r['accuracy']:.4f}  "
              f"sens@90spec {r['sensitivity_at_90_specificity']:.4f}  ECE {r['ece']:.4f}")
    line("RSNA float", rsna_f)
    line("RSNA int8", rsna_q)
    line("Kermany fold 0 float", kerm_f)
    print(f"float/8-bit class agreement on RSNA: {out['class_agreement_float_vs_int8']:.4f}")
    print("wrote reports/xray_cnn_rsna.csv and reports/xray_rsna.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
