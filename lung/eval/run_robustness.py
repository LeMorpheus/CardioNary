"""S4: plain vs adapted accuracy for all four corruptions at all five severities.

Data: Kermany fold 0 (never trained on). Each image is corrupted deterministically
(seed = image index). N = 32 corrupted images adapt the model's batch-normalisation
statistics (lung.tta.bn_adapt); every image of those patients is then left out of scoring,
so nothing used to adapt is ever scored. Plain and adapted models are scored on the same
remaining images.  recovery = (adapted - plain) / (clean - plain).

Usage:
    python -m lung.eval.run_robustness
    python -m lung.eval.run_robustness --n-adapt 64 --out reports/xray_robustness.json
Writes:
    reports/xray_robustness.json, reports/figures/xray_robustness.png
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from lung.eval.common import (REPORT_DIR, load_float_model, load_kermany, load_norm, normalise,
                              predict_float)
from lung.eval.corrupt import SEVERITIES, apply_corruption
from lung.eval.metrics import evaluate_predictions
from lung.tta.bn_adapt import apply_bn_adaptation


def score(pid, y, p) -> dict:
    r = evaluate_predictions(pid, y, p, n_bootstrap=1000, seed=0)
    r["image_accuracy"] = float(np.mean((p >= 0.5) == y))
    return r


def recovery(clean, plain, adapted):
    d = clean - plain
    return None if abs(d) < 1e-3 else float((adapted - plain) / d)


def plot(res: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, c in zip(axes, SEVERITIES):
        sev = sorted(int(s) for s in res["results"][c])
        for key, colour, label in (("plain", "#0645ad", "plain"), ("adapted", "#2a7f62", "adapted")):
            ax.plot(sev, [res["results"][c][str(s)][key]["accuracy"] for s in sev], marker="o",
                    color=colour, label=label)
        ax.axhline(res["clean"]["accuracy"], color="#888888", linestyle="--", linewidth=1, label="clean")
        ax.set_title(c); ax.set_xlabel("severity"); ax.set_xticks(sev)
    axes[0].set_ylabel("per-patient accuracy"); axes[0].legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-adapt", type=int, default=32)
    ap.add_argument("--out", default=str(REPORT_DIR / "xray_robustness.json"))
    args = ap.parse_args()

    X, y, pid, fold = load_kermany()
    m0 = fold == 0
    X, y, pid = X[m0], y[m0], pid[m0]
    mu, sd = load_norm()
    model = load_float_model()

    rng = np.random.default_rng(0)
    adapt_idx = rng.permutation(len(X))[:args.n_adapt]
    ev = ~np.isin(pid, pid[adapt_idx])          # no patient used for adapting is scored
    print(f"Kermany fold 0: {len(X)} images; adapt on {len(adapt_idx)}; "
          f"score {int(ev.sum())} images of {len(set(pid[ev]))} patients")

    clean = score(pid[ev], y[ev], predict_float(model, X[ev], mu, sd))
    print(f"clean: accuracy {clean['accuracy']:.4f}  AUROC {clean['auroc']:.4f}")

    results = {}
    for c in SEVERITIES:
        results[c] = {}
        for s in range(1, 6):
            Xc = np.stack([apply_corruption(X[i], c, s, seed=int(i)) for i in range(len(X))])
            plain = score(pid[ev], y[ev], predict_float(model, Xc[ev], mu, sd))
            adapted_model, collapsed = apply_bn_adaptation(model, normalise(Xc[adapt_idx], mu, sd))
            adapted = score(pid[ev], y[ev], predict_float(adapted_model, Xc[ev], mu, sd))
            results[c][str(s)] = {
                "plain": plain, "adapted": adapted, "collapse_guard_triggered": bool(collapsed),
                "recovery": {k: recovery(clean[k], plain[k], adapted[k]) for k in ("accuracy", "auroc")},
            }
            print(f"{c:9s} severity {s}: plain acc {plain['accuracy']:.3f}  adapted acc "
                  f"{adapted['accuracy']:.3f}   AUROC {plain['auroc']:.3f} -> {adapted['auroc']:.3f}"
                  + ("   [collapse guard]" if collapsed else ""))

    out = {"data": "Kermany fold 0 (held out from the final model)", "n_adapt": int(args.n_adapt),
           "scored_images": int(ev.sum()), "scored_patients": int(len(set(pid[ev]))),
           "seed": 0, "clean": clean, "results": results}
    from pathlib import Path
    Path(args.out).write_text(json.dumps(out, indent=2))
    plot(out, REPORT_DIR / "figures" / "xray_robustness.png")
    print(f"wrote {args.out} and reports/figures/xray_robustness.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
