"""Re-quantise the heart model for a new site (roadmap S4, proposal 4.3 and 4.5).

On new devices the 8-bit model agrees less often with the float model (PhysioNet
2022 96.5%, the clinic recordings 94%, Yaseen fold 0 100%). Its 8-bit input range
was fitted to Yaseen windows, so inputs from another device that fall outside it
are clipped. Proposal 4.5: adaptation updates the full-precision model, which is
then re-quantised on site in seconds.

Each site is split by patient into two halves. N unlabelled recordings from one
half stand for the site's first cases: they adapt the model and form the
representative set. The other half is measured; then the halves swap. Three
8-bit models are compared on the measured half, each against its own float model:

  A  original      the shipped 8-bit model (representative set from Yaseen)
  B  requantised   the original float model, re-quantised with the site's windows
  C  adapted       the float model after batch-norm re-estimation on the site's
                   windows (proposal 4.3), re-quantised with the same windows

For each: recording and window agreement with a 95% bootstrap interval over
patients; agreement on the refer-or-not decision; agreement split by how far the
float model's top class leads the runner-up; the share of normalised input values
clipped by more than half an 8-bit step; conversion time, file size and time per
window.

The clinic recordings follow their data rules: read in place, the re-quantised
models exist only in memory, and only totals and averages are written.

Usage:
    python -m cardio.data.build_unseen --dataset circor     # PhysioNet caches, once
    python -m cardio.eval.requantise --site physionet_2022
    python -m cardio.eval.requantise --site clinic --data-dir <path to Real_World_Data>

Writes (merging with earlier runs):
    reports/heart_requantisation.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from cardio.eval.agreement import manifest_rows
from cardio.eval.intervals import agreement_interval
from cardio.eval.real_world import NAME as CLINIC_NAME
from cardio.eval.real_world import adapt_bn
from cardio.model.final_model import FinalModel
from core.features import CARDIO_CONFIG, features_for_file
from core.quantize import tflite_predict

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "reports" / "heart_requantisation.json"
N_ADAPT = 64          # proposal 4.3: the site's first N cases, N between 8 and 128
REP_WINDOWS = 300     # the number of representative windows core/quantize.py uses
VARIANTS = ["A_original", "B_requantised", "C_adapted_requantised"]


def to_int8(net, rep: np.ndarray) -> bytes:
    """Full-integer 8-bit conversion with core/quantize.py's settings; only the
    representative windows (already normalised) come from the site."""
    import tensorflow as tf

    def gen():
        for x in rep:
            yield [x[None, ...].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(net)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def interpreter(blob: bytes):
    import tensorflow as tf

    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    return it


def input_range(it) -> tuple[float, float, float]:
    """The normalised input values an 8-bit model can represent, and its step."""
    scale, zp = it.get_input_details()[0]["quantization"]
    return (-128 - zp) * scale, (127 - zp) * scale, scale


def clipped(x: np.ndarray, it) -> int:
    """Input values the 8-bit model clips by more than half a step. Values at the
    log-mel floor sit a hair below the calibrated minimum and quantise to it
    exactly, so they do not count."""
    lo, hi, step = input_range(it)
    return int(np.sum((x < lo - step / 2) | (x > hi + step / 2)))


MARGIN_BINS = [(0.0, 0.05), (0.05, 0.2), (0.2, 1.01)]     # float top-1 minus top-2 probability


def spread_by_loudness(x: np.ndarray, k: int) -> np.ndarray:
    """Up to k windows evenly spaced from quiet to loud, as core/quantize.py picks."""
    if len(x) <= k:
        return x
    order = np.argsort(x.mean(axis=(1, 2, 3)))
    return x[order[np.linspace(0, len(order) - 1, k).astype(int)]]


def site_rows(site: str, data_dir: Path | None) -> tuple[list[dict], list[set]]:
    """Recordings (patient_id, load) and the two patient halves."""
    if site == "physionet_2022":
        rows = manifest_rows(REPO_ROOT / "data" / "manifests" / "circor_manifest.csv")
        patients = sorted({r["patient_id"] for r in rows})
        order = np.random.default_rng(0).permutation(len(patients))
        half = len(patients) // 2
        halves = [{patients[i] for i in order[:half]}, {patients[i] for i in order[half:]}]
    elif site == "clinic":
        files = sorted(f for f in data_dir.glob("p*_*.wav") if CLINIC_NAME.match(f.name))
        rows = [{"patient_id": CLINIC_NAME.match(f.name).group(1),
                 "load": (lambda f=f: features_for_file(str(f), CARDIO_CONFIG))} for f in files]
        patients = sorted({r["patient_id"] for r in rows}, key=lambda s: int(s[1:]))
        halves = [set(patients[: len(patients) // 2]), set(patients[len(patients) // 2:])]   # as in S3
    else:
        raise ValueError(site)
    return rows, halves


def run_site(site: str, data_dir: Path | None, model: FinalModel) -> dict:
    rows, halves = site_rows(site, data_dir)
    acc = {v: {"rf": [], "rq": [], "groups": [], "win": 0, "same": 0, "out": 0, "vals": 0, "sec": 0.0}
           for v in VARIANTS}
    halves_info = []
    for adapt_on in halves:
        pool = [r for r in rows if r["patient_id"] in adapt_on]
        if site == "physionet_2022":
            pick = np.random.default_rng(0).permutation(len(pool))[:N_ADAPT]
            pool = [pool[i] for i in sorted(pick)]
        feats = [f for f in (r["load"]() for r in pool) if len(f)]
        windows = np.concatenate(feats)
        rep = spread_by_loudness(model.normalise(windows), REP_WINDOWS)

        t0 = time.perf_counter()
        blob_b = to_int8(model.float_model, rep)
        t_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        adapted = adapt_bn(windows, model)
        blob_c = to_int8(adapted, rep)
        t_c = time.perf_counter() - t0
        nets = {"A_original": (model.float_model, model.interp),
                "B_requantised": (model.float_model, interpreter(blob_b)),
                "C_adapted_requantised": (adapted, interpreter(blob_c))}
        halves_info.append({
            "adaptation_recordings": len(feats), "adaptation_windows": int(len(windows)),
            "representative_windows": int(len(rep)),
            "seconds_to_requantise": round(t_b, 2), "seconds_to_adapt_and_requantise": round(t_c, 2),
            "int8_bytes": {"B_requantised": len(blob_b), "C_adapted_requantised": len(blob_c)},
            "input_range": {v: [round(a, 4) for a in input_range(nets[v][1])[:2]] for v in VARIANTS},
        })

        for r in rows:
            if r["patient_id"] in adapt_on:
                continue
            f = r["load"]()
            if len(f) == 0:
                continue
            x = model.normalise(f)
            p_float = {}                          # A and B share the original float model
            for v, (net, it) in nets.items():
                if id(net) not in p_float:
                    p_float[id(net)] = np.asarray(net(x, training=False)).reshape(len(x), -1)
                pf = p_float[id(net)]
                t0 = time.perf_counter()
                pq = tflite_predict(it, x)
                a = acc[v]
                a["sec"] += time.perf_counter() - t0
                a["out"] += clipped(x, it); a["vals"] += x.size
                a["win"] += len(x); a["same"] += int(np.sum(pf.argmax(1) == pq.argmax(1)))
                a["rf"].append(pf.mean(0)); a["rq"].append(pq.mean(0)); a["groups"].append(r["patient_id"])

    variants = {}
    for v in VARIANTS:
        a = acc[v]
        rf, rq = np.array(a["rf"]), np.array(a["rq"])
        same = rf.argmax(1) == rq.argmax(1)
        top2 = np.sort(rf, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]               # how far the float model's top class leads
        refer_f, refer_q = (1 - rf[:, 0]) > 0.5, (1 - rq[:, 0]) > 0.5
        by_margin = {}
        for lo, hi in MARGIN_BINS:
            m = (margin >= lo) & (margin < hi)
            by_margin[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {
                "recordings": int(m.sum()), "agreement": float(same[m].mean()) if m.any() else None}
        variants[v] = {
            "recording_agreement": float(same.mean()),
            "recording_agreement_interval": agreement_interval(rq.argmax(1), rf, np.array(a["groups"])),
            "window_agreement": a["same"] / a["win"],
            "refer_or_not_agreement": float(np.mean(refer_f == refer_q)),
            "agreement_by_float_margin": by_margin,
            "share_of_inputs_clipped": a["out"] / a["vals"],
            "ms_per_window": round(1000 * a["sec"] / a["win"], 3),
        }
    groups = acc["A_original"]["groups"]
    return {
        "measured_recordings": len(groups), "measured_patients": len(set(groups)),
        "measured_windows": acc["A_original"]["win"],
        "split": ("patients in two random halves (seed 0); N = %d recordings drawn from one half adapt "
                  "and calibrate, the other half is measured, then swap" % N_ADAPT
                  if site == "physionet_2022" else
                  "patients in two halves of 5 by patient number (as in S3); all 25 recordings of one "
                  "half adapt and calibrate, the other half is measured, then swap"),
        "halves": halves_info,
        "variants": variants,
        "target": "agreement >= 98% with the full-precision model (proposal 4.7, edge budget)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", required=True, choices=["physionet_2022", "clinic"])
    ap.add_argument("--data-dir", type=Path, help="the Real_World_Data folder, for --site clinic")
    args = ap.parse_args()
    if args.site == "clinic" and not args.data_dir:
        ap.error("--site clinic needs --data-dir")

    model = FinalModel()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["model"] = model.files
    out[args.site] = res = run_site(args.site, args.data_dir, model)
    res["run_on"] = dt.date.today().isoformat()
    if args.site == "clinic":
        res["data_rules"] = "aggregates only; re-quantised models existed in memory only"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"{args.site}: {res['measured_recordings']} measured recordings from {res['measured_patients']} patients")
    for v, r in res["variants"].items():
        ci = r["recording_agreement_interval"]
        print(f"  {v:22s} agreement {r['recording_agreement']:.2%} [{ci['lower_95']:.2%}, {ci['upper_95']:.2%}]"
              f"  refer-or-not {r['refer_or_not_agreement']:.2%}  clipped {r['share_of_inputs_clipped']:.3%}"
              f"  {r['ms_per_window']} ms/window")
        print("      agreement by float margin:", {k: (b["recordings"], None if b["agreement"] is None
                                                        else round(b["agreement"], 3))
                                                    for k, b in r["agreement_by_float_margin"].items()})
    for h in res["halves"]:
        print(f"  half: requantise {h['seconds_to_requantise']} s, adapt+requantise "
              f"{h['seconds_to_adapt_and_requantise']} s, sizes {h['int8_bytes']}")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
