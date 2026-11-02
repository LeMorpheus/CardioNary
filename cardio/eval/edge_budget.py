"""Edge budget of the heart branch on an ordinary laptop (roadmap S4, proposal 4.5, 4.7).

Targets (proposal 4.7): each 8-bit model under 1 MB; at least 98% agreement with
the full-precision model; under 1 s per case on a laptop CPU.

One case is one recording, timed stage by stage with the shipped 8-bit model on
public recordings (Yaseen fold 0, and a sample of PhysioNet 2022):

  load and resample -> quality check -> front-end (band-pass, windows, log-mel)
  -> 8-bit CNN on every window -> secondary characteristics
  -> systole/diastole attribution (occlusion on the 8-bit model)

Votes and late fusion (two logistic regressions, roadmap S2 for the fusion owner)
are not built yet and are listed as pending. Peak memory is the whole process's
peak working set, so it includes the full TensorFlow runtime, which a deployed
tool using only the 8-bit interpreter would not need. Agreement comes from
reports/heart_agreement.json and reports/heart_requantisation.json.

Usage:
    python -m cardio.eval.edge_budget

Writes:
    reports/heart_edge_budget.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from cardio.model.final_model import MODEL_DIR, STEM, FinalModel
from core.descriptors import audio_descriptors, audio_phase_attribution, segment_cycles
from core.features import CARDIO_CONFIG, extract, iter_windows, load_wav, preprocess
from core.ood import signal_provenance
from core.quantize import tflite_predict

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"
FS = CARDIO_CONFIG.sample_rate
STAGES = ["load_and_resample", "quality_check", "front_end", "cnn_8bit",
          "secondary_characteristics", "systole_diastole_attribution"]


def peak_memory_mb() -> float | None:
    """Peak working set of this process, in MB (Windows; Linux and macOS fall back)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2 ** 20
        return None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10
    except ImportError:
        return None


def time_case(path: Path, model: FinalModel) -> dict | None:
    """Seconds per stage for one recording, with the shipped 8-bit model."""
    t, clock = {}, time.perf_counter
    t0 = clock(); x8 = load_wav(str(path), CARDIO_CONFIG); t["load_and_resample"] = clock() - t0
    pcm = (np.clip(x8, -1.0, 1.0) * 32767.0).astype(np.int16)
    t0 = clock(); signal_provenance(pcm.tobytes()); t["quality_check"] = clock() - t0
    t0 = clock()
    feats = [extract(w, CARDIO_CONFIG) for w in iter_windows(preprocess(x8, CARDIO_CONFIG), CARDIO_CONFIG, 0.5)]
    t["front_end"] = clock() - t0
    if not feats:
        return None
    x = model.normalise(np.stack(feats))
    t0 = clock(); tflite_predict(model.interp, x); t["cnn_8bit"] = clock() - t0
    t0 = clock(); audio_descriptors(pcm, fs=FS); t["secondary_characteristics"] = clock() - t0
    t0 = clock()
    inp = model.interp.get_input_details()[0]
    scale, zp = inp["quantization"]
    q = np.clip(np.round(x[:1] / scale + zp), -128, 127).astype(inp["dtype"])
    audio_phase_attribution(model.interp, q, segment_cycles(pcm[: 3 * FS].astype(np.float64), FS), FS)
    t["systole_diastole_attribution"] = clock() - t0
    t["total"] = sum(t[s] for s in STAGES)
    t["seconds_of_audio"] = len(x8) / FS
    t["windows"] = len(feats)
    return t


def summarise(times: list[dict]) -> dict:
    out = {"recordings": len(times),
           "median_seconds_of_audio": float(np.median([t["seconds_of_audio"] for t in times])),
           "median_windows": float(np.median([t["windows"] for t in times]))}
    for s in STAGES + ["total"]:
        v = np.array([t[s] for t in times]) * 1000
        out[f"{s}_ms"] = {"median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                          "max": round(float(v.max()), 2)}
    return out


def public_recordings(n_physionet: int) -> dict[str, list[Path]]:
    with (REPO_ROOT / "data" / "manifests" / "yaseen_manifest.csv").open(encoding="utf-8") as fh:
        yaseen = [REPO_ROOT / r["path"] for r in csv.DictReader(fh) if r["fold"] == "0"]
    sets = {"yaseen_fold0": [p for p in yaseen if p.exists()]}
    circor = REPO_ROOT / "data" / "manifests" / "circor_manifest.csv"
    if circor.exists():
        with circor.open(encoding="utf-8") as fh:
            paths = [REPO_ROOT / r["path"] for r in csv.DictReader(fh)]
        paths = [p for p in paths if p.exists()]
        if paths:
            pick = np.random.default_rng(0).permutation(len(paths))[:n_physionet]
            sets["physionet_2022_sample"] = [paths[i] for i in sorted(pick)]
    return sets


def agreement_summary() -> dict:
    out = {}
    a = REPORTS / "heart_agreement.json"
    if a.exists():
        a = json.loads(a.read_text())
        if "yaseen_fold0" in a:
            out["yaseen_fold0_clip_agreement"] = a["yaseen_fold0"]["clip_agreement"]
        if "physionet_2022" in a:
            out["physionet_2022_recording_agreement_shipped_model"] = a["physionet_2022"]["recording_agreement"]
    r = REPORTS / "heart_requantisation.json"
    if r.exists():
        r = json.loads(r.read_text())
        for site in ("physionet_2022", "clinic"):
            if site in r:
                out[f"{site}_after_requantisation"] = {
                    v: r[site]["variants"][v]["recording_agreement"] for v in r[site]["variants"]}
                out[f"{site}_refer_or_not_agreement"] = {
                    v: r[site]["variants"][v]["refer_or_not_agreement"] for v in r[site]["variants"]}
                out[f"{site}_ms_per_window"] = {
                    v: r[site]["variants"][v]["ms_per_window"] for v in r[site]["variants"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--physionet-sample", type=int, default=100,
                    help="number of PhysioNet 2022 recordings to time (seed 0)")
    args = ap.parse_args()

    model = FinalModel()
    sizes = {"int8_bytes": (MODEL_DIR / f"{STEM}_int8.tflite").stat().st_size,
             "float_bytes": (MODEL_DIR / f"{STEM}_float.keras").stat().st_size}
    sets = public_recordings(args.physionet_sample)
    time_case(sets["yaseen_fold0"][0], model)             # warm-up, not counted
    timing = {}
    for name, paths in sets.items():
        times = [t for t in (time_case(p, model) for p in paths) if t is not None]
        timing[name] = summarise(times)
    agree = agreement_summary()
    slowest = max(v["total_ms"]["p95"] for v in timing.values())
    out = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version()},
        "model": model.files,
        "sizes": sizes,
        "timing_per_case": timing,
        "votes_and_fusion": "pending: not built yet (two logistic regressions on a few dozen numbers)",
        "agreement": agree,
        "peak_memory_mb_whole_process": peak_memory_mb(),
        "targets": {
            "int8_under_1_MB": sizes["int8_bytes"] < 1_000_000,
            "case_under_1_s_at_p95": slowest < 1000,
            "agreement_at_least_98_percent_on_yaseen_fold0": agree.get("yaseen_fold0_clip_agreement", 0) >= 0.98,
        },
        "notes": "Times are wall-clock on one laptop CPU, after one warm-up case. Peak memory "
                 "includes the TensorFlow runtime loaded by this script.",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "heart_edge_budget.json").write_text(json.dumps(out, indent=2))

    print(f"8-bit model {sizes['int8_bytes'] / 1024:.1f} kB, float {sizes['float_bytes'] / 1024:.1f} kB")
    for name, s in timing.items():
        stages = ", ".join(f"{k}: {s[k + '_ms']['median']}" for k in STAGES)
        print(f"{name} ({s['recordings']} recordings, median {s['median_seconds_of_audio']:.1f} s of audio): "
              f"total median {s['total_ms']['median']} ms, p95 {s['total_ms']['p95']} ms")
        print(f"   median ms per stage: {stages}")
    print(f"peak memory (whole process): {out['peak_memory_mb_whole_process']:.0f} MB")
    print("targets:", out["targets"])
    print("wrote reports/heart_edge_budget.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
