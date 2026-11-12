"""Edge budget of the heart branch on an ordinary laptop (roadmap S4 and S5, proposal 4.5, 4.7).

Targets (proposal 4.7): each 8-bit model under 1 MB; at least 98% agreement with
the full-precision model; under 1 s per case on a laptop CPU.

One case is one recording, run through cardio.predict.analyse_heart exactly as the
dashboard runs it, so the time per case is the dashboard's own. Stages:

  load and resample -> quality check -> front-end (band-pass, windows, log-mel)
  -> 8-bit CNN on every window, then temperature -> the 17 secondary characteristics
  -> systole or diastole from the measured energy -> votes and late fusion

Recordings come from Yaseen fold 0 and a seeded sample of PhysioNet 2022. A case is
timed whole even when it fails the quality check (analyse_heart with complete=True).
Until heart_fusion and its parameter file exist, "votes and fusion" is the CNN vote
alone, and the report says so; run this again once they land.

The report also gives the share of every public recording that passes the quality
check, with a 95% interval over patients: the dashboard withholds the heart result
for the rest.

Peak memory is the whole process's peak working set, so it includes the full
TensorFlow runtime, which a deployed tool using only the 8-bit interpreter would not
need. Agreement comes from reports/heart_agreement.json and
reports/heart_requantisation.json.

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
from pathlib import Path

import numpy as np

from cardio.eval.intervals import statistic_interval
from cardio.model.final_model import MODEL_DIR, STEM, FinalModel
from cardio.predict import MIN_SAMPLES, STAGES, analyse_heart, fusion_function, temperature
from core.features import CARDIO_CONFIG, load_wav
from core.ood import OOD_HIBAND_THRESHOLD, signal_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"
MANIFESTS = REPO_ROOT / "data" / "manifests"
STAGE_NOTES = {
    "load_and_resample": "read the WAV, take channel 0, convert to 8 kHz",
    "quality_check": "at least 1.5 s of sound, and the energy test of core/ood.py",
    "front_end": "band-pass, 3-second windows with 50% overlap, log-mel",
    "cnn_8bit": "the shipped 8-bit model on every window, mean softmax, temperature",
    "secondary_characteristics": "the 17 numbers of cardio.model.features",
    "systole_or_diastole": "which phase holds the extra sound, from the measured energy",
    "votes_and_fusion": "heart_fusion, or the CNN vote alone until it exists",
}


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
    """Milliseconds per stage for one recording, through the dashboard's own code."""
    res = analyse_heart(path, model=model, complete=True)
    if res.withheld in ("unreadable", "too_short"):
        return None
    t = {s: res.timings_ms[s] for s in STAGES}
    t["total"] = res.total_ms
    t["seconds_of_audio"] = res.seconds_of_audio
    t["windows"] = res.windows
    t["passed_quality_check"] = res.withheld is None
    return t


def summarise(times: list[dict]) -> dict:
    out = {"recordings": len(times),
           "passed_quality_check": int(sum(t["passed_quality_check"] for t in times)),
           "median_seconds_of_audio": float(np.median([t["seconds_of_audio"] for t in times])),
           "median_windows": float(np.median([t["windows"] for t in times]))}
    for s in list(STAGES) + ["total"]:
        v = np.array([t[s] for t in times])
        out[f"{s}_ms"] = {"median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                          "max": round(float(v.max()), 2)}
    return out


def manifest_rows(name: str) -> list[dict]:
    path = MANIFESTS / f"{name}_manifest.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if (REPO_ROOT / r["path"]).exists()]


def public_recordings(n_physionet: int) -> dict[str, list[Path]]:
    sets = {"yaseen_fold0": [REPO_ROOT / r["path"] for r in manifest_rows("yaseen") if r["fold"] == "0"]}
    paths = [REPO_ROOT / r["path"] for r in manifest_rows("circor")]
    if paths:
        pick = np.random.default_rng(0).permutation(len(paths))[:n_physionet]
        sets["physionet_2022_sample"] = [paths[i] for i in sorted(pick)]
    return sets


def _share_of_ones(p: np.ndarray) -> float:
    """The bootstrap helper turns a 0/1 column into (1 - x, x); the share is column 1's mean."""
    return float(np.mean(p[:, 1]))


def quality_survey(name: str) -> dict | None:
    """Share of a public dataset's recordings that pass the quality check, by patient."""
    rows = manifest_rows(name)
    passes, groups, hi_band, short = [], [], [], 0
    for r in rows:
        x8 = load_wav(str(REPO_ROOT / r["path"]), CARDIO_CONFIG)
        if len(x8) < MIN_SAMPLES:
            short += 1
            continue
        q = signal_provenance((np.clip(x8, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        passes.append(float(bool(q["in_distribution"])))
        groups.append(r["patient_id"])
        hi_band.append(q["hi_band"])
    if not passes:
        return None
    hb = np.array(hi_band)
    return {
        "recordings": len(passes),
        "patients": len(set(groups)),
        "shorter_than_1_5_s": short,
        "share_passing": statistic_interval(np.array(passes), np.array(groups), _share_of_ones),
        "share_of_energy_above_1_khz": {"median": float(np.median(hb)), "p10": float(np.percentile(hb, 10)),
                                        "p90": float(np.percentile(hb, 90))},
        "threshold": OOD_HIBAND_THRESHOLD,
    }


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
    survey = {name: s for name in ("yaseen", "circor") if (s := quality_survey(name)) is not None}
    survey = {("physionet_2022" if k == "circor" else k): v for k, v in survey.items()}
    agree = agreement_summary()
    slowest = max(v["total_ms"]["p95"] for v in timing.values())
    fused = fusion_function() is not None
    out = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version()},
        "model": model.files,
        "sizes": sizes,
        "pipeline": "cardio.predict.analyse_heart, as the dashboard runs it; every case timed whole",
        "stages": STAGE_NOTES,
        "timing_per_case": timing,
        "votes_and_fusion": ("heart_fusion from cardio/fusion/late_fusion.py" if fused else
                             "CNN vote alone with weight 1: heart_fusion and reports/heart_fusion_params.json "
                             "are not in this branch yet. Run this again once they are."),
        "temperature": temperature(),
        "quality_check_on_public_data": survey,
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
        print(f"{name} ({s['recordings']} recordings, {s['passed_quality_check']} pass the quality check, "
              f"median {s['median_seconds_of_audio']:.1f} s of audio): "
              f"total median {s['total_ms']['median']} ms, p95 {s['total_ms']['p95']} ms")
        print(f"   median ms per stage: {stages}")
    for name, s in survey.items():
        share = s["share_passing"]
        print(f"quality check, all of {name}: {share['estimate']:.1%} pass "
              f"[{share['lower_95']:.1%}, {share['upper_95']:.1%}] of {s['recordings']} recordings")
    print("votes and fusion:", out["votes_and_fusion"])
    print(f"peak memory (whole process): {out['peak_memory_mb_whole_process']:.0f} MB")
    print("targets:", out["targets"])
    print("wrote reports/heart_edge_budget.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
