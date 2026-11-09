"""Benchmark inference latency and memory footprint for float and INT8 models.

Measures:
  * Mean inference time per image using time.perf_counter.
  * Peak memory allocation during inference using tracemalloc.
  * Model binary size on disk.

Usage:
  python -m lung.eval.timing
  python -m lung.eval.timing --runs 100 --warmup 10 --out reports/xray_timing.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import tracemalloc

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow import keras

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"
CACHE_DIR = REPO_ROOT / "data" / "cache"


def get_test_images(num_images: int = 100) -> tuple[np.ndarray, float, float]:
    """Retrieve test images and normalisation constants.

    Tries Kermany cache fold 0 first, falls back to Test_Data PNGs,
    and finally synthetic images matching Kermany statistics.
    """
    mu, sd = 0.57182133, 0.17511542
    qr_path = REPORT_DIR / "quantization_chiron_pulmo_v1.json"
    if qr_path.exists():
        try:
            with open(qr_path) as f:
                qr = json.load(f)
            mu, sd = float(qr["norm_mean"]), float(qr["norm_std"])
        except Exception:
            pass

    cache_file = CACHE_DIR / "kermany_logmel.npz"
    if cache_file.exists():
        cache = np.load(cache_file, allow_pickle=True)
        f0 = cache["fold"] == 0
        X_f0 = cache["X"][f0]
        if len(X_f0) > 0:
            take = min(num_images, len(X_f0))
            raw = X_f0[:take]
            norm = ((raw - mu) / sd)[..., None].astype(np.float32)
            return norm, mu, sd

    # Fallback to Test_Data/pulmo_images
    td_dir = REPO_ROOT / "Test_Data" / "pulmo_images"
    if td_dir.exists():
        from core.images import load_image
        pngs = sorted(td_dir.glob("*.png"))[:num_images]
        if pngs:
            raw = np.stack([load_image(str(p)) for p in pngs])
            norm = ((raw - mu) / sd)[..., None].astype(np.float32)
            return norm, mu, sd

    # Synthetic fallback
    rng = np.random.default_rng(0)
    synthetic_raw = rng.uniform(0.1, 0.9, (num_images, 96, 96)).astype(np.float32)
    norm = ((synthetic_raw - mu) / sd)[..., None].astype(np.float32)
    return norm, mu, sd


def benchmark_float_model(
    model_path: Path,
    images: np.ndarray,
    warmup: int = 10,
) -> dict:
    """Measure latency and peak memory of float Keras model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = keras.models.load_model(model_path)
    n_images = len(images)

    # Warmup
    for _ in range(warmup):
        _ = model(images[:1], training=False)

    # Latency & memory measurement
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()

    for i in range(n_images):
        _ = model(images[i : i + 1], training=False)

    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = t1 - t0
    mean_time_ms = (total_time / n_images) * 1000.0
    throughput = n_images / total_time
    file_bytes = model_path.stat().st_size

    return {
        "model": model_path.name,
        "format": "float32 Keras",
        "file_size_bytes": file_bytes,
        "file_size_kb": file_bytes / 1024,
        "file_size_mb": file_bytes / (1024 * 1024),
        "num_inferences": n_images,
        "total_time_s": total_time,
        "mean_time_ms": mean_time_ms,
        "mean_time_s": total_time / n_images,
        "throughput_fps": throughput,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_kb": peak_bytes / 1024,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
    }


def benchmark_int8_model(
    model_path: Path,
    images: np.ndarray,
    warmup: int = 10,
) -> dict:
    """Measure latency and peak memory of INT8 TFLite model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    # Pre-quantize inputs according to TFLite model input details
    q_inputs = np.clip(
        np.round(images / in_scale + in_zp), -128, 127
    ).astype(inp["dtype"])

    n_images = len(q_inputs)

    # Warmup
    for _ in range(warmup):
        interp.set_tensor(inp["index"], q_inputs[:1])
        interp.invoke()
        _ = interp.get_tensor(out["index"])

    # Latency & memory measurement
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()

    for i in range(n_images):
        interp.set_tensor(inp["index"], q_inputs[i : i + 1])
        interp.invoke()
        _ = interp.get_tensor(out["index"])

    t1 = time.perf_counter()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = t1 - t0
    mean_time_ms = (total_time / n_images) * 1000.0
    throughput = n_images / total_time
    file_bytes = model_path.stat().st_size

    return {
        "model": model_path.name,
        "format": "int8 TFLite",
        "file_size_bytes": file_bytes,
        "file_size_kb": file_bytes / 1024,
        "file_size_mb": file_bytes / (1024 * 1024),
        "num_inferences": n_images,
        "total_time_s": total_time,
        "mean_time_ms": mean_time_ms,
        "mean_time_s": total_time / n_images,
        "throughput_fps": throughput,
        "peak_memory_bytes": peak_bytes,
        "peak_memory_kb": peak_bytes / 1024,
        "peak_memory_mb": peak_bytes / (1024 * 1024),
    }


def run_timing_benchmarks(
    float_path: Path | str | None = None,
    int8_path: Path | str | None = None,
    num_images: int = 100,
    warmup: int = 10,
) -> dict:
    """Run full benchmark comparison between float and int8 models."""
    if float_path is None:
        float_path = MODEL_DIR / "chiron_pulmo_v1_float.keras"
    if int8_path is None:
        int8_path = MODEL_DIR / "chiron_pulmo_v1_int8.tflite"

    float_path = Path(float_path)
    int8_path = Path(int8_path)

    images, mu, sd = get_test_images(num_images=num_images)
    print(f"Benchmarking on {len(images)} images (shape {images[0].shape}, norm mu={mu:.4f}, sd={sd:.4f})...\n")

    print(f"--- Benchmarking Float Model: {float_path.name} ---")
    float_res = benchmark_float_model(float_path, images, warmup=warmup)
    print(f"Size:        {float_res['file_size_mb']:.2f} MB ({float_res['file_size_kb']:.1f} kB)")
    print(f"Latency:     {float_res['mean_time_ms']:.2f} ms/image ({float_res['throughput_fps']:.1f} img/s)")
    print(f"Peak Memory: {float_res['peak_memory_kb']:.1f} kB ({float_res['peak_memory_mb']:.3f} MB)\n")

    print(f"--- Benchmarking INT8 Model: {int8_path.name} ---")
    int8_res = benchmark_int8_model(int8_path, images, warmup=warmup)
    print(f"Size:        {int8_res['file_size_kb']:.1f} kB ({int8_res['file_size_mb']:.2f} MB)")
    print(f"Latency:     {int8_res['mean_time_ms']:.2f} ms/image ({int8_res['throughput_fps']:.1f} img/s)")
    print(f"Peak Memory: {int8_res['peak_memory_kb']:.1f} kB ({int8_res['peak_memory_mb']:.3f} MB)\n")

    speedup = float_res["mean_time_ms"] / max(int8_res["mean_time_ms"], 1e-6)
    size_reduction = float_res["file_size_bytes"] / max(int8_res["file_size_bytes"], 1)

    print("=== Comparison Summary ===")
    print(f"Model Size Reduction:  {size_reduction:.1f}x smaller ({float_res['file_size_kb']:.1f} kB -> {int8_res['file_size_kb']:.1f} kB)")
    print(f"Inference Speedup:     {speedup:.2f}x ({float_res['mean_time_ms']:.2f} ms -> {int8_res['mean_time_ms']:.2f} ms)")
    print(f"Meets Edge Budget:     Size < 1MB: {int8_res['file_size_mb'] < 1.0}, Time < 1s: {int8_res['mean_time_s'] < 1.0}")

    return {
        "float": float_res,
        "int8": int8_res,
        "size_reduction_ratio": size_reduction,
        "speedup_ratio": speedup,
        "meets_edge_budget": {
            "size_under_1mb": bool(int8_res["file_size_mb"] < 1.0),
            "time_under_1s": bool(int8_res["mean_time_s"] < 1.0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark chest X-ray models.")
    parser.add_argument("--float-model", type=str, default=str(MODEL_DIR / "chiron_pulmo_v1_float.keras"))
    parser.add_argument("--int8-model", type=str, default=str(MODEL_DIR / "chiron_pulmo_v1_int8.tflite"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    results = run_timing_benchmarks(
        float_path=args.float_model,
        int8_path=args.int8_model,
        num_images=args.runs,
        warmup=args.warmup,
    )

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2))
        print(f"\nWrote benchmark results to {out_p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

