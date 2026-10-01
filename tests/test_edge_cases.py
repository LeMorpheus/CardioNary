"""Edge-case behaviour of the deployed INT8 models.

A screening device that answers confidently on silence is worse than one that
declines. These tests pin the behaviour we intend to ship, and they feed the
input-quality / OOD gate in docs/13_REAL_WORLD_VALIDATION.md section 4.

Run:
    python -m pytest tests/test_edge_cases.py -v -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"

CONFIDENCE_FLOOR = 0.60          # below this the firmware reports UNCERTAIN


def _load(node: str):
    """Return (interpreter, quant_report, config, modality) for a deployed node.

    Modality is read from the exported vector manifest rather than assumed, so
    these tests follow a node when it changes dataset or modality.
    """
    tf = pytest.importorskip("tensorflow")
    from core.features import CARDIO_CONFIG, PULMO_CONFIG
    from core.images import CXR_CONFIG

    path = MODEL_DIR / f"chiron_{node}_v1_int8.tflite"
    qr = REPORT_DIR / f"quantization_chiron_{node}_v1.json"
    if not path.exists() or not qr.exists():
        pytest.skip(f"{path.name} not built yet")

    vm = REPO_ROOT / "vectors" / f"chiron_{node}_vectors_manifest.json"
    modality = json.loads(vm.read_text())["modality"] if vm.exists() else "audio"

    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    q = json.loads(qr.read_text())
    if modality == "image":
        cfg = CXR_CONFIG
    else:
        cfg = CARDIO_CONFIG if node == "cardio" else PULMO_CONFIG
    return interp, q, cfg, modality


def _infer(interp, q, cfg, x: np.ndarray, modality: str = "audio") -> np.ndarray:
    """Raw input -> class probabilities, through the real front-end."""
    if modality == "image":
        feat = x.astype(np.float64)          # already (size, size) in [0,1]
    else:
        from core.features import preprocess, extract
        feat = extract(preprocess(x, cfg), cfg)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s, zp = inp["quantization"]
    qx = np.clip(np.round(((feat - q["norm_mean"]) / q["norm_std"]) / s + zp),
                 -128, 127).astype(inp["dtype"])
    interp.set_tensor(inp["index"], qx[None, ..., None])
    interp.invoke()
    os_, ozp = out["quantization"]
    raw = interp.get_tensor(out["index"]).reshape(-1).astype(np.float32)
    return (raw - ozp) * os_


IMAGE_EDGE_CASES = {
    "all_black": lambda n, r: np.zeros((n, n)),
    "all_white": lambda n, r: np.ones((n, n)),
    "mid_grey": lambda n, r: np.full((n, n), 0.5),
    "uniform_noise": lambda n, r: r.uniform(0, 1, (n, n)),
    "salt_pepper": lambda n, r: (r.random((n, n)) > 0.5).astype(float),
    "horizontal_gradient": lambda n, r: np.tile(np.linspace(0, 1, n), (n, 1)),
    "vertical_gradient": lambda n, r: np.tile(np.linspace(0, 1, n), (n, 1)).T,
    "checkerboard": lambda n, r: np.indices((n, n)).sum(0) % 2 * 1.0,
}

EDGE_CASES = {
    "digital_silence": lambda n, r: np.zeros(n),
    "dc_offset": lambda n, r: np.full(n, 0.5),
    "white_noise": lambda n, r: r.normal(0, 0.1, n),
    "full_scale_noise": lambda n, r: r.uniform(-1, 1, n),
    "clipped_square": lambda n, r: np.sign(np.sin(np.arange(n) * 0.05)),
    "pure_tone_1khz": lambda n, r: 0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / 4000),
    "impulse": lambda n, r: np.concatenate([[1.0], np.zeros(n - 1)]),
    "denormal_tiny": lambda n, r: np.full(n, 1e-12),
}


@pytest.mark.parametrize("node", ["cardio", "pulmo"])
def test_edge_cases_do_not_crash_and_are_reported(node):
    interp, q, cfg, modality = _load(node)
    rng = np.random.default_rng(0)
    if modality == "image":
        n, cases = cfg.size, IMAGE_EDGE_CASES
    else:
        n, cases = cfg.window_samples, EDGE_CASES

    print(f"\n--- {node} ({modality}) edge cases "
          f"(confidence floor {CONFIDENCE_FLOOR}) ---")
    rows = []
    for name, fn in cases.items():
        x = fn(n, rng)
        prob = _infer(interp, q, cfg, x, modality)

        assert prob.shape[0] >= 2, f"{name}: bad output shape {prob.shape}"
        assert np.all(np.isfinite(prob)), f"{name}: non-finite output {prob}"
        assert abs(prob.sum() - 1.0) < 0.05, f"{name}: probs sum to {prob.sum():.3f}"

        conf = float(prob.max())
        rows.append((name, conf, int(prob.argmax())))
        verdict = "UNCERTAIN" if conf < CONFIDENCE_FLOOR else f"class {prob.argmax()}"
        print(f"  {name:18s} max_conf {conf:.3f}  -> {verdict}")

    # Report, do not fail: a confident answer on silence is a finding to fix in
    # the OOD gate, not a reason to block the pipeline.
    confident = [r for r in rows if r[1] >= CONFIDENCE_FLOOR]
    if confident:
        print(f"\n  NOTE: {len(confident)}/{len(rows)} degenerate inputs produced a "
              f"confident answer. The confidence floor alone is NOT sufficient; "
              f"the SNR/OOD gate (docs/13 section 4) is required.")


@pytest.mark.parametrize("node", ["cardio", "pulmo"])
def test_short_and_long_inputs_are_handled(node):
    interp, q, cfg, modality = _load(node)
    if modality == "image":
        pytest.skip("image nodes take a fixed-size input; no windowing to test")
    rng = np.random.default_rng(1)
    from core.features import preprocess, iter_windows

    # shorter than one window -> zero windows or one padded window, never a crash
    for frac in (0.1, 0.49, 0.51, 1.0, 2.7):
        x = preprocess(rng.normal(0, 0.05, int(cfg.window_samples * frac)), cfg)
        wins = list(iter_windows(x, cfg))
        assert all(len(w) == cfg.window_samples for w in wins), \
            f"{node}: window length wrong at frac {frac}"
        if frac < 0.5:
            assert len(wins) == 0, f"{node}: kept a <50% tail at frac {frac}"


@pytest.mark.parametrize("node", ["cardio", "pulmo"])
def test_amplitude_invariance(node):
    """RMS normalisation must make the model insensitive to absolute gain.

    If this fails, the model keys on recording level -- a device artefact.
    """
    interp, q, cfg, modality = _load(node)
    rng = np.random.default_rng(2)
    if modality == "image":
        # Images are not RMS-normalised; brightness IS diagnostic-ish signal.
        # The equivalent invariance to check is small brightness offsets.
        base = np.clip(rng.normal(0.5, 0.1, (cfg.size, cfg.size)), 0, 1)
        probs = [_infer(interp, q, cfg, np.clip(base + d, 0, 1), modality)
                 for d in (-0.02, 0.0, 0.02)]
    else:
        base = rng.normal(0, 0.05, cfg.window_samples)
        probs = [_infer(interp, q, cfg, base * g) for g in (0.1, 1.0, 8.0)]
    for p in probs[1:]:
        assert p.argmax() == probs[0].argmax(), (
            f"{node}: prediction changed with gain -- "
            f"{[float(x.max()) for x in probs]}"
        )
