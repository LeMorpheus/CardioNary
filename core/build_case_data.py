"""Build gateway/case_data.json -- per-case detail for the clinical dashboard.

Two kinds of information, deliberately kept apart in the output because they
have very different epistemic status:

  measured  -- computed from the ACTUAL signal stored on the device (heart rate,
               rhythm regularity, signal quality). These are real measurements of
               this specific recording.

  model     -- what the classifier produced (class, confidence, distribution).

Anything else the dashboard shows (typical clinical associations, referral
urgency) is generic reference material keyed by class, held in
gateway/clinical.py -- NOT derived from the signal, and labelled as such in the
UI. Blurring those three together is how a screening demo starts implying things
it cannot support.

Parses the vector .c files rather than the source WAVs so the numbers describe
exactly the bytes that are flashed on the board.

    python core/build_case_data.py
"""
import json
import os
import re
import sys

import numpy as np
from scipy.signal import butter, sosfilt, find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.descriptors import (audio_descriptors, audio_phase_attribution,  # noqa: E402
                            segment_cycles, vision_descriptors,
                            vision_zone_attribution)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_CARDIO = 8000


def parse_int16_vectors(path):
    text = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"const int16_t chiron_vec_(\d+)\[\d+\]\s*=\s*\{(.*?)\};", text, re.S):
        out[int(m.group(1))] = np.array(re.findall(r"-?\d+", m.group(2)), dtype=np.int64)
    return out


def parse_uint8_vectors(path):
    text = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"const uint8_t chiron_vec_(\d+)\[\d+\]\s*=\s*\{(.*?)\};", text, re.S):
        out[int(m.group(1))] = np.array(re.findall(r"-?\d+", m.group(2)), dtype=np.uint8)
    return out


def envelope(x, fs):
    """Shannon-energy envelope, the standard front-end for heart-sound timing."""
    x = x / (np.max(np.abs(x)) + 1e-9)
    se = -(x ** 2) * np.log(x ** 2 + 1e-12)
    sos = butter(3, 20.0, btype="low", fs=fs, output="sos")
    e = sosfilt(sos, se)
    e -= e.min()
    return e / (e.max() + 1e-9)


def heart_rate(x, fs):
    """Autocorrelation of the envelope over a physiological 40-180 bpm window.

    A cardiac cycle has TWO loud events (S1 and S2), so the envelope's strongest
    periodicity can be the S1->S2 gap rather than the beat-to-beat period,
    reporting roughly double the true rate.

    An explicit octave correction (prefer 2x lag when its autocorrelation is
    comparable) was tried and made this WORSE -- it doubled periods that were
    already correct, pushing 5 of 53 cases outside 45-120 bpm where previously
    only 1 was. Constraining the search to a resting-adult window is both simpler
    and empirically better here: 52 of 53 then land in 60-94 bpm.

    The cap is a stated assumption, not a measurement: a genuine tachycardia
    above 140 bpm would be clipped. Acceptable for this corpus (resting clinical
    recordings); it would NOT be acceptable for live monitoring.
    """
    e = envelope(x, fs)
    e = e - np.mean(e)
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lo, hi = int(fs * 60 / 140), int(fs * 60 / 45)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] <= 0:
        return None
    return round(60.0 * fs / lag)


def rhythm_regularity(x, fs):
    """Percent regularity of beat-to-beat intervals (100 = perfectly even)."""
    e = envelope(x, fs)
    peaks, _ = find_peaks(e, height=0.25, distance=int(0.20 * fs))
    if len(peaks) < 4:
        return None, len(peaks)
    iv = np.diff(peaks) / fs
    # S1/S2 alternate, so take every other interval to get beat-to-beat spacing
    beat = iv[::2] if len(iv) >= 4 else iv
    if len(beat) < 2 or np.mean(beat) <= 0:
        return None, len(peaks)
    cv = float(np.std(beat) / np.mean(beat))
    return round(max(0.0, min(100.0, 100.0 * (1.0 - cv))), 1), len(peaks)


def signal_quality_db(x, fs):
    """Envelope peak-to-baseline ratio -- a proxy for how clean the capture is."""
    e = envelope(x, fs)
    hi = float(np.percentile(e, 95))
    lo = float(np.percentile(e, 20)) + 1e-6
    return round(20.0 * np.log10(hi / lo), 1)


def _quantise(f, scale, zp):
    return np.clip(np.round(f / scale) + zp, -128, 127).astype(np.int8)


def validate(cases, cm, pm):
    """Do the descriptors behave the way physiology says they should?

    This is the check that decides whether these numbers are worth showing at
    all. If a systolic-murmur measure does not light up on the systolic-murmur
    classes, it is measuring noise and should be withdrawn, not displayed.
    """
    print("\n=== validation: murmur phase vs true class ===")
    print("  expectation: AS/MR/MVP systolic, MS diastolic, Normal none")
    by_class = {}
    for v in cm["vectors"]:
        d = cases["1"][str(v["vector_id"])].get("descriptors") or {}
        by_class.setdefault(v["expected_class"], []).append(d.get("murmur_phase"))
    for cls in ["Normal", "AS", "MS", "MR", "MVP"]:
        ph = [p for p in by_class.get(cls, []) if p]
        if not ph:
            continue
        counts = {p: ph.count(p) for p in set(ph)}
        print(f"  {cls:<7} n={len(ph):<3} {counts}")

    print("\n=== validation: phase energy ratios ===")
    print("  expectation: AS/MR/MVP raise the systolic ratio, MS the diastolic")
    for cls in ["Normal", "AS", "MS", "MR", "MVP"]:
        sr, dr = [], []
        for v in cm["vectors"]:
            if v["expected_class"] != cls:
                continue
            d = cases["1"][str(v["vector_id"])].get("descriptors") or {}
            if d.get("systolic_ratio") is not None:
                sr.append(d["systolic_ratio"])
                dr.append(d["diastolic_ratio"])
        if sr:
            print(f"  {cls:<7} systolic={np.mean(sr):.3f}+/-{np.std(sr):.3f}   "
                  f"diastolic={np.mean(dr):.3f}+/-{np.std(dr):.3f}")

    print("\n=== validation: image statistics vs true class ===")
    print("  expectation (tested, NOT assumed): pneumonia differs from normal")
    for lbl, name in [(0, "Normal"), (1, "Pneumonia")]:
        b = [cases["2"][str(v["vector_id"])]["descriptors"]["dense_area_pct"]
             for v in pm["vectors"] if v["expected_label"] == lbl]
        t = [cases["2"][str(v["vector_id"])]["descriptors"]["texture_heterogeneity"]
             for v in pm["vectors"] if v["expected_label"] == lbl]
        if b:
            print(f"  {name:<10} dense_area={np.mean(b):5.1f}+/-{np.std(b):4.1f}%   "
                  f"texture={np.mean(t):5.1f}+/-{np.std(t):4.1f}")

    print("\n=== validation: where the vision model looks (occlusion) ===")
    from collections import Counter
    for lbl, name in [(0, "Normal"), (1, "Pneumonia")]:
        rows = [cases["2"][str(v["vector_id"])]["attribution"]["peak_region"].split(" / ")[0]
                for v in pm["vectors"] if v["expected_label"] == lbl
                and (cases["2"][str(v["vector_id"])].get("attribution") or {}).get("attribution_ok")]
        print(f"  {name:<10} {dict(Counter(rows))}")

    print("\n=== validation: what drives the cardiac model ===")
    drv = {}
    for v in cm["vectors"]:
        a = cases["1"][str(v["vector_id"])].get("attribution") or {}
        if a.get("attribution_ok"):
            drv.setdefault(v["expected_class"], []).append(a.get("driven_by"))
    for cls, vals in drv.items():
        counts = {p: vals.count(p) for p in set(vals) if p}
        print(f"  {cls:<7} {counts}")


def main():
    import tensorflow as tf
    from core.features import CARDIO_CONFIG, extract, preprocess

    cases = {"1": {}, "2": {}}

    # ---- cardio: real measurements from the flashed audio ------------------
    # The DEPLOYED int8 models, not the float twins: attribution should describe
    # what actually ships.
    c_interp = tf.lite.Interpreter(
        model_path=os.path.join(REPO, "models", "chiron_cardio_v1_int8.tflite"))
    c_interp.allocate_tensors()
    c_in = c_interp.get_input_details()[0]
    c_scale, c_zp = c_in["quantization"]

    p_interp = tf.lite.Interpreter(
        model_path=os.path.join(REPO, "models", "chiron_pulmo_v1_int8.tflite"))
    p_interp.allocate_tensors()
    p_in = p_interp.get_input_details()[0]

    cm = json.load(open(os.path.join(REPO, "vectors",
                                     "chiron_cardio_vectors_manifest.json")))
    arrays = parse_int16_vectors(os.path.join(
        REPO, "firmware", "chiron_cardio", "vectors", "chiron_cardio_vectors.c"))
    print(f"cardio: {len(cm['vectors'])} manifest entries, {len(arrays)} audio arrays")

    for v in cm["vectors"]:
        vid = v["vector_id"]
        rec = {
            "vector_id": vid,
            "source_file": v["source_file"],
            "patient_id": v.get("patient_id", ""),
            "expected_label": v["expected_label"],
            "expected_class": v["expected_class"],
            "offline_pred": v["offline_int8_pred"],
            "offline_confidence": round(float(v["confidence"]), 4),
            "probs": v.get("probs", []),
            "measured": {},
        }
        x = arrays.get(vid)
        if x is not None:
            xf = x.astype(np.float64)
            hr = heart_rate(xf, FS_CARDIO)
            reg, npk = rhythm_regularity(xf, FS_CARDIO)
            rec["measured"] = {
                "heart_rate_bpm": hr,
                "rhythm_regularity_pct": reg,
                "sound_events_detected": npk,
                "signal_quality_db": signal_quality_db(xf, FS_CARDIO),
                "duration_s": round(len(x) / FS_CARDIO, 1),
                "sample_rate_hz": FS_CARDIO,
            }
            rec["descriptors"] = audio_descriptors(x, FS_CARDIO)

            # Probe the DEPLOYED int8 model: which phase of the cycle is the
            # existing prediction actually leaning on? No gradients, no retraining.
            feats = extract(preprocess(xf / 32768.0, CARDIO_CONFIG), CARDIO_CONFIG)
            q = _quantise(feats, c_scale, c_zp).reshape(c_in["shape"])
            rec["attribution"] = audio_phase_attribution(
                c_interp, q, segment_cycles(xf, FS_CARDIO), FS_CARDIO,
                n_frames=c_in["shape"][2])
        cases["1"][str(vid)] = rec

    # ---- pulmo: an image classifier measures no physiology -----------------
    pm = json.load(open(os.path.join(REPO, "vectors",
                                     "chiron_pulmo_vectors_manifest.json")))
    pix = parse_uint8_vectors(os.path.join(
        REPO, "firmware", "chiron_pulmo", "vectors", "chiron_pulmo_vectors.c"))
    print(f"pulmo : {len(pm['vectors'])} manifest entries, {len(pix)} images")

    for v in pm["vectors"]:
        vid = v["vector_id"]
        rec = {
            "vector_id": vid,
            "source_file": v["source_file"],
            "patient_id": v.get("patient_id", ""),
            "expected_label": v["expected_label"],
            "expected_class": v["expected_class"],
            "offline_pred": v["offline_int8_pred"],
            "offline_confidence": round(float(v["confidence"]), 4),
            "probs": v.get("probs", []),
            # No vital signs here, ever: a chest X-ray classifier yields a class,
            # not physiology. Inventing numbers would be the dishonest part.
            "measured": {"image_size_px": "96 x 96 grayscale"},
        }
        img = pix.get(vid)
        if img is not None:
            im = img.reshape(96, 96)
            rec["descriptors"] = vision_descriptors(im)
            q = (im.astype(np.int16) - 128).astype(np.int8).reshape(p_in["shape"])
            rec["attribution"] = vision_zone_attribution(p_interp, q, grid=3)
        cases["2"][str(vid)] = rec

    meta = {
        "cardio_classes": cm["classes"],
        "pulmo_classes": pm["classes"],
        "cardio_cv_accuracy": "98.9% +/- 1.1%",
        "pulmo_cv_accuracy": "96.37% image / 97.85% patient",
        "cardio_dataset": "Yaseen 2018 heart sounds",
        "pulmo_dataset": "Kermany paediatric chest X-ray",
    }

    validate(cases, cm, pm)

    out = os.path.join(REPO, "gateway", "case_data.json")
    json.dump({"meta": meta, "cases": cases}, open(out, "w"), indent=1)
    size = os.path.getsize(out)
    print(f"wrote {out} ({size/1024:.0f} kB)")

    hrs = [c["measured"].get("heart_rate_bpm") for c in cases["1"].values()]
    hrs = [h for h in hrs if h]
    print(f"heart rate: n={len(hrs)} min={min(hrs)} max={max(hrs)} "
          f"mean={sum(hrs)/len(hrs):.0f} bpm")


if __name__ == "__main__":
    main()
