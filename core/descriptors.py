"""Derived descriptors from the EXISTING models and EXISTING inputs.

Design constraint driving every choice here: no new network, no retraining, no
new prediction head. Everything is either (a) classical signal/image processing
on the same bytes the device already holds, or (b) probing the already-trained
INT8 model that is already deployed.

Two honesty rules are enforced throughout:

  1. Every descriptor is tagged with a TIER (see TIERS below). Anything that
     cannot be defended from this data is not computed at all.
  2. Measured quantities never become diagnoses. "Systolic murmur energy is
     high" is a measurement; "this patient has aortic stenosis" is the model's
     job, and the model is the anchor. Descriptors annotate it, never override it.

Where the computation runs: offline, in this file, over the exact vectors that
are flashed on the device. That costs the firmware zero flash and zero RAM.
Each function notes whether it would also be feasible on-device.
"""
import numpy as np
from scipy.signal import butter, find_peaks, sosfilt, welch
from scipy.ndimage import label as cc_label

TIERS = {
    1: "directly extractable from the existing signal or model output",
    2: "simple signal/image processing",
    3: "weak heuristic - indicative only",
    4: "not defensibly inferable (not computed)",
}


# ===========================================================================
# Audio: phonocardiogram descriptors
# ===========================================================================
def _envelope(x, fs):
    """Shannon-energy envelope: the standard front-end for heart-sound timing."""
    x = x / (np.max(np.abs(x)) + 1e-9)
    se = -(x ** 2) * np.log(x ** 2 + 1e-12)
    sos = butter(3, 20.0, btype="low", fs=fs, output="sos")
    e = sosfilt(sos, se)
    e -= e.min()
    return e / (e.max() + 1e-9)


def segment_cycles(x, fs):
    """Locate S1/S2 events and label the gaps as systole or diastole.

    Physiological basis: within one cardiac cycle the S1->S2 gap (systole) is
    SHORTER than the S2->S1 gap (diastole), at normal rates. So the alternating
    sequence of inter-peak gaps splits into two interleaved groups and the group
    with the smaller mean is systole. This needs no learning and no labels.

    Tier 2. On-device feasible: peak-picking on an envelope is a few hundred
    operations per second of audio.
    """
    e = _envelope(x, fs)
    peaks, _ = find_peaks(e, height=0.20, distance=int(0.16 * fs))
    if len(peaks) < 4:
        return None

    gaps = np.diff(peaks)
    even, odd = gaps[0::2], gaps[1::2]
    if len(even) == 0 or len(odd) == 0:
        return None

    # The interleaved group with the smaller mean gap is systole.
    systole_first = np.mean(even) < np.mean(odd)
    windows = []
    for i, g in enumerate(gaps):
        is_sys = (i % 2 == 0) if systole_first else (i % 2 == 1)
        windows.append({"start": int(peaks[i]), "end": int(peaks[i + 1]),
                        "phase": "systole" if is_sys else "diastole"})

    sys_ms = float(np.mean([w["end"] - w["start"] for w in windows
                            if w["phase"] == "systole"]) / fs * 1000)
    dia_ms = float(np.mean([w["end"] - w["start"] for w in windows
                            if w["phase"] == "diastole"]) / fs * 1000)
    return {"peaks": peaks, "windows": windows, "envelope": e,
            "systole_ms": round(sys_ms), "diastole_ms": round(dia_ms)}


def _band_powers(x, fs):
    """Fraction of total power in bands that matter for murmurs. Tier 2."""
    f, p = welch(x, fs=fs, nperseg=min(512, len(x)))
    bands = {"20_100": (20, 100), "100_250": (100, 250),
             "250_500": (250, 500), "500_800": (500, 800)}
    tot = float(np.sum(p[(f >= 20) & (f <= 800)])) + 1e-12
    out = {}
    for k, (lo, hi) in bands.items():
        out[k] = round(100.0 * float(np.sum(p[(f >= lo) & (f < hi)])) / tot, 1)
    centroid = float(np.sum(f[(f >= 20) & (f <= 800)] * p[(f >= 20) & (f <= 800)])
                     / tot)
    return out, round(centroid)


def audio_descriptors(x_int16, fs=8000):
    """Descriptive acoustic phenotype. Never a diagnosis.

    Returns None-valued fields rather than guesses when segmentation fails --
    a missing number is honest, a fabricated one is not.
    """
    x = x_int16.astype(np.float64)
    seg = segment_cycles(x, fs)
    bands, centroid = _band_powers(x, fs)

    out = {
        "band_power_pct": bands,          # tier 2
        "spectral_centroid_hz": centroid,  # tier 2
        "murmur_detected": None,
        "murmur_phase": None,
        "murmur_duration_pct": None,
        "murmur_intensity_db": None,
        "systole_ms": None,
        "diastole_ms": None,
        "s1_s2_ratio_db": None,
        "click_suspected": None,
        "segmentation_ok": bool(seg is not None),
    }
    if seg is None:
        return out

    out["systole_ms"] = seg["systole_ms"]
    out["diastole_ms"] = seg["diastole_ms"]

    # Energy in the INTERIOR of each gap (excluding the S1/S2 transients that
    # bound it) versus the energy of the bounding sounds themselves. A murmur is
    # exactly "sustained energy where there should be relative silence".
    #
    # Measured on the BAND-PASSED WAVEFORM, not the Shannon envelope. The
    # envelope is tuned to emphasise sharp transients (S1/S2), which is right for
    # timing and wrong for murmurs: it suppressed the low-amplitude diastolic
    # rumble of mitral stenosis so completely that 9/9 MS cases were reported as
    # having no murmur at all. RMS on a 25-600 Hz band recovers it.
    sos_bp = butter(3, [25.0, 600.0], btype="band", fs=fs, output="sos")
    xb = sosfilt(sos_bp, x / (np.max(np.abs(x)) + 1e-9))

    e = seg["envelope"]
    # Reference level: RMS in a short window centred on each detected heart sound.
    half = int(0.03 * fs)
    ref = [float(np.sqrt(np.mean(xb[max(p - half, 0):p + half] ** 2)))
           for p in seg["peaks"]]
    peak_rms = float(np.mean(ref)) + 1e-9
    peak_e = float(np.mean(e[seg["peaks"]])) + 1e-9

    phase_energy, phase_dur = {"systole": [], "diastole": []}, {"systole": [], "diastole": []}
    for w in seg["windows"]:
        n = w["end"] - w["start"]
        if n < int(0.06 * fs):
            continue
        m = int(0.25 * n)                      # drop 25% at each end (S1/S2 skirts)
        seg_x = xb[w["start"] + m: w["end"] - m]
        if seg_x.size == 0:
            continue
        rms = float(np.sqrt(np.mean(seg_x ** 2)))
        phase_energy[w["phase"]].append(rms / peak_rms)
        # fraction of the interior whose local RMS exceeds 20% of the reference
        step = max(int(0.01 * fs), 1)
        loc = np.array([np.sqrt(np.mean(seg_x[i:i + step] ** 2))
                        for i in range(0, max(len(seg_x) - step, 1), step)])
        phase_dur[w["phase"]].append(float(np.mean(loc > 0.20 * peak_rms))
                                     if loc.size else 0.0)

    sys_r = float(np.mean(phase_energy["systole"])) if phase_energy["systole"] else 0.0
    dia_r = float(np.mean(phase_energy["diastole"])) if phase_energy["diastole"] else 0.0

    # Calibrated against the observed held-out distribution, not guessed:
    #
    #            systolic          diastolic
    #   Normal   0.001 +/- 0.001   0.006 +/- 0.017
    #   AS       0.859 +/- 0.131   0.078 +/- 0.161
    #   MS       0.058 +/- 0.043   0.124 +/- 0.046   <-- diastolic predominance
    #   MR       0.486 +/- 0.170   0.019 +/- 0.021
    #   MVP      0.320 +/- 0.252   0.080 +/- 0.120
    #
    # The first cut tried was 0.18, which sat ABOVE the mitral-stenosis diastolic
    # ratio and so reported "no murmur" for 9/9 MS cases -- the one class whose
    # defining sign is a diastolic murmur. 0.06 is ~3 SD above the Normal
    # diastolic mean and comfortably below the MS mean, separating "quiet gap"
    # from "filled gap" across all five classes.
    #
    # This is a descriptive cut-off, not a clinical grading.
    MURMUR_R = 0.06
    dominant = "systole" if sys_r >= dia_r else "diastole"
    ratio = max(sys_r, dia_r)
    out["systolic_ratio"] = round(sys_r, 4)
    out["diastolic_ratio"] = round(dia_r, 4)
    out["murmur_detected"] = bool(ratio >= MURMUR_R)
    out["murmur_phase"] = dominant if out["murmur_detected"] else "none"
    out["murmur_intensity_db"] = round(20.0 * np.log10(ratio + 1e-6), 1)
    dur = phase_dur[dominant]
    out["murmur_duration_pct"] = round(100.0 * float(np.mean(dur)), 1) if dur else None

    # S1 vs S2 amplitude. Peaks alternate; the group that opens systole is S1.
    ev = e[seg["peaks"]]
    if len(ev) >= 4:
        a, b = float(np.mean(ev[0::2])), float(np.mean(ev[1::2]))
        out["s1_s2_ratio_db"] = round(20.0 * np.log10((a + 1e-9) / (b + 1e-9)), 1)

    # Mid-systolic click (MVP): a SHARP, brief transient near mid-systole. Tier 3
    # -- reported as "suspected" only, and only when segmentation succeeded.
    clicks = 0
    for w in seg["windows"]:
        if w["phase"] != "systole":
            continue
        n = w["end"] - w["start"]
        if n < int(0.10 * fs):
            continue
        mid = e[w["start"] + int(0.30 * n): w["end"] - int(0.25 * n)]
        if mid.size < 8:
            continue
        base = float(np.median(mid)) + 1e-9
        pk, _ = find_peaks(mid, height=max(4.0 * base, 0.15 * peak_e),
                           width=(None, int(0.05 * fs)))
        if len(pk):
            clicks += 1
    n_sys = sum(1 for w in seg["windows"] if w["phase"] == "systole")
    out["click_suspected"] = bool(n_sys and clicks / n_sys >= 0.5)
    return out


# ===========================================================================
# Vision: chest radiograph descriptors
# ===========================================================================
def vision_descriptors(img):
    """Measurable image properties of a 96x96 grayscale chest radiograph.

    IMPORTANT RESOLUTION CAVEAT: the stored image is 96x96 and centre-cropped.
    That is enough for coarse burden and distribution, and NOT enough for
    costophrenic angles, cardiac borders or pleural lines. Anything requiring
    those (effusion, cardiomegaly, pneumothorax) is tier 4 and deliberately not
    computed -- see docs/18.

    Radiographic convention: denser tissue is BRIGHTER. Standard frontal films
    place the patient's RIGHT lung on the viewer's LEFT; sides are reported as
    patient sides under that assumption, and the assumption is stated.
    """
    im = img.astype(np.float64)
    if im.max() > 1.5:
        im = im / 255.0
    h, w = im.shape

    # Crude thoracic ROI: trim the border, then exclude the central mediastinal
    # column where heart and spine dominate and "opacity" is not lung opacity.
    b = int(0.08 * h)
    roi = np.zeros_like(im, dtype=bool)
    roi[b:h - b, b:w - b] = True
    mid_lo, mid_hi = int(0.44 * w), int(0.56 * w)
    roi[:, mid_lo:mid_hi] = False

    vals = im[roi]
    # ABSOLUTE threshold, deliberately not a percentile.
    #
    # This started as `np.percentile(vals, 70)`, which is circular: taking the
    # top 30% of pixels guarantees a "burden" of ~30% for every image, whatever
    # it contains. Measured over the held-out set it produced Normal 29.5% vs
    # Pneumonia 29.3% -- the statistic was reporting its own threshold.
    #
    # A fixed cut on the shared intensity scale at least measures the image.
    # Images are not per-image normalised (centre-crop + resize only), so levels
    # are comparable across films.
    thr = 0.72
    mask = (im > thr) & roi
    burden = 100.0 * float(mask.sum()) / float(roi.sum())

    # Texture heterogeneity: aerated lung is high-contrast (vessels against air);
    # consolidation fills that in and flattens it. This is the ONLY simple image
    # statistic that showed any separation on the held-out set -- Normal 43.8+/-6.8
    # vs Pneumonia 37.8+/-9.0, roughly 0.75 SD. Real, but far too overlapping to
    # decide anything on its own. Tier 3, and labelled as such.
    heterogeneity = float(np.std(im[roi] * 255.0))

    # Laterality (viewer halves -> patient sides)
    left_view = mask[:, :mid_lo] & roi[:, :mid_lo]
    right_view = mask[:, mid_hi:] & roi[:, mid_hi:]
    lv = float(left_view.sum()) / max(roi[:, :mid_lo].sum(), 1)
    rv = float(right_view.sum()) / max(roi[:, mid_hi:].sum(), 1)
    asym = abs(lv - rv) / (lv + rv + 1e-9)
    if asym < 0.20:
        side = "bilateral"
    else:
        # viewer-left is the patient's right
        side = "patient right" if lv > rv else "patient left"

    # Vertical zones
    zb = [(b, int(h / 3)), (int(h / 3), int(2 * h / 3)), (int(2 * h / 3), h - b)]
    zone_names = ["upper", "middle", "lower"]
    zvals = []
    for (z0, z1) in zb:
        zm = mask[z0:z1, :] & roi[z0:z1, :]
        zvals.append(float(zm.sum()) / max(roi[z0:z1, :].sum(), 1))
    zsum = sum(zvals) + 1e-9
    zone_pct = {n: round(100.0 * v / zsum, 1) for n, v in zip(zone_names, zvals)}
    predominant_zone = zone_names[int(np.argmax(zvals))]

    # Focal vs diffuse from connected components
    lab, n = cc_label(mask)
    if n == 0:
        pattern, largest_frac = "none", 0.0
    else:
        sizes = np.bincount(lab.ravel())[1:]
        largest_frac = float(sizes.max()) / float(sizes.sum())
        pattern = "focal" if (largest_frac > 0.55 and burden < 30) else "diffuse"

    return {
        # tier 2 -- genuine measurements of the image
        "dense_area_pct": round(burden, 1),
        "distribution": side,
        "asymmetry_index": round(asym, 3),
        "zone_distribution_pct": zone_pct,
        "predominant_zone": predominant_zone,
        "largest_region_fraction": round(largest_frac, 3),
        "regions_detected": int(n),
        "mean_intensity": round(float(vals.mean()) * 255.0, 1),
        # tier 3 -- weakly separating, never sufficient alone
        "texture_heterogeneity": round(heterogeneity, 1),
        "pattern": pattern,
        # honesty rails
        "discriminates_pneumonia": False,
        "note": "These describe the image, they do NOT detect pneumonia. Measured "
                "over the held-out set, simple intensity statistics do not "
                "separate the classes (mean 138.8 vs 140.8 of 255). The "
                "classification comes from the CNN; these numbers only describe "
                "what is visible.",
        "resolution_note": "96x96 centre-cropped: costophrenic angles, cardiac "
                           "borders and pleural lines are not reliably present, "
                           "so effusion, cardiomegaly and pneumothorax are NOT "
                           "assessable (tier 4).",
    }


# ===========================================================================
# Model attribution by occlusion (no gradients, no retraining)
# ===========================================================================
def audio_phase_attribution(interp, feat_int8, seg, fs, n_frames=192):
    """Which part of the cardiac cycle actually drives the existing prediction?

    Occlusion sensitivity on the DEPLOYED INT8 model: blank a slice of the input
    spectrogram, re-run, and measure how far the winning probability falls. No
    gradients, so it works on the quantised TFLite graph exactly as shipped.

    Returns the confidence drop attributable to systolic vs diastolic frames.
    Tier 1 -- it is a direct property of the existing model.
    """
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    zp_in = inp["quantization"][1]
    o_scale, o_zp = out["quantization"]

    def run(a):
        interp.set_tensor(inp["index"], a.astype(inp["dtype"]))
        interp.invoke()
        p = (interp.get_tensor(out["index"]).reshape(-1).astype(np.float32) - o_zp) * o_scale
        return p

    base = run(feat_int8)
    cls = int(np.argmax(base))
    base_p = float(base[cls])
    if seg is None:
        return {"attribution_ok": False}

    # Map spectrogram frames to time, then to phase.
    total_s = 3.0
    frame_phase = np.array(["none"] * n_frames, dtype=object)
    for w in seg["windows"]:
        f0 = int(n_frames * (w["start"] / fs) / total_s)
        f1 = int(n_frames * (w["end"] / fs) / total_s)
        frame_phase[max(f0, 0):min(f1, n_frames)] = w["phase"]

    drops, per_frame = {}, {}
    for phase in ("systole", "diastole"):
        idx = np.where(frame_phase == phase)[0]
        if idx.size == 0:
            drops[phase] = per_frame[phase] = None
            continue
        occ = feat_int8.copy()
        occ[0, :, idx, 0] = zp_in       # zero in real terms == the zero point
        d = base_p - float(run(occ)[cls])
        drops[phase] = round(d, 4)
        # Normalise by frames removed. Diastole is simply longer than systole, so
        # raw drops reward whichever phase occupies more of the recording rather
        # than whichever phase carries the information. Without this the measure
        # reported "systole" for every class including Normal -- i.e. nothing.
        per_frame[phase] = round(d / float(idx.size), 6)

    valid = [k for k in per_frame if per_frame[k] is not None]
    return {
        "attribution_ok": True, "predicted_class": cls,
        "base_confidence": round(base_p, 4),
        "confidence_drop": drops,
        "confidence_drop_per_frame": per_frame,
        "driven_by": (max(valid, key=lambda k: per_frame[k]) if valid else None),
        # MEASURED OUTCOME: this does NOT discriminate. Over the held-out set it
        # answered "systole" for every class, Normal included -- i.e. it reflects
        # where the signal energy is, not where the diagnostic information is.
        # Per-frame normalisation improved it only marginally. It is therefore
        # NOT surfaced in the dashboard; the DSP phase-energy ratio is used
        # instead, which does separate all five classes. Kept here so the
        # negative result stays visible rather than being quietly dropped.
        "discriminative": False,
    }


def vision_zone_attribution(interp, img_int8, grid=3):
    """Which region of the radiograph drives the existing prediction?

    Same occlusion idea in 2D, on the deployed INT8 model. A 3x3 grid matches
    the coarse spatial resolution actually available at 96x96 -- claiming a
    finer localisation than the input supports would be false precision.
    Tier 1.
    """
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    zp_in = inp["quantization"][1]
    o_scale, o_zp = out["quantization"]

    def run(a):
        interp.set_tensor(inp["index"], a.astype(inp["dtype"]))
        interp.invoke()
        p = (interp.get_tensor(out["index"]).reshape(-1).astype(np.float32) - o_zp) * o_scale
        return p

    base = run(img_int8)
    cls = int(np.argmax(base))
    base_p = float(base[cls])

    h = img_int8.shape[1]
    step = h // grid
    heat = np.zeros((grid, grid), dtype=float)
    for r in range(grid):
        for c in range(grid):
            occ = img_int8.copy()
            occ[0, r * step:(r + 1) * step, c * step:(c + 1) * step, 0] = zp_in
            heat[r, c] = base_p - float(run(occ)[cls])

    r, c = np.unravel_index(int(np.argmax(heat)), heat.shape)
    rows = ["upper", "middle", "lower"]
    # viewer-left is the patient's right on a standard frontal film
    cols = ["patient right", "central", "patient left"]
    return {"attribution_ok": True, "predicted_class": cls,
            "base_confidence": round(base_p, 4),
            "heatmap": [[round(v, 4) for v in row] for row in heat],
            "peak_region": f"{rows[r]} / {cols[c]}"}
