"""Display spectrogram for heart-sound recordings.

Renders the log-magnitude spectrogram of the same 8 kHz audio the model
classifies, as a PNG data URI: a two-tone colour map that separates the strong
signal from the quiet background, with labelled time and frequency axes.

Extracted unchanged from the earlier project's live-input code; the sample rate
now comes from core.features so it is defined in one place.
"""

from __future__ import annotations

import io

from core.features import CARDIO_CONFIG

AUDIO_SR = CARDIO_CONFIG.sample_rate      # 8 kHz, the rate the model was trained at

SPEC_FRAMES = 192          # matches the model's frame count
SPEC_BINS = 96             # display resolution, not the model's 40 mels
SPEC_FLOOR_DB = -70.0
SPEC_FMAX = 1000.0         # heart sounds sit well below this; 4 kHz is empty

# Two-tone map: background in muted slate so it recedes, signal in the
# dashboard's green so the eye goes straight to it.
#
# The split is a PERCENTILE of the frame, not a fixed level. A fixed threshold
# put almost every cell of a quiet field recording into the background tone and
# the picture carried no information. Taking the strongest cells of whatever was
# actually recorded means the distinctive region is always visible, at any
# capture level, which is the whole point of showing it.
SPEC_SPLIT_PCTL = 82.0
SPEC_SPLIT_MIN = 0.18
SPEC_LOW_RGB = (28, 42, 58)      # quiet region, floor
SPEC_LOW_HI_RGB = (58, 84, 112)  # quiet region, top of its range
SPEC_HI_RGB = (34, 120, 58)      # signal region, floor
SPEC_HI_TOP_RGB = (126, 245, 150)  # signal region, peak

AXIS_RGB = (150, 196, 232)       # distinct from both ramps
AXIS_BG_RGB = (10, 14, 19)

PAD_L, PAD_B, PAD_T, PAD_R = 46, 26, 20, 12


def spectrogram_png(pcm_bytes: bytes) -> str | None:
    """Return a base64 PNG data URI of the log-magnitude spectrogram.

    Computed from the same 8 kHz PCM the node receives, so what is displayed is
    the signal that was actually classified. Axes are drawn in a colour used
    nowhere else in the plot, so the scale can never be mistaken for data.
    Returns None if the optional dependencies are missing.
    """
    try:
        import base64
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    x = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float64)
    if x.size < 512:
        return None
    x /= (np.abs(x).max() + 1e-12)
    duration_s = x.size / float(AUDIO_SR)

    n_fft = 512
    hop = max(1, (x.size - n_fft) // SPEC_FRAMES)
    win = np.hanning(n_fft)

    cols = []
    for i in range(SPEC_FRAMES):
        a = i * hop
        seg = x[a:a + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        cols.append(np.abs(np.fft.rfft(seg * win)))
    S = np.stack(cols, axis=1)

    f = np.fft.rfftfreq(n_fft, 1.0 / AUDIO_SR)
    S = S[f <= SPEC_FMAX]
    idx = np.linspace(0, S.shape[0] - 1, SPEC_BINS).astype(int)
    S = S[idx]

    db = 20.0 * np.log10(S + 1e-10)
    db = np.clip(db - db.max(), SPEC_FLOOR_DB, 0.0)
    norm = (db - SPEC_FLOOR_DB) / (-SPEC_FLOOR_DB)
    norm = np.flipud(norm)                       # low frequencies at the bottom

    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    split = max(SPEC_SPLIT_MIN, float(np.percentile(norm, SPEC_SPLIT_PCTL)))
    if split >= 0.999:
        split = 0.999
    quiet = norm < split
    loud = ~quiet

    def ramp(t, c0, c1):
        t = t[..., None]
        return (np.array(c0) + t * (np.array(c1) - np.array(c0))).astype(np.uint8)

    if quiet.any():
        t = np.clip(norm[quiet] / split, 0.0, 1.0)
        rgb[quiet] = ramp(t, SPEC_LOW_RGB, SPEC_LOW_HI_RGB)
    if loud.any():
        t = np.clip((norm[loud] - split) / max(1e-6, 1.0 - split), 0.0, 1.0)
        rgb[loud] = ramp(t, SPEC_HI_RGB, SPEC_HI_TOP_RGB)

    # 3x the frame grid: for a capture this image is the whole result, so it
    # needs to carry detail at full card width rather than be upscaled by the
    # browser into mush.
    plot = Image.fromarray(rgb, mode="RGB").resize(
        (SPEC_FRAMES * 3, SPEC_BINS * 3), Image.BILINEAR)
    pw, ph = plot.size

    canvas = Image.new("RGB", (PAD_L + pw + PAD_R, PAD_T + ph + PAD_B), AXIS_BG_RGB)
    canvas.paste(plot, (PAD_L, PAD_T))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x0, y0, x1, y1 = PAD_L, PAD_T, PAD_L + pw, PAD_T + ph
    d.rectangle([x0 - 1, y0 - 1, x1, y1], outline=AXIS_RGB)

    # Frequency axis: 0 at the bottom, SPEC_FMAX at the top.
    for hz in (0, 250, 500, 750, 1000):
        yy = y1 - int((hz / SPEC_FMAX) * ph)
        yy = min(max(yy, y0), y1)
        d.line([x0 - 4, yy, x0 - 1, yy], fill=AXIS_RGB)
        if font:
            # Nudge the extreme labels inward so they stay inside the canvas
            # and clear of the unit caption.
            ty = yy - 4
            if hz == 1000:
                ty = yy + 1
            elif hz == 0:
                ty = yy - 8
            d.text((14, ty), "%d" % hz, fill=AXIS_RGB, font=font)
    if font:
        d.text((2, 3), "Hz", fill=AXIS_RGB, font=font)

    # Time axis across the actual duration of the buffer.
    ticks = 4
    for i in range(ticks + 1):
        tsec = duration_s * i / ticks
        xx = x0 + int((i / ticks) * pw)
        d.line([xx, y1 + 1, xx, y1 + 4], fill=AXIS_RGB)
        if font:
            lbl = "%.1f" % tsec
            tx = xx - 8
            if i == 0:
                tx = xx - 2
            elif i == ticks:
                tx = xx - 16
            d.text((tx, y1 + 6), lbl, fill=AXIS_RGB, font=font)
    if font:
        # Unit under the left end of the axis, where no tick label sits.
        d.text((2, y1 + 6), "s", fill=AXIS_RGB, font=font)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
