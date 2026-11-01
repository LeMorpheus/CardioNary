"""Live-sample input: upload a radiograph, or capture audio, and run it on a node.

The nodes normally classify golden vectors held in their own flash. This module
adds the other direction: take a sample the gateway holds, push it to the node
over the Sample Upload characteristic, and ask the node to run it through the
same front-end and the same model.

Two sources:

  image  -- any format Pillow can open, converted to 96x96 8-bit greyscale and
            sent to the pulmonary node.
  audio  -- captured from the I2S MEMS microphone via arecord, or supplied as a
            WAV, resampled to 8 kHz mono and sent to the cardiac node as int16
            PCM. The mel front-end still runs on the Dev Kit, so what is being
            demonstrated is unchanged; only the source of the PCM differs.

Chunk format on the wire is offset16 (LE) || payload. Writing at offset 0 resets
the node's staging buffer, so an interrupted transfer cannot be mistaken for a
resumed one.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import os
import math
import struct
import subprocess
import wave

LOG = logging.getLogger("chiron.live")

# --- sample geometry, must match the firmware ------------------------------
IMG_W = IMG_H = 96
IMG_BYTES = IMG_W * IMG_H                 # 9,216 uint8 pixels

AUDIO_SR = 8000
AUDIO_SAMPLES = 24000                     # 3.0 s
AUDIO_BYTES = AUDIO_SAMPLES * 2           # int16 little-endian

# Conservative chunk payload: 20 bytes fits the default ATT MTU, but every stack
# we talk to negotiates far above that. 180 keeps us inside a 247-byte MTU with
# room for the 2-byte offset header and ATT overhead.
CHUNK = 180

# ALSA card INDICES ARE NOT STABLE ACROSS REBOOTS. The I2S MEMS mic has been
# card 3 and card 2 on this same Pi, and the index it vacates gets taken by an
# unrelated playback device -- which fails as "audio open error: No such file
# or directory" and looks like the microphone is missing. Resolve by card NAME
# instead; ALSA accepts plughw:CARD=<id>,DEV=0 and does the lookup itself.
CAPTURE_DEVICE_ENV = "CHIRON_CAPTURE_DEVICE"
# ALSA truncates card ids to 15 characters, so the googlevoicehat overlay
# appears as "sndrpigooglevoi". Match on fragments, not full names.
CAPTURE_CARD_HINTS = ("googlevoi", "sndrpi", "i2s", "mems", "voicehat")
CAPTURE_SR = 48000
CAPTURE_SECONDS = 5


def _capture_cards():
    """Card ids that can actually capture, in `arecord -l` order."""
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True,
                             timeout=10).stdout.decode(errors="replace")
    except Exception as exc:
        LOG.warning("arecord -l failed: %s", exc)
        return []
    cards = []
    for line in out.splitlines():
        m = re.match(r"card (\d+): (\S+)", line.strip())
        if m:
            cards.append((int(m.group(1)), m.group(2)))
    return cards


def resolve_capture_device():
    """Return an ALSA device string for the MEMS microphone.

    An explicit override always wins. Otherwise prefer a card whose id looks
    like the I2S mic, and fall back to the first capture-capable card so a
    differently-named HAT still works.
    """
    override = os.environ.get(CAPTURE_DEVICE_ENV)
    if override:
        return override

    cards = _capture_cards()
    if not cards:
        raise RuntimeError(
            "no ALSA capture device found. Check the I2S microphone overlay "
            "(`arecord -l` should list a capture card).")

    for _idx, cid in cards:
        low = cid.lower()
        if any(h in low for h in CAPTURE_CARD_HINTS):
            return "plughw:CARD=%s,DEV=0" % cid

    idx, cid = cards[0]
    LOG.warning("no I2S-looking capture card; falling back to card %d (%s)", idx, cid)
    return "plughw:CARD=%s,DEV=0" % cid


# ===========================================================================
# Conversion
# ===========================================================================
def image_to_payload(raw: bytes) -> bytes:
    """Decode an uploaded image and return 9,216 greyscale bytes.

    The firmware applies `int8 = pixel - 128` itself, so what goes over the wire
    is plain uint8 exactly as the golden vectors are stored.
    """
    try:
        from PIL import Image
    except ImportError as exc:                                # pragma: no cover
        raise RuntimeError(
            "Pillow is required to accept image uploads. Install it into the "
            "gateway venv: pip install pillow") from exc

    img = Image.open(io.BytesIO(raw))
    img = img.convert("L").resize((IMG_W, IMG_H), Image.BILINEAR)
    out = img.tobytes()
    if len(out) != IMG_BYTES:
        raise ValueError("expected %d greyscale bytes, got %d"
                         % (IMG_BYTES, len(out)))
    return out


def _decode_wav(raw: bytes):
    """Return (samples as a list of float in -1..1, sample_rate).

    Handles the 8/16/24/32-bit PCM cases we actually see. The MEMS capture is
    32-bit stereo, and only channel 0 carries signal -- channel 1 is the unused
    I2S slot, so averaging the channels would halve the amplitude for nothing.
    """
    with wave.open(io.BytesIO(raw), "rb") as w:
        ch, width, sr, n = (w.getnchannels(), w.getsampwidth(),
                            w.getframerate(), w.getnframes())
        frames = w.readframes(n)

    if width == 2:
        fmt, full = "<%dh" % (len(frames) // 2), 32768.0
        vals = struct.unpack(fmt, frames)
    elif width == 4:
        fmt, full = "<%di" % (len(frames) // 4), 2147483648.0
        vals = struct.unpack(fmt, frames)
    elif width == 1:
        vals = [b - 128 for b in frames]
        full = 128.0
    elif width == 3:
        vals = []
        for i in range(0, len(frames), 3):
            v = frames[i] | (frames[i + 1] << 8) | (frames[i + 2] << 16)
            if v & 0x800000:
                v -= 1 << 24
            vals.append(v)
        full = 8388608.0
    else:
        raise ValueError("unsupported sample width %d" % width)

    if ch > 1:
        vals = vals[::ch]                 # channel 0 only, see docstring
    return [v / full for v in vals], sr


def _resample(x, src_sr: int, dst_sr: int):
    """Linear resample. Adequate here: the signal is band-limited well below
    the 4 kHz Nyquist of the destination rate, and the model's own front-end
    applies a mel filterbank immediately afterwards."""
    if src_sr == dst_sr:
        return list(x)
    ratio = dst_sr / float(src_sr)
    n_out = int(len(x) * ratio)
    out = []
    for i in range(n_out):
        pos = i / ratio
        j = int(pos)
        frac = pos - j
        a = x[j] if j < len(x) else 0.0
        b = x[j + 1] if j + 1 < len(x) else a
        out.append(a + (b - a) * frac)
    return out


def audio_to_payload(raw_wav: bytes, normalise="auto") -> bytes:
    """Convert a WAV into exactly 24,000 int16 samples at 8 kHz.

    A window is taken from the middle of the recording, which avoids the contact
    transients that cluster at the start and end of a hand-held capture. Short
    input is zero-padded; the node classifies whatever it is given.
    """
    xs, sr = _decode_wav(raw_wav)
    xs = _resample(xs, sr, AUDIO_SR)

    if len(xs) > AUDIO_SAMPLES:
        start = (len(xs) - AUDIO_SAMPLES) // 2
        xs = xs[start:start + AUDIO_SAMPLES]
    elif len(xs) < AUDIO_SAMPLES:
        xs = xs + [0.0] * (AUDIO_SAMPLES - len(xs))

    # "auto" is the right default for a UI that accepts both microphone captures
    # and replayed reference vectors. Microphone input sits near -25 dBFS and
    # needs the gain; a reference vector is already at the training level and
    # must pass through untouched, because scaling it changes the answer.
    peak = max((abs(v) for v in xs), default=0.0)
    mode = str(normalise).lower()
    if mode in ("auto", "true", "1", "yes"):
        want = (mode != "auto") or (peak < 0.25)
        if want and peak > 1e-6:
            gain = min(0.85 / peak, 64.0)
            xs = [v * gain for v in xs]
            LOG.info("applied %.1fx gain (peak was %.4f)", gain, peak)
        else:
            LOG.info("no gain applied (peak %.4f)", peak)

    pcm = bytearray()
    for v in xs:
        s = int(max(-1.0, min(1.0, v)) * 32767.0)
        pcm += struct.pack("<h", s)
    if len(pcm) != AUDIO_BYTES:
        raise ValueError("expected %d PCM bytes, got %d" % (AUDIO_BYTES, len(pcm)))
    return bytes(pcm)


# ===========================================================================
# Capture
# ===========================================================================
async def capture_wav(seconds: int = CAPTURE_SECONDS,
                      device: str | None = None) -> bytes:
    """Record from the MEMS microphone and return WAV bytes.

    The device is resolved per call rather than cached, so a mic that moves
    between reboots -- or is plugged in after the gateway started -- is picked
    up without a restart.
    """
    if device is None:
        device = resolve_capture_device()
    cmd = ["arecord", "-D", device, "-f", "S32_LE", "-r", str(CAPTURE_SR),
           "-c", "2", "-d", str(seconds), "-t", "wav", "-q", "-"]
    LOG.info("capturing %d s from %s", seconds, device)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0 or not out:
        detail = err.decode(errors="replace").strip()[:240]
        cards = ", ".join("%d:%s" % c for c in _capture_cards()) or "none"
        raise RuntimeError("arecord failed on %s (%s): %s [capture cards: %s]"
                           % (device, proc.returncode, detail, cards))
    return out


# ===========================================================================
# Transfer
# ===========================================================================
async def push_sample(client, char_upload: str, payload: bytes,
                      chunk: int = CHUNK) -> float:
    """Write the payload to the node in offset-tagged chunks.

    Returns the seconds spent QUEUING the writes, which is not the delivery
    time. Write-without-response returns as soon as the host stack accepts a
    packet; actual delivery is paced by the BLE connection interval. Measured on
    this link: 9 kB (an image) lands in well under a second, 48 kB (three
    seconds of PCM) takes on the order of ten seconds. Do not present this
    number to an operator as a transfer time.

    A dropped chunk shows up as a short offset high-water mark on the node,
    which the firmware reports rather than silently classifying a partial
    buffer.
    """
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    for off in range(0, len(payload), chunk):
        part = payload[off:off + chunk]
        await client.write_gatt_char(char_upload,
                                     struct.pack("<H", off) + part,
                                     response=False)
    return loop.time() - t0

# ===========================================================================
# Signal provenance
# ===========================================================================
# The cardiac model was trained on Yaseen: clean, curated teaching recordings.
# Audio captured through a MEMS microphone on a stethoscope is a different
# signal chain, and the model has never been validated on it. A softmax
# classifier has no way to say so -- it returns a confident class regardless --
# which is precisely the documented failure mode this gate exists to catch.
#
# The discriminator is the fraction of spectral energy above 1 kHz, which is
# essentially absent from the training corpus and present in contact/handling
# noise. Measured on 53 training vectors and 50 field recordings:
#
#     training    median 0.00002   p95 0.00018   max 0.00133
#     field       median 0.00428   p05 0.00087   min 0.00014
#
# A threshold of 0.0015 flags 82 % of field recordings and 0 % of the training
# set. It is a measurement, not a tuned guess, and it is deliberately set where
# no reference vector is ever flagged.
OOD_HIBAND_THRESHOLD = 0.0015


def signal_provenance(pcm_bytes: bytes):
    """Judge whether a sample lies inside the envelope the model was validated on.

    Returns {in_distribution, hi_band, threshold}. This does not touch the model
    or its output; it qualifies the INPUT, so the dashboard can decline to
    present an unvalidated class as a clinical finding.

    Statistic: fraction of spectral energy at or above 1 kHz, on a
    peak-normalised, Hann-windowed frame. Level-invariant by construction, so a
    quiet capture and a loud one are judged the same way.

    Two earlier attempts were wrong and are worth recording. A decimated
    Goertzel put Nyquist below the band of interest and read zero for every
    input. A one-pole high-pass at 1 kHz has alpha ~= 0.56 at this sample rate
    -- far too gentle to isolate the band -- and inverted the relationship. The
    threshold below was measured with THIS estimator and no other.
    """
    n = len(pcm_bytes) // 2
    if n < 512:
        return {"in_distribution": None, "hi_band": None,
                "threshold": OOD_HIBAND_THRESHOLD}
    try:
        import numpy as np
    except ImportError:
        # No numpy: report unknown rather than guess. An unqualified result is
        # better than a qualified one built on a statistic we cannot compute.
        return {"in_distribution": None, "hi_band": None,
                "threshold": OOD_HIBAND_THRESHOLD}

    x = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float64)
    x /= (np.abs(x).max() + 1e-12)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / AUDIO_SR)
    p = X ** 2 + 1e-20
    ratio = float(p[f >= 1000.0].sum() / p.sum())
    return {"in_distribution": bool(ratio <= OOD_HIBAND_THRESHOLD),
            "hi_band": round(ratio, 6),
            "threshold": OOD_HIBAND_THRESHOLD}

# ===========================================================================
# Spectrogram
# ===========================================================================
# Rendered only for microphone captures. An uploaded file already tells the
# operator what it is by its name; a live capture has no such handle, and the
# spectrogram is the only way to see whether the stethoscope was actually
# picking up a heartbeat rather than handling noise.
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
