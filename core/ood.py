"""Out-of-distribution check for heart-sound recordings.

This qualifies the INPUT, not the model. It asks whether a recording resembles
the data the heart-sound model was validated on, so that a result from an
unfamiliar recording can be withheld rather than presented as a finding.

Extracted unchanged from the earlier project's live-input code; the sample rate
now comes from core.features so it is defined in one place.
"""

from __future__ import annotations

from core.features import CARDIO_CONFIG

AUDIO_SR = CARDIO_CONFIG.sample_rate      # 8 kHz, the rate the model was trained at

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
