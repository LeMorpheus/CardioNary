"""Single-clip white/pink noise and low-pass filtering, with a WAV CLI."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


def _audio(samples):
    x = np.asarray(samples)
    if x.ndim not in (1, 2) or x.shape[0] < 2 or (x.ndim == 2 and x.shape[1] == 0):
        raise ValueError("Expected mono (samples,) or multichannel (samples,channels) audio.")
    if np.issubdtype(x.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(x.dtype).max + 1) / 2
        x = (x.astype(float) - midpoint) / midpoint
    elif np.issubdtype(x.dtype, np.signedinteger):
        x = x.astype(float) / float(-np.iinfo(x.dtype).min)
    else:
        x = x.astype(float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Audio must be finite.")
    return x


def add_noise(samples, snr_db, kind="white", seed=0):
    """Add noise scaled to exact aggregate RMS SNR, without clipping.

    Pink noise uses 1/sqrt(f) amplitude shaping (1/f power), with DC removed.
    Channels receive independent noise. SNR is undefined for silent input,
    which raises ValueError. Floating WAV output preserves peaks above one.
    """
    x = _audio(samples)
    if kind not in ("white", "pink") or not np.isfinite(snr_db):
        raise ValueError("Choose white or pink noise and finite SNR in dB.")
    power = np.mean(x*x)
    if power == 0:
        raise ValueError("Cannot define SNR for a silent clip.")
    noise = np.random.default_rng(seed).normal(size=x.shape)
    if kind == "pink":
        frequencies = np.fft.rfftfreq(x.shape[0])
        scale = np.zeros_like(frequencies)
        scale[1:] = 1 / np.sqrt(frequencies[1:])
        if x.ndim == 2:
            scale = scale[:, None]
        noise = np.fft.irfft(np.fft.rfft(noise, axis=0) * scale, n=x.shape[0], axis=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gain = np.sqrt(power / np.mean(noise*noise)) * 10.0 ** (-float(snr_db)/20)
    if not np.isfinite(gain) or gain == 0:
        raise ValueError("SNR exceeds the supported numerical range.")
    result = x + noise * gain
    if not np.all(np.isfinite(result)):
        raise ValueError("Corruption overflowed; choose a less extreme SNR.")
    return result


def low_pass(samples, sample_rate, cutoff_hz, order=4):
    """Zero-phase Butterworth low-pass filter along the time axis."""
    x = _audio(samples)
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("Cutoff must be between zero and the Nyquist frequency.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("Filter order must be a positive integer.")
    sos = butter(order, cutoff_hz, fs=sample_rate, output="sos")
    try:
        return sosfiltfilt(sos, x, axis=0)
    except ValueError as exc:
        raise ValueError("Clip is too short for zero-phase filtering at this order.") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--noise", choices=["white", "pink"])
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--cutoff-hz", type=float)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if bool(args.noise) != (args.snr_db is not None) or (not args.noise and args.cutoff_hz is None):
        parser.error("Provide --noise with --snr-db, --cutoff-hz, or both.")
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Choose a new output path; existing files are not overwritten.")
    rate, samples = wavfile.read(args.input)
    samples = _audio(samples)
    if args.noise:
        samples = add_noise(samples, args.snr_db, args.noise, args.seed)
    if args.cutoff_hz is not None:
        samples = low_pass(samples, rate, args.cutoff_hz, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        wavfile.write(output, rate, samples.astype(np.float32))


if __name__ == "__main__":
    main()
