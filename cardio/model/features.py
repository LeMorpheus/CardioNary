import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from core.descriptors import _envelope, audio_descriptors
from core.features import CARDIO_CONFIG, load_wav

FS = CARDIO_CONFIG.sample_rate  # 8,000 samples per second


def heart_rate(x16, fs=FS, rate=100):
    """Heart rate in beats per minute, and regularity from 0 to 1."""
    e = _envelope(x16.astype(float), fs)[:: fs // rate]  # 100 values per second is plenty
    e = e - e.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1 :]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(0.33 * rate), min(int(1.5 * rate), len(e) // 2)  # need at least two beats
    if hi <= lo:
        return None, None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60.0 * rate / lag, float(ac[lag])


def characteristics(path):
    return characteristics_from_audio(load_wav(str(path), CARDIO_CONFIG))  # mono, 8 kHz, from any input rate


def characteristics_from_audio(x):
    """The same 17 characteristics from 8 kHz samples in [-1, 1], with no file.

    cardio/predict.py uses this so a dashboard upload is measured in memory.
    """
    x16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    d = audio_descriptors(x16, fs=FS)
    row = {f"band_{k}": v for k, v in d["band_power_pct"].items()}
    for k in (
        "spectral_centroid_hz",
        "murmur_duration_pct",
        "murmur_intensity_db",
        "systole_ms",
        "diastole_ms",
        "s1_s2_ratio_db",
        "systolic_ratio",
        "diastolic_ratio",
    ):
        row[k] = d.get(k)
    for k in ("murmur_detected", "click_suspected", "segmentation_ok"):
        row[k] = int(bool(d.get(k)))
    row["heart_rate_bpm"], row["regularity"] = heart_rate(x16)
    return row


def main():
    parser = argparse.ArgumentParser(description="Extract secondary acoustic characteristics.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=["yaseen", "physionet", "bmdhs"], help="Target dataset name.")
    group.add_argument("--folder", type=Path, help="Folder containing WAV files.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.dataset == "yaseen":
        manifest_path = Path("data/manifests/yaseen_manifest.csv")
        out_path = args.out or reports_dir / "heart_features_yaseen.csv"
        df_manifest = pd.read_csv(manifest_path)
        print(f"Extracting features for Yaseen ({len(df_manifest)} recordings)...")
        for idx, item in df_manifest.iterrows():
            feats = characteristics(item["path"])
            row = {
                "patient_id": item["patient_id"],
                "fold": item["fold"],
                "label": item["label"],
                **feats,
            }
            rows.append(row)
    elif args.dataset in ("physionet", "bmdhs"):
        manifest_name = "circor" if args.dataset == "physionet" else "bmdhs"
        manifest_path = Path("data/manifests") / f"{manifest_name}_manifest.csv"
        out_path = args.out or reports_dir / f"heart_features_{args.dataset}.csv"
        df_manifest = pd.read_csv(manifest_path)
        print(f"Extracting features for {args.dataset} ({len(df_manifest)} recordings)...")
        for idx, item in df_manifest.iterrows():
            feats = characteristics(item["path"])
            row = {
                "patient_id": item["patient_id"],
                "path": item["path"],
                **feats,
            }
            rows.append(row)
    elif args.folder:
        folder_path = Path(args.folder)
        out_path = args.out or reports_dir / f"heart_features_{folder_path.name}.csv"
        wav_files = sorted(list(folder_path.glob("*.wav")) + list(folder_path.glob("*.WAV")))
        print(f"Extracting features for {len(wav_files)} recordings in {folder_path}...")
        for wav in wav_files:
            feats = characteristics(wav)
            row = {
                "path": str(wav),
                **feats,
            }
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"Saved {len(df_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
