"""Build recording manifests from local BMD-HS and CirCor dataset releases.

Parsers use dataset-relative paths; the CLI writes repository-relative paths.
Labels and metadata are JSON objects in CSV cells. No task-specific cross-dataset
label mapping is inferred. Extraction uses the shared core.features front-end.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _recording(root, path):
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Recording escapes dataset root: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved.relative_to(root.resolve()).as_posix()


def bmdhs_rows(root):
    """Expand train.csv recording_1..8; preserve multilabel annotations.

    Optional additional_metadata.csv is joined by patient_id. Unknown label
    values (blank, N/A, nan) become JSON null, never a negative class.
    """
    root = Path(root)
    extra = {}
    if (root / "additional_metadata.csv").exists():
        for row in _csv(root / "additional_metadata.csv"):
            patient = row.pop("patient_id")
            if not patient or patient in extra:
                raise ValueError("Missing or duplicate patient ID in metadata.")
            extra[patient] = row
    rows, patients = [], set()
    for row in _csv(root / "train.csv"):
        patient = row.get("patient_id", "").strip()
        if not patient or patient in patients:
            raise ValueError("Missing or duplicate patient ID in train.csv.")
        patients.add(patient)
        label_cols = [c for c in ("AS", "AR", "MR", "MS", "MD", "N") if c in row]
        if not label_cols:
            raise ValueError("BMD-HS train.csv has no recognized label columns.")
        labels = {}
        for key in label_cols:
            value = row[key].strip()
            if value.lower() in ("", "n/a", "nan"):
                labels[key] = None
            elif value in ("0", "1"):
                labels[key] = int(value)
            else:
                raise ValueError(f"Invalid {key} label for {patient}: {value}")
        recordings = [key for key in row if key.startswith("recording_")]
        for key in recordings:
            name = row[key].strip()
            if not name or name.lower() in ("nan", "n/a"):
                continue
            name = name if Path(name).suffix else name + ".wav"
            path = _recording(root, Path("train") / name)
            rows.append(dict(dataset="bmdhs", path=path, patient_id=patient,
                             label_scope="patient", labels=labels.copy(),
                             metadata={**extra.get(patient, {}), "recording_slot": key}))
    return _check_rows(rows)


def circor_rows(root):
    """Parse official subject descriptions; missing hidden-set labels stay null."""
    root = Path(root)
    rows, patients = [], set()
    for description in sorted(root.rglob("*.txt")):
        lines = description.read_text(encoding="utf-8-sig").splitlines()
        if not lines:
            continue
        header = lines[0].split()
        if not header:
            continue
        # Ignore README/license files, but fail on malformed subject files.
        if description.stem != header[0]:
            continue
        if len(header) < 3:
            raise ValueError(f"Malformed patient header: {description}")
        patient, count = header[0], int(header[1])
        if count < 1 or patient in patients or len(lines) < count + 1:
            raise ValueError(f"Invalid/duplicate patient description: {description}")
        patients.add(patient)
        metadata = {}
        for line in lines[count+1:]:
            if line.startswith("#") and ":" in line:
                key, value = line[1:].split(":", 1)
                metadata[key.strip()] = value.strip()
        labels = {key.lower(): metadata.pop(key, None) for key in ("Murmur", "Outcome")}
        for line in lines[1:count+1]:
            parts = line.split()
            wavs = [value for value in parts[1:] if value.lower().endswith(".wav")]
            if len(wavs) != 1:
                raise ValueError(f"Expected one WAV per recording line: {description}")
            path = _recording(root, description.parent.relative_to(root) / wavs[0])
            rows.append(dict(dataset="circor", path=path, patient_id=patient,
                             label_scope="patient", labels=labels.copy(),
                             metadata={**metadata, "location": parts[0], "sample_rate": header[2]}))
    # Repeated visits can carry different subject IDs. Additional ID links
    # those visits to the same patient for patient-level bootstrap/splits.
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    for row in rows:
        patient = row['patient_id']
        other = row['metadata'].get('Additional ID', '')
        if other and other.lower() not in ('nan', 'n/a'):
            a, b = find(patient), find(other)
            parent[max(a, b)] = min(a, b)
    for row in rows:
        row['metadata']['subject_id'] = row['patient_id']
        row['patient_id'] = find(row['patient_id'])
    return _check_rows(rows)


def _check_rows(rows):
    if not rows:
        raise ValueError("No recordings found; check the dataset layout/root.")
    paths = [row["path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("A recording appears more than once in the manifest.")
    return sorted(rows, key=lambda row: row["path"])


def build_caches(rows, root, cache_dir, config, feature_fn=None, workers=1):
    """Call the project extractor once per file and save numeric NPZ caches.

    Each invocation rebuilds caches, avoiding stale configuration reuse.
    Extractors may return an array or a dict of numeric arrays. Other return
    contracts require adaptation after inspecting the real core.features.
    """
    if feature_fn is None:
        from core.features import features_for_file
        feature_fn = features_for_file
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    root, cache_dir = Path(root), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    def cache_one(row):
        audio_path = root / row["path"]
        features = feature_fn(str(audio_path), config)
        values = features if isinstance(features, dict) else {"features": np.asarray(features, dtype=np.float32)}
        if not values or any(not isinstance(key, str) or key == "file" for key in values):
            raise ValueError("Features must be an array or a nonempty dict of numeric arrays.")
        values = {key: np.asarray(value) for key, value in values.items()}
        if any(value.dtype.kind not in "biufc" for value in values.values()):
            raise TypeError("Only numeric feature arrays can be cached without pickle.")
        key = hashlib.sha256(row["path"].encode()).hexdigest()
        name = f"{row['dataset']}_{key}.npz"
        destination = cache_dir / name
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as output:
            temporary = Path(output.name)
        try:
            np.savez_compressed(temporary, **values)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {**row, "cache_path": name}
    result = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cached in pool.map(cache_one, rows):
            result.append(cached)
            if len(result) % 100 == 0:
                print(f"Cached {len(result)}/{len(rows)} recordings", flush=True)
    return result


def write_manifest(rows, output):
    rows = _check_rows(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "path", "patient_id", "label_scope", "labels", "metadata", "cache_path"]
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "labels": json.dumps(row["labels"], sort_keys=True),
                                 "metadata": json.dumps(row["metadata"], sort_keys=True)})
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bmdhs", "circor"], required=True)
    parser.add_argument("--root", type=Path, help="Defaults to data/raw/<dataset> in this repository")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/cache/unseen")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Parallel recording extractors (default: 4)")
    parser.add_argument("--config", default="core.features:CARDIO_CONFIG", help="Config module:attribute")
    args = parser.parse_args(argv)
    if not args.manifest_only and not args.config:
        parser.error("Supply the actual --config module:attribute, or use --manifest-only.")
    root = (args.root or REPO_ROOT / 'data/raw' / args.dataset).resolve()
    rows = (bmdhs_rows if args.dataset == "bmdhs" else circor_rows)(root)
    if not args.manifest_only:
        module, attr = args.config.split(":", 1)
        config = getattr(importlib.import_module(module), attr)
        rows = build_caches(rows, root, args.cache_dir, config, workers=args.workers)
        config_output = args.cache_dir / f'{args.dataset}_config.json'
        config_output.write_text(json.dumps(config.to_dict(), indent=2), encoding='utf-8')
    # Match the shared pipeline: committed paths are repository-relative,
    # including cache references. External roots are allowed for ad-hoc runs
    # but absolute machine-specific paths should not be committed.
    for row in rows:
        path = root / row['path']
        row['path'] = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else path.as_posix()
        if row.get('cache_path'):
            cache = args.cache_dir.resolve() / row['cache_path']
            row['cache_path'] = cache.relative_to(REPO_ROOT).as_posix() if cache.is_relative_to(REPO_ROOT) else cache.as_posix()
    output = args.output or REPO_ROOT / f"data/manifests/{args.dataset}_manifest.csv"
    write_manifest(rows, output)
    print(f"Wrote {len(rows)} recordings to {output}")


if __name__ == "__main__":
    main()
