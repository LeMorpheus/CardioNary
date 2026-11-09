"""Evaluate the float heart CNN on BMD-HS and CirCor (roadmap S2).

With no model arguments, write provisional reference-model results to scratch/.
For the team run, supply the final float model and its matching normalisation
report, and set --out reports/heart_unseen.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from cardio.eval.agreement import REPO_ROOT, load_cached
from cardio.eval.intervals import metric_intervals
from cardio.model.final_model import CLASSES, STEM


SHARED_CLASSES = ("Normal", "AS", "MS", "MR")
BMD_DISEASES = ("AS", "AR", "MS", "MR")
REFERRAL_METRICS = ("auroc", "macro_f1", "sensitivity_at_90_specificity", "ece")


def patient_cohort(dataset, manifest):
    """Select labelled patients and keep the recording rows used for scoring."""
    patients = defaultdict(list)
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != dataset or not row["patient_id"]:
                raise ValueError(f"Unexpected dataset or missing patient ID in {manifest}")
            patients[row["patient_id"]].append(row)

    targets, shared, recordings = {}, {}, []
    exclusions = Counter()
    for patient, rows in patients.items():
        labels = [json.loads(row["labels"]) for row in rows]
        if dataset == "circor":
            murmur = [label.get("murmur") for label in labels]
            known = {label for label in murmur if label in ("Present", "Absent")}
            if len(known) != 1:
                exclusions["conflicting_murmur_labels" if known else "unknown_murmur"] += 1
                continue
            label = known.pop()
            targets[patient] = int(label == "Present")
            # Drop Unknown visits even if another visit has a known label.
            recordings.extend(row for row, value in zip(rows, murmur) if value == label)
            continue

        label = labels[0]
        if any(other != label for other in labels[1:]):
            exclusions["conflicting_disease_labels"] += 1
            continue
        if any(label.get(name) not in (0, 1) for name in (*BMD_DISEASES, "N")):
            exclusions["missing_disease_labels"] += 1
            continue
        diseases = [name for name in BMD_DISEASES if label[name] == 1]
        if bool(diseases) == bool(label["N"]):
            exclusions["inconsistent_disease_labels"] += 1
            continue
        targets[patient] = int(bool(diseases))
        if not diseases:
            shared[patient] = "Normal"
        elif len(diseases) == 1 and diseases[0] in SHARED_CLASSES:
            shared[patient] = diseases[0]
        recordings.extend(rows)

    counts = {
        "manifest_patients": len(patients),
        "manifest_recordings": sum(map(len, patients.values())),
        "excluded_patients": dict(exclusions),
        "recordings_with_eligible_labels": len(recordings),
    }
    return recordings, targets, shared, counts


def predict_patients(rows, predict, batch_size=256):
    """Average window probabilities per patient, weighting each window equally."""
    sums = defaultdict(lambda: np.zeros(len(CLASSES), dtype=np.float64))
    counts = Counter()
    used_recordings = 0
    pending, patient_ids = [], []

    def flush():
        if not pending:
            return
        probabilities = np.asarray(predict(np.concatenate(pending)), dtype=np.float64)
        if (probabilities.shape != (len(patient_ids), len(CLASSES))
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1.0)):
            raise ValueError("Expected five finite softmax probabilities per window")
        for patient, score in zip(patient_ids, probabilities):
            sums[patient] += score
            counts[patient] += 1
        pending.clear()
        patient_ids.clear()

    for index, row in enumerate(rows, 1):
        features = load_cached(row)
        used_recordings += int(len(features) > 0)
        start = 0
        while start < len(features):
            end = min(start + batch_size - len(patient_ids), len(features))
            pending.append(features[start:end])
            patient_ids.extend([row["patient_id"]] * (end - start))
            start = end
            if len(patient_ids) == batch_size:
                flush()
        if index % 250 == 0:
            print(f"Read {index}/{len(rows)} eligible recordings", flush=True)
    flush()
    return {patient: sums[patient] / count for patient, count in counts.items()}, used_recordings, sum(counts.values())


def evaluate(dataset, predict):
    manifest = REPO_ROOT / "data" / "manifests" / f"{dataset}_manifest.csv"
    rows, targets, shared, counts = patient_cohort(dataset, manifest)
    print(f"{dataset}: scoring {len(targets)} eligible patients", flush=True)
    probabilities, recordings, windows = predict_patients(rows, predict)
    patients = sorted(probabilities)
    if not patients:
        raise ValueError(f"No eligible patients with usable windows in {manifest}")

    truth = np.asarray([targets[patient] for patient in patients], dtype=int)
    scores = np.clip(1.0 - np.asarray([probabilities[patient][0] for patient in patients]), 0, 1)
    report = {
        **counts,
        "patients_without_windows": len(targets) - len(patients),
        "recordings_scored": recordings,
        "windows_scored": windows,
        "refer_or_not": {
            "patients": len(patients),
            "positive_patients": int(truth.sum()),
            "predicted_refer_patients": int(np.count_nonzero(scores >= 0.5)),
            "metrics": metric_intervals(truth, scores, patients, REFERRAL_METRICS),
        },
    }
    if dataset == "bmdhs":
        single_label_patients = [patient for patient in patients if patient in shared]
        if not single_label_patients:
            raise ValueError("No BMD-HS patients are eligible for the shared-class test")
        y = np.asarray([SHARED_CLASSES.index(shared[patient]) for patient in single_label_patients])
        columns = [CLASSES.index(name) for name in SHARED_CLASSES]
        p = np.asarray([probabilities[patient][columns] for patient in single_label_patients])
        totals = p.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("Cannot renormalise a BMD-HS prediction with zero shared-class probability")
        metrics = metric_intervals(y, p / totals, single_label_patients,
                                   ("macro_f1", *(f"recall_{i}" for i in range(len(SHARED_CLASSES)))))
        report["shared_classes"] = {
            "patients": len(single_label_patients),
            "class_support": {name: int(np.count_nonzero(y == i)) for i, name in enumerate(SHARED_CLASSES)},
            "macro_f1": metrics["macro_f1"],
            "per_class_recall": {name: metrics[f"recall_{i}"] for i, name in enumerate(SHARED_CLASSES)},
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Team's final float .keras model")
    parser.add_argument("--normalization-report", type=Path, help="JSON from the same training run")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "scratch" / "heart_unseen_reference.json")
    args = parser.parse_args(argv)
    if bool(args.model) != bool(args.normalization_report):
        parser.error("Supply --model and --normalization-report together.")
    provisional = args.model is None
    model_path = args.model or REPO_ROOT / "models" / f"{STEM}_float.keras"
    norm_path = args.normalization_report or REPO_ROOT / "reports" / f"quantization_{STEM}.json"
    if provisional and args.out.resolve() == (REPO_ROOT / "reports" / "heart_unseen.json").resolve():
        parser.error("Supply the team model and its matching report for reports/heart_unseen.json.")

    norm_bytes = norm_path.read_bytes()
    normalization = json.loads(norm_bytes)
    mean, std = float(normalization["norm_mean"]), float(normalization["norm_std"])
    if not np.isfinite([mean, std]).all() or std <= 0 or normalization.get("classes") != CLASSES:
        raise ValueError("Expected valid training-set normalisation and the heart model's class order")

    from tensorflow import keras

    model = keras.models.load_model(model_path, compile=False)

    def predict(features):
        normalized = ((features - mean) / std)[..., None].astype(np.float32)
        return model(normalized, training=False).numpy().reshape(len(features), -1)

    output = {
        "model_status": "provisional_reference" if provisional else "supplied_model",
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "normalization_report_sha256": hashlib.sha256(norm_bytes).hexdigest(),
        "normalization": {"mean": mean, "std": std},
        "classes": CLASSES,
        "method": {
            "patient_probability": "mean over eligible windows across all recordings for each patient",
            "referral_probability": "1 - p(Normal)",
            "referral_threshold": 0.5,
            "circor_label": "Present vs Absent; drop Unknown visits and patients with conflicting known labels",
            "bmdhs_label": "any annotated disease vs Normal; exclude missing labels",
            "bmdhs_shared_class_test": "single-label Normal, AS, MS, MR; remove MVP and renormalise",
            "intervals": "95% percentile bootstrap over patients, 1000 resamples, seed 0",
        },
        "datasets": {dataset: evaluate(dataset, predict) for dataset in ("bmdhs", "circor")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.out.parent, delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        temp_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp_path, args.out)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
