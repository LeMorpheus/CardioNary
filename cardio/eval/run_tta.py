"""S3 test-time adaptation on public heart-sound datasets and controlled shifts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

from cardio.eval.agreement import load_cached
from cardio.eval.corrupt import add_noise, low_pass
from cardio.eval.run_unseen import patient_cohort
from cardio.model.final_model import CLASSES, MODEL_DIR, REPO_ROOT, STEM, FinalModel
from cardio.tta.bn_adapt import RecordingBatch, adapt_bn
from core.features import CARDIO_CONFIG, extract, iter_windows, load_wav, preprocess


SITES = ("bmdhs", "circor", "white_noise", "low_pass")
SIZES = (8, 16, 32, 64)
N_BOOTSTRAP = 1000
METRICS = ("macro_f1", "auroc")


@dataclass(frozen=True)
class Recording:
    patient: str
    label: int
    load: Callable[[], np.ndarray]


def yaseen_recordings():
    """Fold 0 clips with usable windows, in manifest order."""
    clean_windows = defaultdict(list)
    with np.load(REPO_ROOT / "data/cache/yaseen_logmel.npz", allow_pickle=False) as cache:
        holdout = cache["fold"] == 0
        for patient, features in zip(cache["patient_id"][holdout], cache["X"][holdout]):
            clean_windows[str(patient)].append(features)
    rows = []
    with (REPO_ROOT / "data/manifests/yaseen_manifest.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            patient = row["patient_id"]
            if row["fold"] == "0" and patient in clean_windows:
                rows.append((patient, int(row["diagnosis"] != "Normal"), REPO_ROOT / row["path"],
                             np.stack(clean_windows[patient])))
    return rows


def corrupted_windows(path, site, seed, *, severity=None):
    samples = load_wav(str(path), CARDIO_CONFIG)
    if site in ("white_noise", "pink_noise"):
        shifted = add_noise(samples, 5 if severity is None else severity,
                            kind=site.removesuffix("_noise"), seed=seed)
    elif site == "low_pass":
        shifted = low_pass(samples, CARDIO_CONFIG.sample_rate, 300 if severity is None else severity)
    else:
        raise ValueError(f"Unsupported audio shift: {site}")
    signal = preprocess(shifted, CARDIO_CONFIG)
    return np.stack([extract(window, CARDIO_CONFIG) for window in iter_windows(signal, CARDIO_CONFIG)])


def site_recordings(site, yaseen):
    if site in ("bmdhs", "circor"):
        manifest = REPO_ROOT / "data/manifests" / f"{site}_manifest.csv"
        rows, labels, _, _ = patient_cohort(site, manifest)
        return [Recording(row["patient_id"], labels[row["patient_id"]], lambda row=row: load_cached(row))
                for row in rows]
    return [Recording(patient, label, lambda path=path, i=i: corrupted_windows(path, site, i))
            for i, (patient, label, path, _) in enumerate(yaseen)]


def split_patients(records, max_score_patients=None):
    """Reserve a fixed, stratified quarter of patients for all draws at a site."""
    labels = {row.patient: row.label for row in records}
    patients = sorted(labels)
    pool, test = train_test_split(patients, test_size=0.25, random_state=0,
                                   stratify=[labels[patient] for patient in patients])
    if max_score_patients is not None and len(test) > max_score_patients:
        _, test = train_test_split(test, test_size=max_score_patients, random_state=0,
                                   stratify=[labels[patient] for patient in test])
    if len({labels[patient] for patient in test}) != 2:
        raise ValueError("The scoring panel needs both classes. Increase --max-score-patients.")
    pool, test = set(pool), set(test)
    return ([row for row in records if row.patient in pool],
            [row for row in records if row.patient in test])


def draw_indices(pool, count, fraction, rng):
    """Sample recordings without replacement. Labels only define stress batches."""
    if fraction is None:
        return rng.choice(len(pool), size=count, replace=False)
    positive = int(round(count * fraction))
    labels = np.array([row.label for row in pool])
    return np.concatenate([
        rng.choice(np.flatnonzero(labels == 0), size=count - positive, replace=False),
        rng.choice(np.flatnonzero(labels == 1), size=positive, replace=False),
    ])


@dataclass(frozen=True)
class EvaluationSet:
    batch: RecordingBatch
    patients: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    window_counts: np.ndarray

    @classmethod
    def from_records(cls, records, normalise):
        patients, groups = np.unique([row.patient for row in records], return_inverse=True)
        label_by_patient = {row.patient: row.label for row in records}
        batch = RecordingBatch.from_recordings([normalise(row.load()) for row in records])
        return cls(batch, patients, np.array([label_by_patient[patient] for patient in patients]),
                   groups, np.bincount(groups, weights=batch.lengths))

    def scores(self, model):
        probability_normal = self.batch.predict(model)[:, 0]
        total = np.bincount(self.groups, weights=probability_normal * self.batch.lengths)
        return np.clip(1 - total / self.window_counts, 0, 1)


def resample_weights(count, rng):
    """First row is the point estimate, followed by patient bootstrap counts."""
    return np.vstack((np.ones(count), rng.multinomial(count, np.full(count, 1 / count), N_BOOTSTRAP)))


def binary_metrics(labels, scores, weights):
    """Compute macro-F1 at 0.5 and tie-aware AUROC for each patient resample."""
    positive = labels == 1
    refer = scores >= 0.5
    pos = weights @ positive
    neg = weights @ ~positive
    tp = weights @ (positive & refer)
    fp = weights @ (~positive & refer)
    fn, tn = pos - tp, neg - fp

    def f1(correct, errors):
        denominator = 2 * correct + errors
        return np.divide(2 * correct, denominator, out=np.zeros_like(correct), where=denominator > 0)

    # In AUROC, a positive-negative pair earns one point for correct order
    # and half a point for a tie. Multiplicities weight pairs in each resample.
    above = scores[positive, None] > scores[None, ~positive]
    tied = scores[positive, None] == scores[None, ~positive]
    pairs = np.sum((weights[:, positive] @ (above + 0.5 * tied)) * weights[:, ~positive], axis=1)
    auc = np.divide(pairs, pos * neg, out=np.full(len(weights), np.nan), where=pos * neg > 0)
    return {"macro_f1": (f1(tp, fp + fn) + f1(tn, fp + fn)) / 2, "auroc": auc}


def interval(values):
    valid = values[1:][np.isfinite(values[1:])]
    lower, upper = np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else (None, None)
    return {"estimate": float(values[0]) if np.isfinite(values[0]) else None,
            "lower_95": lower, "upper_95": upper, "valid_resamples": len(valid)}


def compare(clean, plain, adapted_draws):
    results = {}
    for name in METRICS:
        after = np.mean([draw[name] for draw in adapted_draws], axis=0)
        gain = after - plain[name]
        gap = clean[name] - plain[name]
        recovery = np.divide(gain, gap, out=np.full_like(gain, np.nan), where=gap > 0)
        results[name] = {"clean": interval(clean[name]), "plain": interval(plain[name]),
                         "adapted": interval(after), "gain": interval(gain), "recovery": interval(recovery)}
    return results


def run_site(site, records, clean_set, clean_weights, clean, model, args):
    pool, test = split_patients(records, args.max_score_patients)
    evaluation = EvaluationSet.from_records(test, model.normalise)
    matched = site in ("white_noise", "low_pass")
    if matched and (not np.array_equal(evaluation.patients, clean_set.patients)
                    or not np.array_equal(evaluation.labels, clean_set.labels)):
        raise ValueError("Clean and corrupted Yaseen cohorts do not match")
    weights = clean_weights if matched else resample_weights(len(evaluation.labels), np.random.default_rng([0, 1]))
    plain = binary_metrics(evaluation.labels, evaluation.scores(model.float_model), weights)
    rng = np.random.default_rng(0)
    conditions = [(size, None) for size in args.sizes]
    if args.imbalance_size:
        conditions += [(args.imbalance_size, fraction) for fraction in (0.1, 0.5, 0.9)]
    # Load each selected recording at most once per site. The test set stays fixed.
    loaded = {}
    output = []
    for count, fraction in conditions:
        draw_scores, clean_scores, details = [], [], []
        for _ in range(args.draws):
            selected = draw_indices(pool, count, fraction, rng).tolist()
            for index in selected:
                if index not in loaded:
                    loaded[index] = model.normalise(pool[index].load())
            batch = RecordingBatch.from_recordings([loaded[index] for index in selected])
            start = perf_counter()
            adapted, guard = adapt_bn(model.float_model, batch)
            seconds = perf_counter() - start
            if guard["collapse_guard_triggered"]:
                shifted, retained = plain, clean
            else:
                shifted = binary_metrics(evaluation.labels, evaluation.scores(adapted), weights)
                retained = binary_metrics(clean_set.labels, clean_set.scores(adapted), clean_weights)
            draw_scores.append(shifted)
            clean_scores.append(retained)
            if args.save_adapted is not None:
                with TemporaryDirectory(dir=args.save_adapted.parent) as staging:
                    saved = Path(staging) / args.save_adapted.name
                    adapted.save(saved)
                    if args.save_adapted.exists():
                        raise FileExistsError(args.save_adapted)
                    saved.rename(args.save_adapted)
            details.append({
                "adaptation_patients": len({pool[index].patient for index in selected}),
                "abnormal_recordings": sum(pool[index].label for index in selected),
                "adaptation_windows": int(batch.lengths.sum()),
                "seconds_to_adapt_with_guard": seconds,
                "shifted_after": {name: float(shifted[name][0]) for name in METRICS},
                **guard,
            })
        output.append({
            "adaptation_recordings": count,
            "target_abnormal_fraction": fraction,
            "draws": details,
            "shifted": compare(clean, plain, draw_scores),
            "clean_after_adaptation": {name: interval(np.mean([draw[name] for draw in clean_scores], axis=0))
                                        for name in METRICS},
        })
        print(f"{site}: completed {args.draws} draws with N={count}, abnormal fraction={fraction}", flush=True)
    return {"score_patients": len(evaluation.labels), "score_positive_patients": int(evaluation.labels.sum()),
            "score_recordings": len(test), "clean_comparison": "matched clips" if matched else "Yaseen reference cohort",
            "conditions": output}


def load_fold_zero_model(model_dir, report_dir):
    """Load the float CNN and its fold-0 metadata on CPU."""
    norm_bytes = (report_dir / f"quantization_{STEM}.json").read_bytes()
    norm = json.loads(norm_bytes)
    normalisation = (norm.get("norm_mean"), norm.get("norm_std"))
    if (norm.get("dataset") != "yaseen" or norm.get("holdout_fold") != 0
            or norm.get("feature") != "logmel" or norm.get("classes") != CLASSES
            or any(type(value) not in (int, float) for value in normalisation)
            or not np.isfinite(normalisation).all() or normalisation[1] <= 0):
        raise ValueError("Use a Yaseen log-mel report for holdout fold 0 with valid training normalisation and class order.")

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    model = FinalModel(model_dir, report_dir, load_int8=False)
    provenance = {
        "model_sha256": hashlib.sha256((model_dir / f"{STEM}_float.keras").read_bytes()).hexdigest(),
        "normalization_report_sha256": hashlib.sha256(norm_bytes).hexdigest(),
        "classes": CLASSES,
    }
    return model, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", choices=SITES, help="Repeat to select sites. Default: all four.")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--imbalance-size", type=int, default=32, help="Use 0 to skip one-sided draws.")
    parser.add_argument("--max-score-patients", type=int, help="Cap the fixed test panel for a pilot.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--reference-model", action="store_true", help="Mark the reference checkpoint as provisional.")
    checkpoint.add_argument("--team-model", action="store_true", help="Confirm that the model and report are the team's matched pair.")
    parser.add_argument("--save-adapted", type=Path, help="Save the selected .keras model from one draw.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if (min(args.sizes) < 1 or args.draws < 1 or args.imbalance_size < 0
            or (args.max_score_patients is not None and args.max_score_patients < 2)):
        parser.error("Use positive sizes/draws, a nonnegative imbalance size, and a test cap of at least two.")
    sites = tuple(dict.fromkeys(args.site or SITES))
    pilot = bool(set(sites) != set(SITES) or args.max_score_patients or tuple(args.sizes) != SIZES
                 or args.draws != 5 or args.imbalance_size != 32)
    official = REPO_ROOT / "reports/heart_tta.json"
    default_output = official if args.team_model else REPO_ROOT / "scratch/heart_tta_reference.json"
    if pilot:
        default_output = REPO_ROOT / "scratch/heart_tta_pilot.json"
    out = args.out or default_output
    if out.suffix.lower() != ".json":
        parser.error("Use a .json path for the report.")
    if (pilot or not args.team_model) and out.resolve() == official.resolve():
        parser.error("The official report requires the full sweep and --team-model.")
    if args.save_adapted is not None:
        if len(sites) != 1 or len(args.sizes) != 1 or args.draws != 1 or args.imbalance_size:
            parser.error("Saving a model requires one --site, one --sizes value, --draws 1, and --imbalance-size 0.")
        if args.save_adapted.suffix.lower() != ".keras" or args.save_adapted.exists():
            parser.error("Choose a new .keras output path.")
        args.save_adapted.parent.mkdir(parents=True, exist_ok=True)

    try:
        model, provenance = load_fold_zero_model(args.model_dir, args.report_dir)
    except ValueError as error:
        parser.error(str(error))
    yaseen = yaseen_recordings()
    clean_records = [Recording(patient, label, lambda windows=windows: windows)
                     for patient, label, _, windows in yaseen]
    _, clean_test = split_patients(clean_records, args.max_score_patients)
    clean_set = EvaluationSet.from_records(clean_test, model.normalise)
    clean_weights = resample_weights(len(clean_set.labels), np.random.default_rng(0))
    clean = binary_metrics(clean_set.labels, clean_set.scores(model.float_model), clean_weights)
    model_status = "unverified_checkpoint"
    if args.team_model:
        model_status = "confirmed_team_checkpoint"
    elif args.reference_model:
        model_status = "provisional_reference"
    result = {
        "model_status": model_status,
        **provenance,
        "pilot": pilot,
        "planned_sites": list(sites),
        "method": {
            "adaptation": "one forward pass to update BN statistics, with a recording-level 95% collapse guard",
            "split": "fixed stratified 25% holdout by patient; Yaseen uses fold 0 clip IDs",
            "score_panel_cap": args.max_score_patients,
            "scoring": "window-weighted mean probability per patient; refer if 1 - p(Normal) >= 0.5",
            "recovery": "(adapted - plain) / (clean - plain); null when the clean-to-shift gap is nonpositive",
            "intervals": "95% patient bootstrap, 1000 resamples, seed 0; metrics averaged across draws before intervals",
            "stress_batches": "class labels control sample composition; the adaptation model only receives windows",
        },
        "clean_yaseen": {"score_clips": len(clean_test), "metrics": {name: interval(clean[name]) for name in METRICS}},
        "sites": {},
    }
    if args.save_adapted is not None:
        result["saved_model_file"] = args.save_adapted.name
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.json")
    for site in sites:
        result["sites"][site] = run_site(site, site_recordings(site, yaseen), clean_set, clean_weights, clean, model, args)
        partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {site} results to {partial}", flush=True)
    partial.replace(out)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
