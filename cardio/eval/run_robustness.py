"""S4 five-class heart accuracy on corrupted, held-out Yaseen fold-0 clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cardio.eval.run_tta import (
    Recording, corrupted_windows, draw_indices, interval, load_fold_zero_model,
    resample_weights, split_patients, yaseen_recordings,
)
from cardio.model.final_model import MODEL_DIR, REPO_ROOT
from cardio.tta.bn_adapt import RecordingBatch, adapt_bn


SNR_LEVELS = (20, 15, 10, 5, 0)
LOW_PASS_LEVELS = (600, 500, 400, 300, 200)
CONDITIONS = {"white_noise": SNR_LEVELS, "pink_noise": SNR_LEVELS, "low_pass": LOW_PASS_LEVELS}
ADAPTATION_RECORDINGS = 32
DRAWS = 5


def accuracy_comparison(plain_correct, adapted_correct, weights):
    """Pair each resample across all draws and both model variants."""
    counts = weights.sum(axis=1)
    plain = (weights @ plain_correct) / counts
    adapted = (weights @ np.mean(adapted_correct, axis=0)) / counts
    return {"plain": interval(plain), "adapted": interval(adapted), "gain": interval(adapted - plain)}


def score_condition(site, severity, model, inputs, test_ids, draw_ids, labels, weights):
    # Cache exactly the audio needed for this condition. A clip uses the same
    # noise seed at every SNR so each level changes the noise magnitude alone.
    used = set(test_ids).union(*draw_ids)
    windows = {
        clip: model.normalise(corrupted_windows(path, site, seed, severity=severity))
        for clip, (path, seed) in inputs.items() if clip in used
    }
    test = RecordingBatch.from_recordings([windows[clip] for clip in test_ids])
    plain_correct = test.predict(model.float_model).argmax(axis=1) == labels
    after, draws = [], []
    for clips in draw_ids:
        batch = RecordingBatch.from_recordings([windows[clip] for clip in clips])
        adapted, guard = adapt_bn(model.float_model, batch)
        if guard["collapse_guard_triggered"]:
            correct = plain_correct
        else:
            correct = test.predict(adapted).argmax(axis=1) == labels
        after.append(correct)
        draws.append({"correct_clips": int(correct.sum()), **guard})
    return {
        "corruption": site,
        "cutoff_hz" if site == "low_pass" else "snr_db": severity,
        "plain_correct_clips": int(plain_correct.sum()),
        "accuracy": accuracy_comparison(plain_correct, after, weights),
        "draws": draws,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--team-model", action="store_true", help="Confirm the team's matched model and report.")
    checkpoint.add_argument("--reference-model", action="store_true", help="Mark a provisional reference run.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    official = REPO_ROOT / "reports/heart_robustness.json"
    out = args.out or (official if args.team_model else REPO_ROOT / "scratch/heart_robustness_reference.json")
    if out.suffix.lower() != ".json" or (out.resolve() == official.resolve() and not args.team_model):
        parser.error("Use a .json output path. The official report requires --team-model.")
    try:
        model, provenance = load_fold_zero_model(args.model_dir, args.report_dir)
    except ValueError as error:
        parser.error(str(error))

    rows = yaseen_recordings()
    records = [Recording(clip, label, lambda windows=windows: windows) for clip, label, _, windows in rows]
    pool, test = split_patients(records)
    test_ids = [row.patient for row in test]
    rng = np.random.default_rng(0)
    draw_ids = [[pool[index].patient for index in draw_indices(pool, ADAPTATION_RECORDINGS, None, rng)]
                for _ in range(DRAWS)]
    with (REPO_ROOT / "data/manifests/yaseen_manifest.csv").open(encoding="utf-8", newline="") as stream:
        classes = {row["patient_id"]: int(row["label"]) for row in csv.DictReader(stream) if row["fold"] == "0"}
    labels = np.array([classes[clip] for clip in test_ids])
    weights = resample_weights(len(test), np.random.default_rng(0))
    clean = RecordingBatch.from_recordings([model.normalise(row.load()) for row in test])
    clean_correct = clean.predict(model.float_model).argmax(axis=1) == labels
    inputs = {clip: (path, i) for i, (clip, _, path, _) in enumerate(rows)}

    status = "unverified_checkpoint"
    if args.team_model:
        status = "confirmed_team_checkpoint"
    elif args.reference_model:
        status = "provisional_reference"
    result = {
        "model_status": status,
        **provenance,
        "fold": 0,
        "eligible_clips": len(records),
        "scored_clips": len(test),
        "adaptation_pool_clips": len(pool),
        "adaptation_recordings_per_draw": ADAPTATION_RECORDINGS,
        "draws_per_condition": DRAWS,
        "method": {
            "split": "fixed stratified quarter of usable fold-0 clips scored; remaining clips used for adaptation",
            "accuracy": "five-class top prediction after averaging each clip's window probabilities",
            "intervals": "95% paired clip bootstrap, 1000 resamples, seed 0; adapted accuracy averages five draws",
            "adaptation": "the same five disjoint-from-scoring batches at every condition; collapse guard can keep original statistics",
        },
        "clean_accuracy": interval((weights @ clean_correct) / weights.sum(axis=1)),
        "conditions": [],
    }
    for site, levels in CONDITIONS.items():
        for severity in levels:
            row = score_condition(site, severity, model, inputs, test_ids, draw_ids, labels, weights)
            result["conditions"].append(row)
            print(f"{site} {severity}: {row['accuracy']['plain']['estimate']:.1%} plain, "
                  f"{row['accuracy']['adapted']['estimate']:.1%} adapted", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.json")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(out)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
