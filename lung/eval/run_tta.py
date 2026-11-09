"""S3: compare BN test-time adaptation on RSNA and controlled image shifts.

Roadmap protocol:
- N in 8, 16, 32, 64, 128; five deterministic random draws by default.
- Adapt only from unlabeled images; never score an image from an adaptation patient.
- For controlled Kermany shifts, compare clean, shifted/plain, shifted/adapted and
  recovery = (adapted - plain) / (clean - plain).
- For RSNA (a new cohort, not a corruption), report adapted-minus-plain instead;
  recovery is undefined because clean and plain are the same uncorrupted RSNA images.
- Also exercise N=32 adaptation batches with 10%, 50%, and 90% pneumonia.

Usage:
    python -m lung.eval.run_tta
    python -m lung.eval.run_tta --sizes 8 16 32 64 128 --draws 5
Writes reports/xray_tta.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lung.eval.common import (
    REPORT_DIR,
    load_float_model,
    load_kermany,
    load_norm,
    load_rsna,
    normalise,
    predict_float,
)
from lung.eval.corrupt import SEVERITIES, apply_corruption
from lung.eval.metrics import evaluate_predictions
from lung.tta.bn_adapt import apply_bn_adaptation


DEFAULT_SIZES = (8, 16, 32, 64, 128)
CORRUPTIONS = ("blur", "noise", "contrast", "jpeg")


def _metrics(patient_ids, labels, probabilities) -> dict:
    return evaluate_predictions(
        np.asarray(patient_ids).astype(str),
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
        n_bootstrap=1000,
        seed=0,
    )


def _recovery(clean: float, plain: float, adapted: float) -> float | None:
    denominator = clean - plain
    if abs(denominator) < 1e-8:
        return None
    return float((adapted - plain) / denominator)


def _choose_adaptation_rows(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
    pneumonia_share: float | None = None,
) -> np.ndarray:
    """Select at most one image per patient so the requested N is patient-distinct."""
    patient_ids = np.asarray(patient_ids).astype(str)
    labels = np.asarray(labels, dtype=int)
    first_row: dict[str, int] = {}
    for row, patient_id in enumerate(patient_ids):
        first_row.setdefault(patient_id, row)

    by_class: dict[int, list[int]] = {0: [], 1: []}
    for row in first_row.values():
        by_class[int(labels[row])].append(row)

    if pneumonia_share is None:
        if n > len(first_row):
            raise ValueError(f"Requested N={n}, but only {len(first_row)} distinct patients are available")
        return np.asarray(rng.choice(list(first_row.values()), size=n, replace=False), dtype=int)

    n_positive = int(round(n * pneumonia_share))
    n_negative = n - n_positive
    if n_positive > len(by_class[1]) or n_negative > len(by_class[0]):
        raise ValueError(
            f"Cannot draw N={n} with {pneumonia_share:.0%} pneumonia: "
            f"available positives={len(by_class[1])}, negatives={len(by_class[0])}"
        )
    positives = rng.choice(by_class[1], size=n_positive, replace=False) if n_positive else []
    negatives = rng.choice(by_class[0], size=n_negative, replace=False) if n_negative else []
    return np.asarray(rng.permutation(np.concatenate([positives, negatives])), dtype=int)


def _evaluate_draw(
    *,
    model,
    X_clean: np.ndarray,
    X_shifted: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    adaptation_rows: np.ndarray,
    mean: float,
    std: float,
    is_corruption: bool,
) -> dict:
    adapt_patient_ids = np.unique(np.asarray(patient_ids).astype(str)[adaptation_rows])
    evaluation_mask = ~np.isin(np.asarray(patient_ids).astype(str), adapt_patient_ids)
    if not evaluation_mask.any():
        raise ValueError("No evaluation patients remain after excluding adaptation patients")

    eval_ids = np.asarray(patient_ids).astype(str)[evaluation_mask]
    eval_y = np.asarray(labels, dtype=int)[evaluation_mask]
    clean_probs = predict_float(model, X_clean[evaluation_mask], mean, std)
    plain_probs = predict_float(model, X_shifted[evaluation_mask], mean, std)

    started = time.perf_counter()
    adapted_model, collapse_guard = apply_bn_adaptation(
        model, normalise(X_shifted[adaptation_rows], mean, std)
    )
    adaptation_seconds = time.perf_counter() - started
    adapted_probs = predict_float(adapted_model, X_shifted[evaluation_mask], mean, std)

    clean = _metrics(eval_ids, eval_y, clean_probs)
    plain = _metrics(eval_ids, eval_y, plain_probs)
    adapted = _metrics(eval_ids, eval_y, adapted_probs)

    recovery = {}
    for metric in ("accuracy", "auroc"):
        recovery[metric] = (
            _recovery(clean[metric], plain[metric], adapted[metric])
            if is_corruption else None
        )

    return {
        "n_adaptation_images": int(len(adaptation_rows)),
        "n_adaptation_patients": int(len(adapt_patient_ids)),
        "n_scored_images": int(evaluation_mask.sum()),
        "n_scored_patients": int(len(np.unique(eval_ids))),
        "adaptation_seconds": float(adaptation_seconds),
        "collapse_guard_triggered": bool(collapse_guard),
        "clean": clean,
        "shifted_plain": plain,
        "shifted_adapted": adapted,
        "adapted_minus_plain": {
            metric: float(adapted[metric] - plain[metric])
            for metric in ("accuracy", "auroc")
        },
        "recovery": recovery,
    }


def _dataset_result(
    name: str,
    X_clean: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    *,
    model,
    mean: float,
    std: float,
    sizes: tuple[int, ...],
    draws: int,
    seed: int,
    corruption: str | None = None,
    severity: int = 3,
    class_mix: bool = False,
) -> dict:
    if corruption is None:
        X_shifted = X_clean
    else:
        X_shifted = np.stack(
            [
                apply_corruption(X_clean[i], corruption, severity, seed=seed + i)
                for i in range(len(X_clean))
            ]
        ).astype(np.float32)

    size_results: dict[str, list[dict]] = {}
    for n in sizes:
        if n > len(np.unique(patient_ids)):
            continue
        size_results[str(n)] = []
        for draw in range(draws):
            rng = np.random.default_rng(seed + 1009 * n + draw)
            rows = _choose_adaptation_rows(patient_ids, labels, n, rng)
            result = _evaluate_draw(
                model=model,
                X_clean=X_clean,
                X_shifted=X_shifted,
                labels=labels,
                patient_ids=patient_ids,
                adaptation_rows=rows,
                mean=mean,
                std=std,
                is_corruption=corruption is not None,
            )
            result.update({"draw": draw, "seed": seed + 1009 * n + draw})
            size_results[str(n)].append(result)

    mixes = {}
    if class_mix and 32 <= len(np.unique(patient_ids)):
        mixes = {}
        for share in (0.10, 0.50, 0.90):
            mixes[f"{share:.0%}"] = []
            for draw in range(draws):
                rng = np.random.default_rng(seed + 500_000 + int(share * 1000) + draw)
                rows = _choose_adaptation_rows(
                    patient_ids, labels, 32, rng, pneumonia_share=share
                )
                result = _evaluate_draw(
                    model=model,
                    X_clean=X_clean,
                    X_shifted=X_shifted,
                    labels=labels,
                    patient_ids=patient_ids,
                    adaptation_rows=rows,
                    mean=mean,
                    std=std,
                    is_corruption=corruption is not None,
                )
                result.update({
                    "draw": draw,
                    "seed": seed + 500_000 + int(share * 1000) + draw,
                    "requested_pneumonia_share": share,
                    "observed_pneumonia_share": float(labels[rows].mean()),
                })
                mixes[f"{share:.0%}"].append(result)

    return {
        "name": name,
        "corruption": corruption,
        "severity": severity if corruption is not None else None,
        "n_images": int(len(X_clean)),
        "n_patients": int(len(np.unique(patient_ids))),
        "by_adaptation_size": size_results,
        "one_sided_n32": mixes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "xray_tta.json")
    args = parser.parse_args()

    if args.draws < 1 or any(n < 1 for n in args.sizes):
        parser.error("--draws and every adaptation size must be positive")

    mean, std = load_norm()
    model = load_float_model()
    results = []

    # Unseen RSNA cohort. For this non-corruption shift, recovery is undefined;
    # use delta from the unadapted RSNA baseline.
    rsna_manifest, X_rsna = load_rsna()
    results.append(
        _dataset_result(
            "rsna",
            X_rsna,
            rsna_manifest.label.to_numpy(int),
            rsna_manifest.patient_id.astype(str).to_numpy(),
            model=model,
            mean=mean,
            std=std,
            sizes=tuple(args.sizes),
            draws=args.draws,
            seed=args.seed,
            class_mix=True,
        )
    )

    # Controlled shifts on Kermany fold 0, excluded from the final CNN training.
    X_all, y_all, patient_all, fold_all = load_kermany()
    held_out = np.asarray(fold_all) == 0
    X_clean = np.asarray(X_all[held_out], dtype=np.float32)
    labels = np.asarray(y_all[held_out], dtype=int)
    patient_ids = np.asarray(patient_all[held_out]).astype(str)
    for corruption in CORRUPTIONS:
        results.append(
            _dataset_result(
                f"kermany_{corruption}_severity3",
                X_clean,
                labels,
                patient_ids,
                model=model,
                mean=mean,
                std=std,
                sizes=tuple(args.sizes),
                draws=args.draws,
                seed=args.seed,
                corruption=corruption,
                severity=3,
                class_mix=True,
            )
        )

    output = {
        "protocol": {
            "method": "batch-normalisation statistics adaptation",
            "model": "float MobileNetV2 lung CNN",
            "normalisation": {"mean": mean, "std": std, "source": "Kermany training report"},
            "sizes": list(args.sizes),
            "draws_per_size": args.draws,
            "seed": args.seed,
            "corruption_severity": 3,
            "bootstrap_resamples": 1000,
            "adaptation_labels_used": False,
            "patient_overlap_between_adaptation_and_scoring": False,
            "rsna_recovery": "null; RSNA has no paired synthetic clean/shifted condition",
        },
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

