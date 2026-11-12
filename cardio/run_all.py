"""Regenerate every heart result file, in order (roadmap S5).

    python -m cardio.run_all --list           # the plan, and what this laptop would skip
    python -m cardio.run_all                  # every step whose code and data are here
    python -m cardio.run_all --clinic-dir <path to Real_World_Data>   # the clinic steps too
    python -m cardio.run_all --only agreement edge_budget
    python -m cardio.run_all --from requantise_physionet
    python -m cardio.run_all --final          # Mon 16 Nov: missing code is an error, not a skip

Each step runs one of the team's scripts as `python -m <module>` in its own process,
from the repository root, in the order the results depend on each other. The run
stops at the first failure. A step is skipped, and listed at the end, when

  * its code is not in this branch yet (with --final this is an error), or
  * its data are not on this laptop. BMD-HS is on one laptop and the clinic
    recordings on another, so no single laptop regenerates everything: each runs
    its part, and the files a laptop skipped come from the other one.

The clinic steps run only with --clinic-dir, on the laptop that holds the data, and
write totals and averages only. After a clinic run, add a line to the status
section of the Real_World_Data README, as its rules require.

The base CNN is not retrained here: its 5-fold run is cached in
reports/heart_oof_yaseen.csv, and the final model in models/. Retrain them with
`python -m cardio.model.cv_oof` and `python -m core.quantize --dataset yaseen
--holdout-fold 0` only on purpose.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("models/chiron_cardio_v1_float.keras", "models/chiron_cardio_v1_int8.tflite",
               "reports/quantization_chiron_cardio_v1.json")
CLINIC = "<clinic>"                      # stands for --clinic-dir in a step's arguments


@dataclass(frozen=True)
class Step:
    name: str
    module: str
    args: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()          # files or folders that must be on this laptop
    model: bool = True                   # needs the final model files
    clinic: bool = False                 # runs only with --clinic-dir

    @property
    def code(self) -> Path:
        return REPO_ROOT / (self.module.replace(".", "/") + ".py")


STEPS = (
    Step("cv_intervals", "cardio.model.cv_oof", ("--from-oof",),
         ("reports/heart_cv_yaseen.json",), ("reports/heart_oof_yaseen.csv",), model=False),
    Step("features_yaseen", "cardio.model.features", ("--dataset", "yaseen"),
         ("reports/heart_features_yaseen.csv",), ("data/raw/yaseen",), model=False),
    Step("features_physionet", "cardio.model.features", ("--dataset", "physionet"),
         ("reports/heart_features_physionet.csv",), ("data/raw/circor",), model=False),
    Step("votes", "cardio.fusion.votes", (),
         ("reports/heart_plr_yaseen.csv", "reports/heart_pm_physionet.csv"),
         ("reports/heart_features_yaseen.csv", "data/manifests/circor_manifest.csv"), model=False),
    Step("calibration", "cardio.fusion.calibrate", (),
         ("reports/heart_calibration.json",), ("reports/heart_oof_yaseen.csv",), model=False),
    Step("unseen", "cardio.eval.run_unseen",
         ("--model", MODEL_FILES[0], "--normalization-report", MODEL_FILES[2], "--out", "reports/heart_unseen.json"),
         ("reports/heart_unseen.json",), ("data/raw/bmdhs", "data/cache/unseen")),
    Step("late_fusion", "cardio.fusion.late_fusion", (),
         ("reports/heart_fusion.json", "reports/heart_fusion_params.json"), ("data/cache/unseen",)),
    Step("adaptation", "cardio.eval.run_tta", ("--team-model",),
         ("reports/heart_tta.json",), ("data/raw/bmdhs", "data/cache/unseen")),
    Step("robustness", "cardio.eval.run_robustness", ("--team-model",),
         ("reports/heart_robustness.json",), ("data/raw/yaseen",)),
    Step("agreement", "cardio.eval.agreement",
         ("--manifest", "data/manifests/circor_manifest.csv", "--name", "physionet_2022"),
         ("reports/heart_agreement.json",), ("data/cache/yaseen_logmel.npz", "data/cache/unseen")),
    Step("requantise_physionet", "cardio.eval.requantise", ("--site", "physionet_2022"),
         ("reports/heart_requantisation.json",), ("data/cache/unseen",)),
    Step("edge_budget", "cardio.eval.edge_budget", (),
         ("reports/heart_edge_budget.json",), ("data/raw/yaseen", "data/raw/circor")),
    Step("clinic", "cardio.eval.real_world", ("--data-dir", CLINIC, "--adapt"),
         ("reports/heart_real_world.json",), clinic=True),
    Step("requantise_clinic", "cardio.eval.requantise", ("--site", "clinic", "--data-dir", CLINIC),
         ("reports/heart_requantisation.json",), clinic=True),
)
NOT_YET = "code not in this branch yet"


def skip_reason(step: Step, clinic_dir: Path | None, produced=frozenset()) -> str | None:
    """None when the step can run on this laptop, otherwise why it is skipped.

    A file that an earlier step of the same run writes counts as present.
    """
    if not step.code.exists():
        return NOT_YET
    if step.clinic and clinic_dir is None:
        return "clinic step: pass --clinic-dir on the laptop that holds Real_World_Data"
    if step.model and not all((REPO_ROOT / f).exists() for f in MODEL_FILES):
        return "model files missing: python -m core.quantize --dataset yaseen --holdout-fold 0"
    missing = [n for n in step.needs if n not in produced and not (REPO_ROOT / n).exists()]
    if missing:
        return "data not on this laptop: " + ", ".join(missing)
    return None


def plan(clinic_dir: Path | None = None, only=None, start: str | None = None) -> list[tuple[Step, str | None]]:
    """The steps to consider, in order, each with None (it runs) or the reason it is skipped."""
    names = [s.name for s in STEPS]
    for name in list(only or []) + ([start] if start else []):
        if name not in names:
            raise ValueError(f"unknown step {name!r}; the steps are: {', '.join(names)}")
    steps = list(STEPS[names.index(start):] if start else STEPS)
    if only:
        steps = [s for s in steps if s.name in only]
    out, produced = [], set()
    for s in steps:
        why = skip_reason(s, clinic_dir, produced)
        if why is None:
            produced.update(s.writes)
        out.append((s, why))
    return out


def command(step: Step, clinic_dir: Path | None) -> list[str]:
    args = [str(clinic_dir) if a == CLINIC else a for a in step.args]
    return [sys.executable, "-m", step.module, *args]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate every heart result file, in order.")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    ap.add_argument("--clinic-dir", type=Path, help="the Real_World_Data folder (only on the laptop that holds it)")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    ap.add_argument("--from", dest="start", metavar="STEP", help="start at this step, for example after a failure")
    ap.add_argument("--final", action="store_true", help="treat a step whose code is missing as an error")
    args = ap.parse_args(argv)

    if args.clinic_dir is not None and not args.clinic_dir.is_dir():
        ap.error(f"--clinic-dir {args.clinic_dir} is not a folder")
    try:
        steps = plan(args.clinic_dir, args.only, args.start)
    except ValueError as exc:
        ap.error(str(exc))

    width = max(len(s.name) for s, _ in steps)
    if args.list:
        for s, why in steps:
            print(f"{s.name:<{width}}  {'runs' if why is None else 'skipped: ' + why}")
            print(f"{'':<{width}}  python -m {s.module} {' '.join(s.args)}  ->  {', '.join(s.writes)}")
        return 0

    missing_code = [s.name for s, why in steps if why == NOT_YET]
    if args.final and missing_code:
        print("--final: these steps have no code in this branch yet: " + ", ".join(missing_code))
        return 2

    done = []
    for s, why in steps:
        if why is not None:
            continue
        cmd = command(s, args.clinic_dir)
        print(f"\n=== {s.name}: python -m {' '.join(cmd[2:])}", flush=True)
        t0 = time.perf_counter()
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
            print(f"\n{s.name} failed; stopping. Fix it, then resume with: python -m cardio.run_all --from {s.name}")
            return 1
        done.append((s.name, time.perf_counter() - t0))

    print("\n=== Summary")
    for name, seconds in done:
        print(f"{name:<{width}}  ran in {seconds:.0f} s")
    for s, why in steps:
        if why is not None:
            print(f"{s.name:<{width}}  skipped: {why}")
    regenerated = {w for s, why in steps if why is None for w in s.writes}
    left = sorted({w for s, why in steps if why is not None for w in s.writes} - regenerated)
    if left:
        print("Not regenerated on this laptop: " + ", ".join(left))
    if args.clinic_dir is not None and any(s.clinic for s, why in steps if why is None):
        print("Clinic steps ran: add a line to the status section of the Real_World_Data README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
