# The heart result for the dashboard, and the edge budget through it

Thu 12 Nov, Aviral, roadmap S5. Code: `cardio/predict.py`, `cardio/run_all.py`,
and a split in `cardio/model/features.py`. Tests: `tests/test_cardio_predict.py`,
`tests/test_cardio_run_all.py`. Results: `reports/heart_edge_budget.json`.

## What was built

- **`cardio/predict.py`.** `analyse_heart(recording, metadata)` is the one call the
  dashboard makes for a heart result. It takes a WAV at any sample rate, as a path,
  bytes or an upload, and never writes it to disk. It returns the `ModelOutput` and
  `FusedOutput` of `core/contracts.py`, the quality-check verdict, systole or
  diastole, and the time of every stage. A recording shorter than 1.5 s, one that
  fails the quality check, or a file that is not a WAV is withheld, with the reason.
- **`characteristics_from_audio(x)`** in `cardio/model/features.py`, so an upload is
  measured in memory; `characteristics(path)` now calls it. Checked against
  Anshuman's committed `reports/heart_features_yaseen.csv` (on `main`): all 1,000
  clips and 17 characteristics agree, the largest difference is 1.4 × 10⁻¹⁴, and no
  missing value moved.
- **`cardio/run_all.py`**, which runs every heart script in order and skips what
  this laptop has no code or data for. On this laptop 6 steps run; votes,
  calibration, late fusion and robustness are not in `cardio` yet, BMD-HS is not
  here (unseen datasets, adaptation), and the clinic steps need `--clinic-dir`.
  Running `cv_intervals` and `agreement` through it gave files identical to the
  committed ones.
- **The edge budget** now times each case through `analyse_heart`, so its time per
  case is the dashboard's own, and it measures the quality check on every public
  recording.

## Choices

- Systole or diastole comes from the measured systolic and diastolic energy, the
  rule `audio_descriptors` uses for `murmur_phase`. Occlusion on the model is not
  used: in v1 it pointed to systole for every class, Normal included.
- Until `heart_fusion()` and `reports/heart_fusion_params.json` exist, the CNN vote
  counts alone with weight 1 (the missing-vote rule), and T = 1, which the result
  marks as not calibrated. Both switch on by themselves when the files land.

## Results

Time per case through `cardio/predict.py`, with the CNN vote alone and T = 1:

| Data | Cases | Pass the quality check | Median length | Median per case | 95th percentile |
|---|---|---|---|---|---|
| Yaseen fold 0 | 197 | 197 | 2.5 s | 9.2 ms | 15.2 ms |
| PhysioNet 2022, 100 at random (seed 0) | 100 | 7 | 22.2 s | 76.4 ms | 141.1 ms |

Against the S4 run (medians 8.7 and 63.0 ms), the characteristics now include
heart rate and regularity (5.2 against 3.8 ms on Yaseen, 38.3 against 29.0 ms on
PhysioNet), and the occlusion step (1.2 to 1.5 ms) is gone. Peak memory is 484 MB
for the whole TensorFlow process (444 MB before). All three targets are still
met: 28,336 bytes, 100% agreement on fold 0, and well under 1 s per case.

The quality check on every public recording, with 95% intervals over patients:

| Data | Recordings (patients) | Pass | Energy above 1 kHz, median |
|---|---|---|---|
| Yaseen | 990 (990) | 99.4% [98.9, 99.8] | 0.003% |
| PhysioNet 2022 | 3,163 (872) | 8.8% [7.6, 10.0] | 1.6% |

The limit is 0.15% of the energy above 1 kHz. PhysioNet's recordings carry far
more energy between 1 and 2 kHz than Yaseen's, so **the dashboard withholds the
heart result for about 9 in 10 PhysioNet recordings.**

## What this means

- The dashboard check on Mon 16 Nov and the demo video need a PhysioNet recording
  that passes. Two that do: `33151_AV.wav` (murmur present; called MVP, p_h 0.78,
  systolic) and `2530_PV.wav` (murmur absent, yet called MS with p_h 0.97). The
  second is one more sign of the weak transfer to PhysioNet that the unseen-dataset
  run shows.
- Not yet measured: whether the check keeps the recordings the model gets right.
  Comparing refer-or-not AUROC on PhysioNet patients who pass and who fail would
  show whether it is a useful gate or only a strict one.

## Next

When `heart_fusion` and its parameter file are in `cardio`, run
`python -m cardio.eval.edge_budget` again, so votes and fusion are timed for real.
On Mon 16 Nov, after the last code change, run `python -m cardio.run_all --final`.
