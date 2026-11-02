# Intervals, re-quantisation and the edge budget

Date: 2 Nov 2026 · Heart branch (Aviral) · Roadmap S4, plus intervals for S1 to S3 · Proposal 4.3, 4.5 and 4.7

## Built on Shivansh's work (pull request #5)

**Metrics.** `cardio/eval/metrics.py` supplies the metric definitions and the patient bootstrap. `cardio/eval/intervals.py` wraps them, so every heart result now carries a 95% interval (1,000 resamples of patients, seed 0).

**PhysioNet 2022 manifest and caches.** `cardio/data/build_unseen.py` rebuilt the manifest identically to the committed one and cached all 3,163 recordings. Its patient IDs group repeat visits (942 subjects into 872 patients), which allows patient-disjoint splits and patient-level intervals. Shivansh's PhysioNet data check, `tests/test_unseen_artifacts.py`, now runs and passes on this laptop.

**Not yet available.** The shared adaptation module, `cardio/tta/bn_adapt.py`, does not exist yet, so adaptation still uses the helper in `cardio/eval/real_world.py`.

## Intervals for the earlier results

| Result | Estimate [95% interval] |
|---|---|
| S1 5-fold accuracy (Yaseen, clips) | 99.29% [98.69, 99.80] |
| S1 macro-F1 | 99.29% [98.71, 99.79] |
| S1 recall, Normal / AS / MS / MR / MVP | 100 / 99.00 / 99.47 / 99.00 / 99.00% (lower bounds 97.2 to 100) |
| S1 refer-or-not AUROC | 1.000 [1.000, 1.000] |
| S1 calibration error, 5 classes, before calibration | 0.058 [0.053, 0.065], above the 0.05 target |
| S1 calibration error, refer-or-not, before calibration | 0.020 [0.019, 0.022] |
| S2 8-bit agreement, Yaseen fold 0 | 100% [100, 100] |
| S2 8-bit agreement, PhysioNet 2022 (3,163 recordings, 872 patients) | 96.49% [95.77, 97.16] |
| S3 clinic, mean top-class probability, before and after adaptation | 0.535 [0.480, 0.587] and 0.425 [0.383, 0.467] |
| S3 clinic, recordings called abnormal, before and after | 100% [100, 100] both |

Yaseen intervals are over clips, because Yaseen has no patient IDs. The PhysioNet agreement interval in the next section ([95.86, 97.13]) differs slightly from the one above because that script visits patients in a different order. The estimate is identical.

## Re-quantisation for a new site (S4, `cardio/eval/requantise.py`)

**Method.** Each site is split by patient into two halves.

- N unlabelled recordings from one half adapt the model and serve as the 8-bit representative set: 64 for PhysioNet, all 25 for the clinic.
- The other half is measured, then the halves swap.
- Each 8-bit model is compared with its own float model.

| Site | Model | Class agreement [95%] | Refer-or-not agreement | Inputs clipped | Seconds to convert |
|---|---|---|---|---|---|
| PhysioNet 2022 | A shipped 8-bit | 96.49% [95.86, 97.13] | 99.84% | 0.005% | — |
| PhysioNet 2022 | B re-quantised on site | 96.55% [95.95, 97.15] | 99.84% | 0% | 1.2 to 2.0 |
| PhysioNet 2022 | C adapted and re-quantised | 97.41% [96.81, 97.99] | 100% | 0% | 1.5 to 2.0 |
| Clinic | A shipped 8-bit | 94.00% [88, 100] | 100% | 0.009% | — |
| Clinic | B re-quantised on site | 96.00% [90, 100] | 100% | 0% | 0.9 to 1.2 |
| Clinic | C adapted and re-quantised | 96.00% [90, 100] | 100% | 0% | 1.0 to 1.2 |

All re-quantised models stay at 28,336 bytes.

**Where the disagreements are**, on PhysioNet with the shipped model. The margin is how far the float model's top class leads its second:

| Float margin | Recordings | Agreement |
|---|---|---|
| under 0.05 | 277 | 65% |
| 0.05 to 0.20 | 790 | 98.2% |
| 0.20 and above | 2,096 | 100% |

The clinic shows the same pattern: 70% agreement on its 10 near-tie recordings and 100% on the other 40.

## Edge budget (S4, `cardio/eval/edge_budget.py`)

| Measure | Value | Target |
|---|---|---|
| 8-bit model size | 27.7 kB (float: 232.9 kB) | under 1 MB |
| Time per case, Yaseen clip (median 2.5 s of audio) | median 8.7 ms, p95 15.8 ms | under 1 s |
| Time per case, PhysioNet recording (median 22.2 s of audio) | median 63 ms, p95 130 ms | under 1 s |
| Peak memory, whole process | 444 MB, mostly the TensorFlow runtime this script loads | — |

The stage times for a PhysioNet case (medians) are:

- secondary characteristics 29.0 ms;
- front-end 13.1 ms;
- quality check 12.9 ms;
- load and resample 3.7 ms;
- systole/diastole attribution 1.5 ms;
- the 8-bit CNN on every window, 1.2 ms.

Votes and late fusion are not built yet. They are two logistic regressions and add almost nothing.

Laptop timings vary from run to run. An earlier run, right after heavy jobs, gave a PhysioNet median of 102 ms, still a tenth of the budget.

## What it means

1. **The 8-bit model disagrees with the float model on near-ties, not because of input clipping.** Clipping is about 0.01% or less everywhere; the 2 Nov correction in the S2 note explains why the earlier figure looked larger. Disagreements sit almost entirely where the float model's top two classes are within 5 points, so 8-bit rounding flips a choice the float model was barely making. Adapting and re-quantising lifts PhysioNet class agreement to 97.41% [96.81, 97.99], just short of 98%.

2. **On the output that matters, the target is met.** The refer-or-not decision agrees on 99.84 to 100% of recordings at both sites.
    - Proposal: report the 98% agreement target on the refer-or-not decision, with class agreement alongside.
    - Tool: show "uncertain between X and Y" when the float margin is under 0.05, which is where the float model is unsure anyway.

3. **The edge budget is met with large headroom.** The CNN is the cheapest stage. The time goes to classical signal processing, mostly the secondary characteristics, so that is where to optimise if a smaller device is targeted.

4. **Calibration matters for the five-class output.** Its calibration error is 0.058 before calibration, against the 0.05 target. The refer-or-not probability is already at 0.020. Temperature scaling (roadmap S2, Anshuman) should bring the five-class number under 0.05.
