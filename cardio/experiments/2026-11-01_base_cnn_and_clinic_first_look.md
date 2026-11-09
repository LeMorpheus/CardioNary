# Base CNN, 8-bit model, and a first look at the clinic recordings

Date: 1 Nov 2026 · Roadmap stages S0, S1 and the first S2 item for the heart branch

## What was run

| Step | Command | Output |
|---|---|---|
| Yaseen manifest check | `python -m core.build_cardio_manifest` | No change against the committed manifest, so the folds are the committed ones |
| Spectrogram cache | `python -m core.preprocess --dataset yaseen` | 1,044 windows from 990 clips; the 10 clips under 1.5 s give no 3-second window |
| 5-fold cross-validation | `python -m cardio.model.cv_oof` | `reports/heart_cv_yaseen.json`, `reports/heart_oof_yaseen.csv` |
| Final model, folds 1 to 4 | `python -m core.quantize --dataset yaseen --holdout-fold 0` | `models/` (not in Git), `reports/quantization_chiron_cardio_v1.json` |
| 8-bit against float | `python -m cardio.eval.agreement` | `reports/heart_agreement.json` |
| Clinic recordings | `python -m cardio.eval.real_world --data-dir <Real_World_Data>` | `reports/heart_real_world.json` (aggregates only) |

## Results

**Base CNN (target ≥ 95%, proposal 4.1).** 5-fold clip accuracy 99.29% ± 0.58%, macro-F1 99.30% ± 0.57%. Recall per class: Normal 100%, AS 99%, MS 99.5%, MR 99%, MVP 99%. Refer-or-not AUROC (p(abnormal) = 1 − p(Normal)) is 1.000 on the out-of-fold probabilities. The earlier project's run of the same settings gave 98.9% ± 1.1%; the difference is within that spread and comes from TensorFlow's non-deterministic CPU kernels.

Yaseen has no patient IDs, so clips from one person may sit in different folds. Treat these numbers as an upper bound; the unseen datasets are the honest test.

**8-bit model (proposal 4.5).** 27.7 kB. On fold 0, which it never trained on, it agrees with the float model on 100% of windows and clips, and both reach 98.48% clip accuracy.

On PhysioNet 2022 (all 3,163 public recordings, 46,654 windows; run on 2 Nov with `--wav-dir data/raw/circor/training_data --name physionet_2022`), agreement drops to 96.49% of recordings and 96.92% of windows, below the proposal's 98%.

The likely cause is the model's 8-bit input range (scale 0.0152, zero point −38). It was fitted to Yaseen windows and covers about −1.4 to +2.5 in normalised units, so inputs from a new device that fall outside it are clipped. Re-quantising after adaptation (S4), with the new site's own windows as the representative set, is the planned fix. Measure agreement again after it.

**Correction, 2 Nov.** The S4 measurement does not support the clipping explanation. The values that seemed to fall below the 8-bit range sit at the log-mel floor, which quantises exactly. Values clipped by more than half a step are almost absent: 0.00% on Yaseen fold 0 and 0.01% on the clinic recordings. Re-quantising on site data removes the clipping that does exist without restoring agreement. See `2026-11-02_requantisation_and_edge_budget.md`.

The downloaded copy matches PhysioNet's checksums for every recording, header, segmentation file and `training_data.csv`. The 942 patient `.txt` files do not match, although their content is well-formed. No result here uses them.

**Clinic recordings (50 recordings, 10 patients, no labels).**

| Measure | Value |
|---|---|
| Pass the quality check | 9 of 50 |
| Called abnormal by the model | 50 of 50 (MVP 31, MS 16, MR 3) |
| Patients whose 5 positions agree | 10 of 10, trivially, since every call is abnormal |
| Heart-cycle segmentation succeeds | 1 of 50 (Yaseen fold 0: 98.5%) |
| 8-bit agrees with float | 94% (Yaseen fold 0: 100%) |
| Mean normalised model input | 0.62 (Yaseen fold 0: 0.01), largest shift in the lowest mel bands |

## What it means

Ten clinic patients being abnormal at every position is not plausible. This is a device shift, not a finding about the patients. The MEMS recordings sit 0.6 standard deviations away from Yaseen in the model's own input units, mostly at the lowest frequencies, and they carry handling and contact noise above 800 Hz, which is what the quality check flags. The envelope peak-picking that segments heart cycles also breaks on them.

This is the case test-time adaptation (S3) exists for. Re-estimating the batch-normalisation statistics on clinic recordings re-centres exactly this kind of mean shift, so the S3 measures to watch are:

- the share of positions called abnormal;
- whether the 5 positions of a patient agree;
- the 8-bit agreement.

Segmentation will not improve through adaptation, because it does not use the model. It probably needs a lower peak threshold or a stronger band-pass for this microphone; that belongs with the secondary characteristics.
