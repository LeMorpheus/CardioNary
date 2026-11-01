# A1 — Data Pipeline Results and Findings

**Document status:** Current as of 2026-08-24.
**Scope:** Model development, corpus evaluation, and the evidence underlying the
pulmonary modality decision.

**Date:** 2026-08-13
**Status:** pipeline built and validated; cardio model exceeds target; **lung corpus does not support the target**

---

## 1. What was built

| Component | File | Status |
|---|---|---|
| Dataset downloader (checksummed) | `core/download_datasets.py` | KAUH + Yaseen |
| KAUH manifest + patient-disjoint folds | `core/build_manifest.py` | Yes |
| Yaseen manifest + stratified folds | `core/build_cardio_manifest.py` | Yes |
| DSP front-end (single source of truth) | `core/features.py` | log-mel + MFCC |
| Windowed cache + leakage assertions | `core/preprocess.py` | 6 rules enforced |
| CV trainer (window + patient level) | `core/train.py` | Yes |
| Separability diagnostic | `core/diagnose_separability.py` | Yes |
| Front-end ablation | `core/ablate_frontend.py` | Yes |

---

## 2. Headline results

| Model | Corpus | Classes | Protocol | Accuracy |
|---|---|---|---|---|
| **Cardio** | Yaseen 2018 | 5 | stratified 5-fold, clip-disjoint | **98.9 % ± 1.1 %** |
| **Pulmo** | KAUH | 4 | **patient-disjoint** 5-fold | **44.6 % ± 11.6 %** |

Both used the **same code**, the same ~13.5 k-parameter CNN, the same training recipe.

### 2a. Cardio node — COMPLETE and deployment-ready

| Metric | Value |
|---|---|
| Cross-validated accuracy | **98.9 % ± 1.1 %** (5-fold) |
| Held-out fold, float32 | **98.06 %** |
| Held-out fold, **INT8** | **98.06 %** |
| **float → int8 delta** | **+0.00 pp** |
| **float/int8 prediction agreement** | **100.00 %** |
| Parameters | 13,509 |
| **`.tflite` size** | **27.7 kB** |
| Input | `[1, 40, 192, 1]` int8, scale 0.015220, zp −38 |
| Output | `[1, 1, 1, 5]` int8, scale 0.003906, zp −128 |
| Representative dataset | 300 training-fold windows, class-balanced, energy-spanned |

**Operator set: `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `SOFTMAX`.**
All four of the first three have MVP-accelerated kernels; `SOFTMAX` runs once on a
5-element vector. **Zero unsupported or dynamic-shape operators.**

> The first conversion attempt emitted `SHAPE`, `STRIDED_SLICE` and `PACK` — Keras computes
> the `Reshape` target at runtime where `Reshape -> Dense` is used. Those have no MVP kernels
> and dynamic shapes are hostile to a static TFLM arena. Replacing the head with a **1×1
> `Conv2D`** (numerically identical to the Dense layer) removed all three and shrank the model
> from 28.6 kB to 27.7 kB. Worth knowing before writing the lung model's head.

**Golden vectors:** 25 clips (5 per class) from held-out fold 0,
**25/25 correct offline**, 1,172 kB of flash as raw int16 PCM.
Emitted as `vectors/chiron_cardio_vectors.{c,h}` plus a provenance manifest recording
source file, patient id, expected label, offline int8 prediction and PCM md5 per vector.

Artefacts: `models/chiron_cardio_v1_int8.tflite`, `models/chiron_cardio_v1_float.keras`,
the quantisation record produced by the training pipeline.

### Cardio detail (target was ≥ 98 %, gate ≥ 90 %)
```
              precision    recall  f1-score   support
      Normal      1.000     1.000     1.000       200
          AS      1.000     0.990     0.995       200
          MS      0.995     0.979     0.987       190
          MR      0.957     1.000     0.978       200
         MVP      0.995     0.975     0.985       200
    accuracy                          0.989       990
```
Perfect separation of Normal; the only meaningful confusion is MVP/MS → MR, which is
clinically sensible (all three are mitral-valve murmurs).

---

## 2a-ter. Pulmonary node resolved — pneumonia from chest radiography 

**Kermany paediatric chest X-ray corpus** (Mendeley `10.17632/rscbjbr9sj`, v2,
`ChestXRay2017.zip`, 1,236 MB). Binary **Normal vs Pneumonia**.

### Final model: ImageNet-pretrained MobileNetV2 (α=0.35) + augmentation

Two levers, both standard in the published chest X-ray literature, applied **without loosening
the evaluation protocol**:

| | v1 `dwsep` (scratch) | **v2 MobileNetV2 + aug** |
|---|---|---|
| float, image level | 0.9031 | **0.9661** |
| float, patient level | — | **0.9803** |
| **INT8, image level** | 0.9015 | **0.9637** |
| **INT8, PATIENT level** | — | **0.9785** |
| float → int8 delta | +0.16 pp | **−0.18 pp** (patient) |
| float/int8 agreement | 99.19 % | **99.27 %** |
| Parameters / `.tflite` | 13,122 / 27.2 kB | 412,773 / **610.4 kB** |
| Golden vectors correct | 19/24 | **22/24** |

*(fold 0; full 5-fold CV in `reports/kermany_mnet_cv.log`)*

**+6.2 pp image-level** from transfer learning plus augmentation. Per-class INT8 recall is now
**Normal 0.960 / Pneumonia 0.965** — the scratch model's weak Normal recall (0.868) is fixed
and the classes are balanced rather than traded off.

Two deliberate choices inside the augmentation:
- **No horizontal flip.** The heart sits on the left; mirroring a chest film manufactures
  dextrocardia and teaches the model that laterality carries no information.
- Rotation ±10°, zoom 0.90–1.10, translation ±8 %, exposure ±10 % — all plausible variation
  between real films. (Note this is the *opposite* call from the lung-audio model, where
  Sen et al. found augmentation degraded results.)

### Base corpus figures

| Metric | Value |
|---|---|
| Images / subjects | 5,856 / 2,790 |
| Protocol | **patient-disjoint** 5-fold |
| **Hurdle** (majority, subject level) | **0.600** |
| v1 patient-level accuracy | 0.9244 ± 0.0347 (+32.4 pp) |
| v1 patient-level macro-F1 | 0.9190 ± 0.0406 |
| Input | `[1, 96, 96, 1]` int8, scale 0.022394, zp 18 |
| Operators | `ADD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `SOFTMAX` |

`ADD` (MobileNetV2's residual connections) **has an MVP kernel**, so the network is still fully
accelerated with no CPU fallback.

### Memory fit on BRD2709A

| | Value |
|---|---|
| Model in flash | 610.4 kB |
| BLE stack + app + TFLM runtime | ~470 kB |
| Golden vectors (24 × 96×96 uint8) | 216 kB |
| **Flash total** | **~1.3 MB / 3200 kB** |
| Largest true activation | 108 kB (`1×48×48×48`) |
| **Estimated tensor arena** | **~145–200 kB / 512 kB** |

> The 140 kB tensor visible in a `get_tensor_details()` dump is a **weight constant**, which
> lives in flash, not the arena. Only activations size the arena. Confirm the real figure with
> `arena_used_bytes()` on device before shipping.

Pooled patient-level confusion (5 folds):

| | pred Normal | pred Pneumonia |
|---|---|---|
| **Normal** (1116) | 969 | 147 |
| **Pneumonia** (1674) | 64 | **1610** |

Recall 0.868 Normal / **0.962 Pneumonia** — the errors fall on the safe side for a
screening device (false alarms rather than missed pneumonia).

### Why this corpus works where COVID sets do not
Both classes come from **one hospital** (Guangzhou Women and Children's Medical Center),
**one paediatric cohort** (1–5 y), one imaging pipeline. There is no source variable to
shortcut on. That is precisely the control condition CoVScreen argues for, and it is why the
number survives an audit that the COVID datasets fail.

### Two audit findings on this corpus
1. **The official train/test split leaks.** **264 subjects (967 images) appear in both**
   official folders. Nearly every published result uses that split, which is a likely source
   of the 96–98 % figures. We recombined and assigned our own patient-disjoint folds.
2. **`personN` is not a global patient id.** `person1_bacteria_1` is in train while
   `person1_virus_11` is in test, and **979 of 1,674 person ids carry both bacterial and
   viral images** — implausible clinically. The numbering restarts per subset. Grouping by
   `personN` may merge two children into one "subject", which is the *conservative* error
   (costs a little data efficiency, cannot cause leakage), so it was kept. It does rule out a
   clean 3-class bacterial/viral task, which is why the node ships **binary**.

### Golden vectors
24 images (12 per class) from held-out fold 0, **19/24 correct offline**, **216 kB** of flash
as raw uint8 pixels — 5.4× cheaper than the cardio node's audio vectors (1,172 kB).
19/24 is an unbiased random draw, not a cherry-pick; at n=24 it is within sampling noise of
the model's 0.9015 fold accuracy.

### Edge-case behaviour is weaker than the audio node
`tests/test_edge_cases.py`, deployed INT8 model:

| input | max conf | verdict |
|---|---|---|
| all black | 0.938 | class 0 |
| all white | 0.996 | class 1 |
| mid grey | 0.918 | class 1 |
| uniform noise | 0.996 | class 0 |
| salt & pepper | 0.996 | class 0 |
| gradients / checkerboard | 0.898–0.996 | confident |

**8/8 degenerate inputs get a confident answer** (cardio: 2/8). A photo of a blank wall is
confidently classified. The confidence floor is useless here on its own — the **OOD gate
(docs/13 §4) is mandatory for this node**, not optional.

---

## 2a-bis. Final determination on the lung *audio* task (2026-08-14)

Five experiments, every one patient-disjoint, every one against its own measured hurdle:

| # | Task | Units | Cls | Hurdle | CNN | Δ | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Yaseen cardio** | 1000 clips | 5 | 0.200 | **0.989** | **+78.9** | ship |
| 2 | KAUH 4-class | 92 pt | 4 | 0.370 | 0.446 | +7.6 | Note: weak |
| 3 | **ICBHI 6-class** (Option 1) | 123 pt | 6 | **0.675** dev | 0.643 | **−3.2** | fail |
| 4 | **ICBHI Meditron-only** (device fixed) | 59 pt | 5 | 0.441 | 0.391 | **−5.0** | fail |
| 5 | **KAUH+ICBHI merged** (Option 2) | 201 pt | 6 | **0.582** dev | 0.592 | **+1.0** | fail |

**Option 1 and Option 2 have both failed.** Row 5 is within its own ±0.072 fold standard
deviation of the device-only baseline — statistically indistinguishable from a classifier that
reads the stethoscope model off the filename and never opens the audio.

Merged per-class recall confirms the mechanism:

| class | recall | source |
|---|---|---|
| COPD | 0.877 | mostly ICBHI |
| Normal | 0.650 | both |
| Heart Failure | 0.444 | KAUH only |
| Asthma | 0.258 | KAUH only |
| **Pneumonia** | **0.000** | ICBHI only |
| **BRON** | **0.000** | ICBHI only |

Two classes are never predicted at all. Fraiwan et al.'s published **99.62 %** on this exact
corpus is not reproducible under a patient-disjoint split with a device baseline in place.

### The vision alternative fails the same audit

Checked before proposing it. [CoVScreen, UNSW Sydney (arXiv:2405.07674)](https://www.alphaxiv.org/abs/2405.07674)
holds the COVID images fixed and changes only the source of the *Normal* class:

| setting | accuracy |
|---|---|
| Paediatric Normal vs Adult COVID (source-confounded) | **0.975** |
| **Adult Normal vs Adult COVID (source-controlled)** | **0.747** |

22.8 pp from paediatric-vs-adult anatomy alone. Their decisive control: a classifier separating
*healthy paediatric* from *healthy adult* X-rays — no pathology in either class — reaches
**0.974**. Severity matters too: PCR-positive but radiographically normal scores **0.427**
(below chance), mild 0.625, moderate 0.744, severe 0.810.

**Conclusion: no public lung corpus, in either audio or imaging, supports an honest 90 %+
multi-class disease model.** The published 95–99 % figures in both modalities are
source/device shortcuts.

### Recommended pivot: ICBHI EVENT-level, not disease-level

Normal / crackle / wheeze / both, at the respiratory-cycle level.

**Why this one is sound where the others are not:** event labels vary *within* a single
patient's recording — some cycles have crackles, some do not, same patient, same stethoscope,
same room. Device and source therefore **cannot** explain the label, so any accuracy is real
acoustic signal. It also supplies **6,898 annotated cycles** rather than 123 patients.

Published SOTA ≈ 0.65 ICBHI-score / 0.75–0.80 accuracy. Lower than 90 %, but it is the
*official benchmark*, so our number stands beside the literature instead of collapsing under
questioning. Requires the cycle annotation `.txt` files, which are not in the parquet mirror.

---

## 2b. Deployment hurdles — what a model must beat

`core/baselines.py`, evaluated on the same patient-disjoint folds. **A model that does not
clear its hurdle is not learning pathology, whatever its headline accuracy says.**

| Corpus | Units | Classes | Majority | Stratified random | **Device-only** | **HURDLE** | Our CNN | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Yaseen** (cardio) | 1000 clips | 5 | 0.2000 | 0.1997 | n/a | **0.2000** | **0.9890** | **+78.9 pp** |
| **KAUH** (lung) | 92 patients | 4 | 0.3696 | 0.2971 | n/a | **0.3696** | 0.4462 | Note: +7.7 pp |
| **ICBHI** (lung) | 123 patients | 6 | 0.5203 | 0.3328 | **0.6748** | **0.6748** | *pending* | — |

> Note: **ICBHI's hurdle is 0.6748, not 0.5203.** A classifier that looks only at which
> *stethoscope* recorded the file — never touching the audio — scores 67.5 % patient-level
> accuracy. Any ICBHI result below that is worse than not listening at all, and a result
> just above it is mostly device identification.

### The ICBHI device confound, quantified

Patient-level device × diagnosis:

| device | Bronchiect. | Bronchiol. | COPD | Healthy | Pneumonia | URTI |
|---|---|---|---|---|---|---|
| AKGC417L | 0 | 0 | **32** | 0 | 0 | 0 |
| Litt3200 | 0 | 0 | **10** | 0 | 0 | 0 |
| LittC2SE | 0 | 0 | 15 | 0 | 5 | 0 |
| Meditron | 7 | 6 | 7 | **26** | 1 | **14** |

At recording level it is starker: **646/646 AKGC417L and 60/60 Litt3200 recordings are COPD**
— 706 of 917 recordings (77 %) are perfectly identified by device alone. Sample rate tracks
device too (44.1 kHz / 10 kHz / 4 kHz), which is why everything is resampled to 4 kHz with an
anti-aliasing filter before feature extraction: otherwise sample rate *is* the fingerprint.

**Consequence for the plan:** the ICBHI result must be reported next to a
**device-controlled** result (Meditron-only: 59 patients, 5 classes, one device) so the
pathology signal can be separated from the device signal.

---

## 2c. Edge-case behaviour (cardio INT8) — measured

`tests/test_edge_cases.py`, run through the real front-end into the deployed `.tflite`:

| input | max confidence | verdict at floor 0.60 |
|---|---|---|
| digital silence | 0.238 | UNCERTAIN |
| DC offset | 0.238 | UNCERTAIN |
| denormal (1e-12) | 0.238 | UNCERTAIN |
| pure 1 kHz tone | 0.277 | UNCERTAIN |
| impulse | 0.289 | UNCERTAIN |
| clipped square | 0.535 | UNCERTAIN |
| **white noise** | **0.637** | **class MVP** |
| **full-scale noise** | **0.801** | **class MVP** |

**6/8 handled correctly; noise is confidently misclassified as Mitral Valve Prolapse.**

This is the empirical case for the OOD gate in
[docs/13 §4](../docs/13_REAL_WORLD_VALIDATION.md): a confidence floor alone does **not**
make the device safe on garbage input. The SNR + Mahalanobis OOD score is required, and now
we have a measured failure to justify it rather than a hypothesis.

Also passing: **amplitude invariance** — predictions are unchanged across 0.1× to 8× input
gain, confirming per-recording RMS normalisation removes absolute level as a cue.

---

## 3. The lung finding, with evidence

The 4-class KAUH task does **not** support ≥ 90 % under an honest patient-disjoint split.
This was established by elimination, not assumed:

| # | Experiment | Result | What it rules out |
|---|---|---|---|
| 1 | CNN, patient-disjoint | 44.6 % (floor 37.0 %) | — |
| 2 | RandomForest on hand-crafted features, same folds | 44.6 % | **CNN/training bug** |
| 3 | Logistic regression, same folds | 39.1 % | — |
| 4 | 10 front-end variants (band, normalisation, mel count, FFT size, window length) | 34.8 – 44.6 % | **front-end mis-configuration** |
| 5 | log-mel vs MFCC, augment vs no-augment | 43.5 – 45.9 % | **feature/augmentation choice** |
| 6 | **Same pipeline on Yaseen** | **98.9 %** | **pipeline correctness** |
| 7 | Deliberately LEAKY window-level split | 59.2 % | leakage alone as the explanation for published results |

### Where the signal actually is (binary sub-tasks, patient-disjoint)
| Task | Accuracy | Majority floor | Δ |
|---|---|---|---|
| Normal vs Heart Failure | 0.769 | 0.654 | **+11.5** |
| Normal vs Asthma | 0.631 | 0.523 | **+10.8** |
| Heart Failure vs COPD | 0.704 | 0.667 | +3.7 |
| Normal vs Abnormal | 0.674 | 0.630 | +4.4 |
| Normal vs COPD | 0.744 | 0.791 | **−4.7** |
| Asthma vs COPD | 0.650 | 0.775 | **−12.5** |
| Asthma vs Heart Failure | 0.592 | 0.633 | −4.1 |

There is *some* real signal (Normal vs Asthma, Normal vs HF), but three of seven binary
sub-tasks are **below their majority-class floor**. COPD, with 9 patients, is not learnable here.

### Why published KAUH numbers are 94–99 %
1. **Segment-level splits.** Windows from one recording land in both train and test.
2. **All three filter renderings used** (B/D/E = 336 files from 112 patients), tripling apparent
   data and putting the same acoustic event on both sides of the split.
3. Sen et al. ([arXiv:2606.10972](https://www.alphaxiv.org/abs/2606.10972)) make exactly this
   criticism of the Koshta et al. KAUH study: *"the possibility that data from the same subject
   or recording appeared both in the training and validation sets was not explicitly ruled
   out… it remains unconvincing that the classifier achieves fair learning."*

Note that our leaky-split experiment reached only 59.2 %, so leakage alone does not reach 94 %;
the published figures also benefit from the 3× rendering inflation and from classical models
fitted on far more segments.

### Root cause: the corpus is too small and too shallow
- **92 usable patients** after exclusions
- **exactly one recording per patient**, at **one** of ten chest locations that varies per patient
- **26.3 minutes** of audio in total → **587** 5-second windows
- COPD: **9 patients**; Heart Failure: 18

Auscultation site is confounded with patient, and patient is confounded with class. With one
recording per patient there is no within-patient variation for the model to learn to ignore.

---

## 4. Corpus defects discovered

The `pcm_md5` duplicate assertion (leakage rule, `core/preprocess.py`) caught three genuine
filing errors in the published KAUH corpus:

| # | Finding | Action |
|---|---|---|
| 1 | `DP100 "N,N,P R M,70,F"` and `DP101 "Asthma,E W,P L M,12,F"` are **byte-identical** (Pearson r = 1.0000) in both B and D renderings, but carry **opposite labels** and different ages (70 vs 12) | **Both patients excluded** — cannot determine which label is correct |
| 2 | `EP76` / `EP77` byte-identical in the E rendering only | Retained; the D renderings are genuinely distinct |
| 3 | `BP9` / `DP9` byte-identical, i.e. the Bell and Diaphragm renderings of P009 are the same file | Harmless; we use D only |

A full pairwise correlation scan of all 94 D-rendering files found **exactly one** pair above
r = 0.90 (the P100/P101 pair). Reproduce with `core/diagnose_separability.py`.

This is worth one line in the report: an automated integrity check found a labelling
contradiction in a published, peer-reviewed, CC BY dataset.

---

## 5. Design decisions changed by this work

| Was | Now | Reason |
|---|---|---|
| Resample lung audio to 8 kHz | **Keep 4 kHz native** | KAUH is natively 4 kHz/int16/mono. Upsampling adds no information, and 4 kHz halves the on-device sample count. Nyquist (2000 Hz) already matches the analysis band. |
| 10-fold CV | **5-fold** | COPD has 9 patients; 10 folds leaves <1 per fold |
| Heavy augmentation assumed helpful | **Ablated; roughly neutral-to-harmful** | Matches Sen et al., who found augmentation degraded pulmonary sound classification |
| log-mel assumed best | **Both implemented, log-mel retained** | MFCC gave 45.9 % vs 43.5 % on KAUH — within noise; log-mel won on Yaseen |
| Lung window 5 s @ 8 kHz = 40,000 samples | **5 s @ 4 kHz = 20,000 samples** | halves DSP cost on the Explorer Kit |

Feature shapes now: **pulmo 64 × 160**, **cardio 40 × 192**.

---

## 6. Recommendation

**Cardio is done and exceeds target.** Freeze it and move it to quantisation + device.

**For lungs, KAUH alone must be replaced.** Recommended: **ICBHI 2017**, which has
**920 recordings from 126 patients** — roughly 7 recordings per patient at multiple chest
locations, versus KAUH's single recording per patient. That within-patient variation is exactly
what is missing here. Options, in order of preference:

1. **ICBHI disease-level, patient-disjoint.** Standard public benchmark, 10× the recordings.
   Must report per-class recall because COPD is 64/126 patients.
2. **KAUH + ICBHI merged (Fraiwan 6-class).** The published 99.62 % subject-level result.
   Needs the source-confound analysis (Heart Failure exists only in KAUH).
3. **Keep KAUH, reduce the task** to Normal / Asthma / Heart Failure (drop COPD). Expect
   ~60–65 %. Honest, but short of target.

The pipeline is dataset-agnostic — adding ICBHI is a manifest parser plus a download, not new
science.
