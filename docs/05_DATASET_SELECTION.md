# Dataset Selection & Evaluation Protocol — DECISIONS

Evidence for these decisions is in [04_DATASETS_RESEARCH.md](04_DATASETS_RESEARCH.md).
This file is the contract: it fixes the task, the labels, the split, and the number we are
allowed to claim.

---

## 1. Decision summary

> ## Note: superseded for the pulmonary node — see report A1
>
> Five patient-disjoint experiments established that **no public lung-*audio* corpus supports
> an honest ≥90 % multi-class disease model**: KAUH 4-class reached 0.446 (hurdle 0.370),
> ICBHI 6-class 0.643 **below** its 0.675 device-only hurdle, ICBHI single-device 0.391 below a
> 0.441 floor, and the KAUH+ICBHI merge 0.592 against a 0.582 device hurdle — i.e.
> indistinguishable from reading the stethoscope model off the filename.
>
> **The pulmonary node is now chest-X-ray pneumonia** (Kermany corpus), which reaches
> **0.9785 INT8 patient-level**. The cardio node is unchanged.

| | **Cardio node** (Dev Kit BRD2608A) | **Pulmo node** (Explorer Kit BRD2709A) |
|---|---|---|
| **Modality** | audio (auscultation) | **image (chest X-ray)** |
| **Dataset** | **Yaseen 2018** | **Kermany paediatric CXR** |
| **Classes** | **5** | **2** — Normal, Pneumonia |
| **Labels** | Normal, Aortic Stenosis, Mitral Stenosis, Mitral Regurgitation, Mitral Valve Prolapse | Normal, Pneumonia |
| **Input** | 40 × 192 log-mel, 8 kHz, 3.0 s | 96 × 96 grayscale |
| **Corpus size** | 1000 clips, 200/class (balanced) | 5,856 images / 2,790 subjects |
| **Split** | clip-disjoint, stratified 5-fold CV | **patient-disjoint** 5-fold CV |
| **Hurdle** | 0.200 | 0.600 |
| **Achieved (INT8)** | **0.9806** | **0.9785** (patient level) |
| **Architecture** | depthwise-separable CNN, 13.5 k params | MobileNetV2 α=0.35, ImageNet-pretrained, 413 k params |

Why the modality split is coherent rather than opportunistic: **cardiac auscultation is a
contact-acoustic problem the MCU can genuinely own, while pulmonary screening at this accuracy
needs imaging.** Both run INT8 TFLM on EFR32MG26 with MVP-accelerated kernels, both report over
the same BLE GATT profile, both render on the same dashboard.

Both tasks are genuinely multi-class. Neither is normal-vs-abnormal. Requirements
**R1, R2, R3** are satisfiable.

> **Changed 2026-08-13.** The pulmo task moved from the 6-class KAUH+ICBHI merge to the
> **KAUH-only 4-class** task (formerly fallback rung C), and the accuracy gates were split into
> a **Track A gate of 90 %** and a **Track B goal of 95 %+**. This follows the explicit
> instruction to get a complete working pipeline first and optimise afterwards. Rung C needs no
> 4 GB ICBHI download, has no cross-device confound to defend, and still clears 95 % in
> practice. The 6-class version is Track B item **B6** — see
> [10_IMPLEMENTATION_PLAN.md](10_IMPLEMENTATION_PLAN.md).
>
> **Modality:** a chest X-ray vision CNN was evaluated as an alternative and **rejected on
> evidence** — it is *less* accurate at MCU scale, not more. See
> [15_LUNG_MODALITY_DECISION.md](15_LUNG_MODALITY_DECISION.md).

---

## 2. Cardio task specification

### 2.1 Labels

| ID | Class | Acoustic signature |
|---|---|---|
| 0 | **N** — Normal | Clean S1/S2, energy concentrated 20–150 Hz, quiet systole and diastole |
| 1 | **AS** — Aortic Stenosis | Harsh crescendo–decrescendo (diamond) systolic ejection murmur, ends before S2 |
| 2 | **MS** — Mitral Stenosis | Opening snap + low-pitched **diastolic** rumble, presystolic accentuation |
| 3 | **MR** — Mitral Regurgitation | **Holosystolic** murmur, roughly constant intensity, obscures S1, runs into S2 |
| 4 | **MVP** — Mitral Valve Prolapse | Mid-systolic click then late-systolic crescendo murmur |

These five are acoustically distinguishable by **timing relative to S1/S2** and **spectral
shape** — exactly what a log-mel CNN sees. That is the mechanistic reason 98 %+ is real here
and not an artefact.

### 2.2 Preparation
1. Load `.wav` (already 8 kHz mono — **verify, do not assume**; assert in code).
2. DC-removal; per-clip peak normalisation to −1 dBFS.
3. Band-pass **20–800 Hz** (Butterworth order 4, zero-phase for offline; the on-device
   equivalent is a fixed IIR — see [06_MODEL_DESIGN.md](06_MODEL_DESIGN.md)).
4. Pad (reflect) or centre-crop to exactly **3.0 s = 24,000 samples**.
5. Log-mel: 256-pt FFT, 32 ms frame, hop 16 ms, **40 mel bins**, 20–800 Hz → **40 × 187**,
   then time-pool to **40 × 96** (see §6 of the model doc for why we fix the shape).

### 2.3 Split
Yaseen clips are individual recordings without a published patient ID. Treat **each file as
an independent unit**, use **stratified 10-fold CV**, and — critically — **augment only
inside the training fold**. Never augment then split.

### 2.4 What we may and may not claim
"5-class valvular heart-sound classification, 98.x % ± y % (stratified 10-fold CV) on the
Yaseen 2018 corpus, deployed as an INT8 TFLM model on EFR32MG26."
"Our device diagnoses aortic stenosis."

---

## 3. Pulmo task specification

### 3.1 Labels — Track A (KAUH only, 4-class)

| ID | Class | Patients | Acoustic signature |
|---|---|---|---|
| 0 | **Normal** | **35** | Vesicular/bronchial, broad energy < 400 Hz, inspiration-dominant |
| 1 | **Asthma** | **32** | Continuous musical **wheeze**, fundamental > 400 Hz, > 100 ms |
| 2 | **Heart Failure** | **21** | Fine bibasal crackles from pulmonary oedema |
| 3 | **COPD** | **9** | Prolonged expiration, wheeze/rhonchi, reduced breath sounds |
| | **Total** | **97 patients** | |

Reasonably balanced apart from COPD. Single recording device (Littmann 3200), single site,
single filter rendering (Diaphragm) — so there is **no cross-device confound to defend**, which
is exactly why this is the Track A task.

"Heart failure in a lung model" is not a mistake — pulmonary oedema is heard in the *lungs*,
and it is a genuinely valuable class for a cardiopulmonary product. It is also a good
talking point in the presentation.

### 3.1b Track B upgrade (B6) — 6-class, KAUH + ICBHI

If time allows, extend to 213 patients / 1,483 recordings by merging ICBHI:

| ID | Class | KAUH + ICBHI |
|---|---|---|
| 0 | Normal | 35 + 26 = **61** |
| 1 | Asthma | 32 + 1 = **33** |
| 2 | COPD | 9 + 64 = **73** |
| 3 | Pneumonia | 5 + 6 = **11** |
| 4 | BRON (bronchiectasis/bronchitis) | 3 + 13 = **16** |
| 5 | Heart Failure | 19 + 0 = **19** |

This is the Fraiwan et al. task with the published **99.62 %** subject-level result. It requires
the device-confound analysis in [04 §6.3](04_DATASETS_RESEARCH.md) and the source-stratified
confusion matrix — which is why it is an upgrade, not the starting point.

### 3.2 Preparation
1. **Select filter rendering `D` (Diaphragm) only** from KAUH — see the leakage rule in §5.2.
2. Resample everything to **8 kHz** mono (`scipy.signal.resample_poly`, anti-aliased).
3. DC-removal; per-recording RMS normalisation (kills device gain as a cue).
4. Band-pass **80–2000 Hz**.
5. Slice into **5.0 s = 40,000-sample** windows, 50 % overlap, drop the tail if < 2.5 s.
6. Log-mel: 512-pt FFT, 64 ms frame, hop 32 ms, **64 mel bins**, 80–2000 Hz →
   **64 × 156**, time-pooled to **64 × 128**.
7. **Label every window with its parent recording's diagnosis and its patient ID.**

### 3.3 Split — non-negotiable
- **Patient-disjoint 10-fold cross-validation.** Patient ID is parsed from the filename
  **before** windowing and carried on every window.
- Folds are stratified by class *at the patient level*, not the window level.
- Class imbalance handled by **class-weighted loss** (or balanced batch sampling), never by
  duplicating windows across the split boundary.

### 3.4 Metrics we report
Accuracy alone is not acceptable when COPD holds only 9 of 97 patients.
Every result table reports:
- **Accuracy** (mean ± std over 10 folds)
- **Macro-F1**
- **Per-class recall** — expect **COPD to be the weak class**; say so rather than hiding it
- **Confusion matrix**
- *(Track B / 6-class only)* **Source-stratified confusion matrix** — KAUH rows vs ICBHI rows

### 3.5 Confound status
The Track A task is **single-device, single-site, single-filter-rendering**, so the
cross-device confound described in [04 §6.3](04_DATASETS_RESEARCH.md) **does not apply**. That
is the main reason it was promoted from fallback to primary: there is no merge artefact to
defend in Q&A.

If Track B item **B6** upgrades to the 6-class KAUH+ICBHI task, the confound returns, and the
Track A 4-class result then serves as the **required ablation** proving the merged result is not
carried by a device shortcut. Keep the Track A metrics — they become the defence.

---

## 4. Fallback ladder

If a primary target does not hold up under honest evaluation, step **down** the ladder and
say so in the report. Do not quietly loosen the split.

### Cardio
| Rung | Task | Expected |
|---|---|---|
| **A (primary)** | Yaseen 5-class | 98–99 % |
| B | Yaseen 5-class, merge MVP+MR (acoustically closest pair) → 4-class | 99 %+ |
| C | Yaseen 5-class + CinC-2016 normals as a 6th "other/noisy" class | 95–97 %, more robust |

### Pulmo
| Rung | Task | Expected | Status |
|---|---|---|---|
| **C** | **KAUH-only 4-class** (Normal/Asthma/HF/COPD), single device | **95 %+**, cleanest story | **Plan of record (Track A)** |
| A | KAUH+ICBHI 6-class, patient-disjoint | 95–99 % | Track B item B6 |
| B | KAUH+ICBHI **5-class** (fold Pneumonia into BRON — both coarse-crackle) | higher, more stable | If B6 proves unstable |
| D | ICBHI **event-level** 4-class (crackle/wheeze/both/normal) | ~65 % | **Secondary scientific result only, never the headline** |

Rung C is a perfectly respectable deliverable: 4 classes, single device, no merge confound,
fully patient-disjoint, and it clears the target. Promoting it from fallback to primary is the
single biggest schedule saving available — it removes a 4 GB download, a merge step, and an
entire confound defence from the critical path.

**A chest X-ray vision model was also considered and rejected** — at MCU-scale resolution it is
*less* accurate than this audio task, and it breaks the edge-AI story.
See [15_LUNG_MODALITY_DECISION.md](15_LUNG_MODALITY_DECISION.md).

---

## 5. Anti-leakage rules (mandatory, enforced in code)

These are the rules that make the accuracy number real. Each must be an **assertion in the
preprocessing script**, not a note in a README.

| # | Rule | Enforcement |
|---|---|---|
| 5.1 | No patient appears in more than one fold | `assert set(train_pids) & set(test_pids) == set()` |
| 5.2 | KAUH filter renderings `B`/`D`/`E` of the same patient never straddle a split | select `D` only; assert no `B`/`E` in the manifest |
| 5.3 | Augmentation applied **after** the split, training fold only | augmentation lives in the `Dataset`/generator, never in preprocessing |
| 5.4 | Overlapping windows from one recording never straddle a split | window→recording→patient chain asserted |
| 5.5 | Normalisation statistics computed on the **training fold only** | fit scaler inside the fold loop |
| 5.6 | Test-set clips used as golden vectors are from the **held-out fold** | vector export takes fold index as an argument |

Rule 5.6 matters for the demo: the clips we flash onto the boards must be clips the model
**never saw during training**, or the live demonstration proves nothing.

---

## 6. Data augmentation (training fold only)

Kept deliberately light and physically plausible:

| Augmentation | Range | Why |
|---|---|---|
| Gain jitter | ±6 dB | microphone/coupling variation |
| Time shift | ±0.5 s circular | window-alignment invariance |
| Time stretch | 0.9–1.1× | heart-rate / respiratory-rate variation |
| Additive noise | SNR 15–30 dB, pink + recorded room tone | ambient robustness |
| SpecAugment | 1 freq mask ≤ 8 bins, 1 time mask ≤ 12 frames | regularisation |

**No pitch shift on cardiac data** — murmur pitch is diagnostic information, and shifting it
teaches the model to ignore a real discriminative cue.

---

## 7. Deliverables from this stage

- [ ] `scripts/download_datasets.py` — fetch + checksum-verify Yaseen, KAUH, ICBHI
- [ ] `DATA_LICENSES.md` — licence + required citation per dataset
- [ ] `data/manifests/cardio_manifest.csv` — `file, class, fold`
- [ ] `data/manifests/pulmo_manifest.csv` — `file, patient_id, source, class, fold`
- [ ] `scripts/preprocess.py` — with all six §5 assertions active
- [ ] `reports/dataset_stats.md` — class counts, patient counts, duration histograms per fold
