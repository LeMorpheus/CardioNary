# Derived Indicators — extracting more from the existing models

**Constraint:** no new network, no retraining, no additional prediction heads, no
new dataset. Everything here is either classical signal/image processing on the
same bytes the device already holds, or probing the already-deployed INT8 model.

**Anchor rule:** the existing classification remains the primary result. Every
indicator below annotates it. None of them overrides it, and none is presented
as an independent diagnosis.

**Where it runs:** offline in `core/descriptors.py`, over the exact vectors flashed
on the device. Firmware cost: **zero flash, zero RAM, zero added inference time.**
On-device feasibility is noted per item for a future live version.

---

## 1. Tier definitions

| Tier | Meaning |
|---|---|
| **1** | Directly extractable from the existing signal or model output |
| **2** | Simple signal/image processing |
| **3** | Weak heuristic — indicative only, never sufficient alone |
| **4** | **Not defensibly inferable — deliberately not computed** |

Everything was **measured against the held-out set** before being shown. Two
proposals failed that test and were changed or withdrawn; both failures are
documented below rather than quietly dropped, because the negative results are
more useful than the positive ones.

---

## 2. Audio — what worked

### 2.1 Cardiac cycle segmentation — tier 2

Physiological basis: within a cycle, the S1→S2 gap (systole) is **shorter** than
the S2→S1 gap (diastole). So the alternating sequence of inter-peak gaps splits
into two interleaved groups, and the group with the smaller mean is systole. No
labels, no learning.

Implementation: Shannon-energy envelope → `find_peaks` → alternate-gap
classification. **On-device feasible** (a few hundred ops per second of audio).

### 2.2 Murmur detection and timing — tier 2 validated

Measured as sustained energy where there should be relative quiet: RMS of the
25–600 Hz band-passed waveform in the *interior* of each gap (25% trimmed at
each end to exclude the S1/S2 skirts), relative to the RMS at the heart sounds.

**Measured phase-energy ratios over the 53 held-out cases:**

| class | systolic ratio | diastolic ratio | expected |
|---|---|---|---|
| Normal | 0.001 ± 0.001 | 0.006 ± 0.017 | neither |
| Aortic stenosis | **0.859 ± 0.131** | 0.078 ± 0.161 | systolic |
| Mitral stenosis | 0.058 ± 0.043 | **0.124 ± 0.046** | diastolic |
| Mitral regurgitation | **0.486 ± 0.170** | 0.019 ± 0.021 | systolic |
| Mitral valve prolapse | **0.320 ± 0.252** | 0.080 ± 0.120 | systolic |

The measurement recovers the correct phase for **all five classes**, including the
diastolic predominance of mitral stenosis.

**Resulting phase call (threshold 0.06): 49 / 52 match physiology.**

| class | result |
|---|---|
| Normal | 11/11 none |
| Aortic stenosis | 10/10 systolic |
| Mitral stenosis | 7/9 diastolic |
| Mitral regurgitation | 11/11 systolic |
| Mitral valve prolapse | 10/11 systolic |

> **A calibration failure worth recording.** The first threshold tried was 0.18,
> which sits *above* the mitral-stenosis diastolic ratio — so all 9 MS cases were
> reported as "no murmur", for the one class whose defining sign is a diastolic
> murmur. The measurement had been right the whole time; the cut-off was wrong.
> This is exactly why each indicator is validated against the labels before being
> displayed, and why a plausible-looking output is not evidence of a working one.

### 2.3 Other audio indicators

| Indicator | Tier | Notes |
|---|---|---|
| Heart rate | 2 | Envelope autocorrelation, 45–140 bpm window. 52/53 valid, all 60–94 bpm |
| Rhythm regularity | 2 | Beat-interval coefficient of variation |
| Murmur duration (% of phase) | 2 | Fraction of the phase interior above 20% of reference |
| Relative murmur intensity (dB) | 2 | Murmur RMS vs heart-sound RMS |
| Spectral centroid, band powers | 2 | Welch PSD over 20–800 Hz in four bands |
| S1/S2 amplitude ratio | 2 | Alternating peak groups |
| Systole / diastole duration (ms) | 2 | From segmentation |
| **Mid-systolic click (MVP)** | **3** | Sharp brief transient in mid-systole. Reported as *"suspected"* only |
| Murmur shape (crescendo etc.) | 3 | Not implemented — envelope shape at this SNR is unreliable |
| **Severity / valve area / gradient** | **4** | **Not computed.** Requires echocardiography, not sound |

---

## 3. Audio — what failed

### Model occlusion by cardiac phase — withdrawn

**Idea:** blank the systolic vs diastolic frames of the input spectrogram, re-run
the deployed INT8 model, and see which phase the prediction depends on. No
gradients, so it works on the quantised graph exactly as shipped.

**Result: it does not discriminate.** It answered "systole" for *every* class,
Normal included. Normalising the confidence drop per occluded frame (diastole is
simply longer) barely helped.

**Interpretation:** it reflects where the signal *energy* is, not where the
diagnostic *information* is. It is computed and retained in `case_data.json` with
`"discriminative": false`, but is **not shown in the dashboard**. The DSP
phase-energy ratio is used instead, because that one demonstrably works.

---

## 4. Vision — what failed, and what worked

### 4.1 Opacity burden by percentile — withdrawn as circular

The first implementation thresholded at the 70th percentile of the image. That is
**tautological**: taking the top 30% of pixels guarantees a "burden" near 30% for
every image regardless of content. Measured output: Normal **29.5%**, Pneumonia
**29.3%** — the statistic was reporting its own threshold.

Replaced with an absolute cut on the shared intensity scale (images are
centre-crop + resize only, never per-image normalised, so levels are comparable).

### 4.2 Simple image statistics do not separate the classes

Measured over the 120 held-out images:

| statistic | Normal | Pneumonia | verdict |
|---|---|---|---|
| mean intensity | 138.8 ± 12.2 | 140.8 ± 20.5 | **no separation** |
| lower-zone mean | 144.7 ± 14.5 | 149.1 ± 18.5 | **no separation** |
| dense area (absolute threshold) | 14.5 ± 9.3% | 14.5 ± 15.2% | **no separation** |
| **texture heterogeneity** | **40.1 ± 4.6** | **34.8 ± 7.2** | weak (~0.9 SD), tier 3 |

Only texture shows any signal — consistent with consolidation filling in
normally-aerated, high-contrast lung. It is far too overlapping to decide
anything alone.

**The honest conclusion: at 96×96 the CNN can separate these classes (96%) and
simple thresholding cannot.** The image descriptors are shown as *descriptions of
the image*, with an explicit on-screen note that they do not detect pneumonia.

### 4.3 Model occlusion by region — tier 1 validated

Blank each cell of a 3×3 grid, re-run the deployed INT8 model, measure the
confidence drop. A 3×3 grid matches the coarse spatial resolution actually
available at 96×96; claiming finer localisation would be false precision.

**Where the model looks, by true class:**

| class | upper | middle | lower |
|---|---|---|---|
| Normal | **36** | 19 | 5 |
| Pneumonia | 9 | **43** | 8 |

A clear, interpretable difference: for pneumonia the model attends to the
**mid-lung fields**; for normal films it attends to the upper zones. This is the
strongest vision indicator available, and it comes from the existing model with
no retraining.

### 4.4 Vision indicators summary

| Indicator | Tier | Shown? |
|---|---|---|
| Region driving the result (occlusion) | 1 | Yes |
| Dense area %, laterality, zone distribution, focal/diffuse, region count | 2 | as *image description*, with non-discrimination note |
| Texture heterogeneity | 3 | labelled weak |
| **Cardiomegaly** | **4** | needs cardiac borders and a cardiothoracic ratio; 96×96 centre-crop does not reliably contain them |
| **Pleural effusion** | **4** | needs costophrenic angles; cropped out |
| **Pneumothorax** | **4** | needs a pleural line; far below this resolution |
| **Pulmonary oedema / congestion / atelectasis** | **4** | **the model has no training data for these.** A pneumonia-trained binary classifier cannot diagnose them, and the descriptors cannot either |

> **The line that must not be crossed:** "the image contains a measurable visual
> feature" is not "the model can diagnose a disease". Every tier-4 item above is
> a condition someone could plausibly *ask* this system about. None is computed,
> and the case view states what the screen cannot tell you.

---

## 5. Physiology reference

For each class: mechanism → what should appear → what we can measure → what we
actually report.

| Class | Mechanism | Expected sign | Measurable here | Reported |
|---|---|---|---|---|
| **Normal** | — | Clean S1/S2, quiet gaps | Phase ratios near zero | "No murmur", heart rate, regularity |
| **Aortic stenosis** | Narrowed aortic valve → turbulent ejection | Systolic ejection murmur, upper right chest | Systolic ratio 0.86 — the strongest of any class | Systolic murmur, duration, intensity, dominant frequency |
| **Mitral stenosis** | Narrowed mitral valve → turbulent filling | Low-pitched **diastolic** rumble at apex | Diastolic ratio 0.124 > systolic 0.058 | Diastolic murmur |
| **Mitral regurgitation** | Incompetent mitral valve → systolic backflow | Holosystolic murmur at apex | Systolic ratio 0.49 | Systolic murmur, duration |
| **Mitral valve prolapse** | Leaflet displacement | Mid-systolic click ± late murmur | Systolic ratio 0.32 (variable); click detection weak | Systolic murmur; click as *suspected* only |
| **Pneumonia** | Alveolar consolidation | Opacity on radiograph | Model attends mid-lung; texture weakly lower | Class + attention region + image description |

### Clinically associated, but NOT inferable here

These belong in the guidance panel as *context*, clearly marked as generic
reference. They are **not** model outputs and must never be presented as such:

- **Aortic stenosis** → syncope, exertional angina, dyspnoea, LV hypertrophy
- **Mitral stenosis** → atrial fibrillation, rheumatic fever history, haemoptysis
- **Mitral regurgitation** → orthopnoea, ankle oedema, left atrial enlargement
- **MVP** → palpitations, atypical chest pain
- **Pneumonia** → fever, hypoxia, tachypnoea, dehydration, chest indrawing in children

None of these is measurable from a 3-second heart sound or a 96×96 radiograph.
The dashboard shows them under **"Suggested next steps"** with the label
*"General reference for this finding — not derived from this recording"* and a
coloured rule separating them from measurements.

---

## 6. What the dashboard shows

Four visually distinct sections, in decreasing epistemic strength:

1. **Analysis result** — the model's class, confidence, full distribution *(anchor)*
2. **Measured from this recording** — heart rate, regularity, sound events, quality
3. **Descriptive indicators** — murmur phase/duration/intensity, spectral content, or the vision attention region and image description
4. **Suggested next steps** + **What this screen cannot tell you** — generic reference, explicitly labelled

Example of the intended shape:

```
Predicted class:  Mitral regurgitation        <- the model, unchanged
  Murmur:                 Yes
  Timing in cycle:        Systole
  Murmur duration:        41%
  Relative intensity:     -6.3 dB
  Dominant frequency:     168 Hz
```

---

## 7. Reproducing

```bash
python core/build_case_data.py
```

Recomputes every indicator from the flashed vectors, prints the full validation
table above, and writes `gateway/case_data.json`. **Re-run it after any change to
the vectors or thresholds, and read the validation output** — that table is the
only thing standing between a descriptive indicator and a confident fiction.
