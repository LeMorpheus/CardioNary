# Real-World Practicality — Assessment and Validation Protocol

**Document status:** Current as of 2026-08-24.
**Scope:** Assessment of the deployed system's readiness for use on signals acquired
outside the training datasets, and the protocol by which that readiness would be
established.

---

## 0. Summary of position

1. The system has **not** been evaluated on clinically acquired signals. No claim of
   clinical validity is made anywhere in this document set.
2. On recordings acquired with different equipment, a substantial reduction in accuracy
   relative to held-out dataset performance should be expected. §2 quantifies the
   mechanisms.
3. A controlled healthy-volunteer study (§3) recovers a significant proportion of the
   relevant insight without the ethical and administrative requirements of a clinical
   study, and is the recommended immediate next step.

**Status note.** This document was originally written when the pulmonary task was lung
auscultation audio. That task was subsequently withdrawn on evidence of a recording-device
confound and replaced with paediatric chest radiography
([15_LUNG_MODALITY_DECISION.md](15_LUNG_MODALITY_DECISION.md)). The lung-audio analysis in
§2 is retained because it constitutes the evidence base for that decision, and because it
applies directly to the cardiac node, which remains an audio task.

---

## 1. Readiness of the system as delivered

| Layer | Real-world readiness |
|---|---|
| **Signal-to-decision algorithm** | Established. A log-mel CNN over auscultation audio is the approach used in the published clinical machine-learning literature. |
| **Edge deployment** | Established. INT8 TFLM on a Cortex-M33 with MVP acceleration, 57 ms per cardiac inference, verified on hardware. This is a deployable inference engine. |
| **Wireless, gateway and dashboard** | Established. BLE GATT to a Raspberry Pi 4 hosting a local dashboard, verified with two nodes operating concurrently. |
| **Acoustic front-end** | **Not implemented.** The delivered system replays clips held in flash. Microphone, acoustic coupling and analogue front-end are deferred to the next development phase. |
| **Clinical validity** | **Not established.** Models are trained on public corpora drawn from a small number of institutions. |

The engineering is established; the clinical claim is not. The system is described
throughout as a **screening and decision-support demonstrator**.

---

## 2. Expected behaviour on clinically acquired audio

The predictions below are recorded in advance of the experiment so that measured results
can be compared against a stated expectation.

| Failure mode | Mechanism | Expected severity |
|---|---|---|
| **Device transfer-function shift** | KAUH was recorded on a 3M Littmann 3200 electronic stethoscope with a specific filter response (Diaphragm rendering, 20–2000 Hz). Any other microphone presents a different frequency response, noise floor and coupling. | **Severe** — the dominant factor |
| **Session-signature reliance** | Diagnosis is a per-patient constant. Even under patient-disjoint splits, a model may rely on room or device cues shared across a corpus. | **Moderate to severe** |
| **Ambient noise** | Ward noise, speech, alarms, footfall and clothing friction. Training data is comparatively quiet. | **Moderate** |
| **Cardiac bleed-through** | Heart sounds carry substantial energy in the lung band (20–150 Hz). An 80–2000 Hz band-pass attenuates but does not remove S1/S2. | **Moderate** |
| **Class-prior shift** | Ward disease prevalence differs from dataset prevalence. A model trained on nine COPD patients does not constitute a COPD screener. | **Moderate** |
| **Curated-source shift (cardiac)** | Yaseen clips are teaching recordings: clean, well-positioned, textbook morphology. Bedside phonocardiography is considerably less controlled. | **Severe for the cardiac node** |

**Quantified expectation.** On recordings acquired with a different stethoscope, accuracy
may fall from approximately 95 % to the **40–70 %** range. More significantly, the model may
be *confidently* incorrect, because a standard softmax classifier carries no representation
of whether an input lies outside its training distribution.

That property is the design problem addressed in §4.

---

## 3. Controlled healthy-volunteer study

A clinical study requires institutional approval, written informed consent and a clinician
to establish ground truth. The healthy-volunteer study below requires none of these and
yields directly comparable distributional evidence.

| Property | Specification |
|---|---|
| Subjects | 4–6 healthy adult volunteers |
| Consent | Written, single-page, covering recording and retention |
| Equipment | The I2S MEMS breakout, or a stethoscope acoustically coupled to a calibrated microphone. Equipment is documented exactly. |
| Sites | Chest anterior and posterior, upper and lower, left and right |
| Ground truth | **Healthy** |
| Duration | One session |

### 3.1 Rationale for healthy-only recruitment

Both models carry a `Normal` class, which makes a healthy-only cohort informative:

- If the model returns **Normal at high confidence**, the domain gap is narrower than the
  §2 analysis predicts.
- If the model returns a **disease class at high confidence on a healthy young adult**,
  the result is a quantified measurement of domain shift.

Both outcomes are measurements. The experiment has no uninformative result.

### 3.2 Measurements

1. Predicted-class distribution across all healthy clips, expected to concentrate on `Normal`
2. Confidence distribution compared against the held-out dataset test set
3. Abstention rate under the §4 gate
4. Measured SNR of acquired recordings against the training corpus
5. Log-mel statistics — per-band mean and standard deviation — for acquired recordings
   against the training distribution, presented as a distribution overlay

---

## 4. Input quality and distribution gating

**Implementation status: proposed, not implemented.** The delivered system applies a
confidence floor only. The gate described in this section is specified here for the next
development phase and is not present in the current firmware.

Three low-cost on-device signals convert an incorrect classification into a quantified
diagnosis of *why* it was incorrect.

### 4.1 Signals

| Signal | Method | Cost |
|---|---|---|
| **`snr_db`** | Band energy (80–2000 Hz) against a noise-floor estimate (10th percentile of frame energies) | Negligible; reuses the existing mel frames |
| **`ood_score`** | Mahalanobis-style distance of an 8-dimensional log-mel summary (per-band mean and standard deviation) from the training-set mean and diagonal covariance, both stored as constants | ~200 bytes of constants; a few hundred cycles |
| **`clip_pct`** | Fraction of samples at full scale | Trivial |

### 4.2 Decision policy

```
if   clip_pct  > 2%      -> LOW_QUALITY            (saturated input)
elif snr_db    < 10      -> LOW_QUALITY            (insufficient SNR)
elif ood_score > tau     -> OUT_OF_DISTRIBUTION    (outside training distribution)
elif confidence < 0.60   -> UNCERTAIN
else                     -> report class
```

`tau` is calibrated as the 99th percentile of `ood_score` over the training fold.

### 4.3 Justification

- A screening device that abstains on an unusable recording is safer than one that
  produces a classification regardless of input quality.
- Abstention is measurable and reportable, whereas an unqualified accuracy figure on
  out-of-distribution input is not interpretable.
- The mechanism is consistent with the confidence-floor behaviour already verified on
  hardware, where raising the floor to 0.99 caused the pulmonary node to abstain on
  114 of 120 cases.

### 4.4 Presentation

The dashboard presents input quality alongside every prediction: SNR, distribution
indicator, and the reason string when a result is withheld. A class is never presented
without an accompanying indication of how far it can be relied upon.

---

## 5. Clinical study protocol

The protocol below is documented for execution when institutional access permits. It is a
specification, not a completed activity.

1. **Approvals.** Institutional permission and written informed consent. No direct
   identifiers recorded.
2. **Equipment.** A 3M Littmann 3200 where obtainable, matching the training device; this
   single choice is expected to dominate transfer performance. Where unobtainable, the
   exact device is documented.
3. **Sites.** The four to six standard chest positions used by KAUH, recorded per subject.
4. **Ground truth.** Attending clinician's diagnosis, recorded independently of the audio.
5. **Sample size.** Not fewer than 20 subjects and not fewer than 5 per target class, so
   that per-class recall is interpretable.
6. **Blinding.** The operator running the model does not see the diagnosis until
   predictions are logged.
7. **Metrics.** Accuracy, macro-F1 and per-class recall, together with abstention rate,
   SNR distribution, and the log-mel distribution overlay against training data.
8. **Pre-registration.** The expected accuracy is recorded before the study is run.

---

## 6. Measured limitations and proposed mitigations

| Measured limitation | Proposed mitigation | Basis |
|---|---|---|
| Device transfer-function shift dominates error | Per-device calibration: record a reference tone and silence, normalise band gains before inference | Standard practice in acoustic sensing |
| Ambient noise degrades SNR | Ambient-noise context node (BGM220 Explorer Kit): a second BLE node reports room noise level; the gateway flags or rejects recordings above a threshold | See §6.1 for the accurate characterisation |
| Model confidently incorrect outside the training distribution | Distribution gate (§4) | Specified; not yet implemented |
| Small training population | Multi-site collection; domain-adversarial training or device augmentation (random EQ, impulse-response convolution) during training | Established technique |
| Cardiac bleed into the lung band | Joint processing across the two nodes: the cardiac node localises S1/S2 in time, permitting lung inference to be gated to inter-beat intervals | Enabled by the two-node topology |
| No clinical validation | Prospective study protocol (§5) | Specified |

The final mechanism in that table is a direct consequence of the two-node architecture:
the cardiac node can determine when the pulmonary node should sample.

### 6.1 Accurate characterisation of the ambient-noise node

- The BGM220 node does **not** perform noise cancellation. Coherent noise subtraction
  requires a reference microphone sampled on the same clock as the primary signal. A
  spatially separated node communicating over BLE is neither synchronised nor
  phase-coherent.
- The node provides **environmental noise context**: an independent ambient SPL and SNR
  measurement that the gateway uses to gate, flag or annotate recordings, and to indicate
  that a recording should be repeated in a quieter position.

---

## 7. Relationship to the implementation plan

Nothing in §1–§6 alters the Track A priority of a complete working pipeline. Specifically:

- The **distribution gate (§4)** is Track B item B2. It is low-cost and high-value, but is
  not scheduled ahead of end-to-end system operation.
- The **healthy-volunteer study (§3)** is Track B. It requires no firmware modification
  provided the gateway accepts a WAV file for offline scoring.
- The **clinical protocol (§5)** is a specification document rather than a scheduled task.

See [10_IMPLEMENTATION_PLAN.md](10_IMPLEMENTATION_PLAN.md).
