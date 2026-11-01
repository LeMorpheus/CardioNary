# Lung Model Modality — Audio vs Chest X-Ray (decision subsequently reversed)

**Question raised:** would a vision CNN on chest X-rays be more practical and more accurate
than a lung-sound audio CNN?

**Decision as recorded at the time: retain audio.** On the evidence then available, chest
radiography did not improve accuracy at the scale available and performed poorly against the
two most heavily weighted criteria.

**This decision was reversed on 2026-08-13.** See the notice below.


> ## Decision status: REVERSED on subsequent evidence (2026-08-13)
>
> **The decision recorded in this document was later overturned.** The delivered system
> runs a **paediatric chest-radiograph model** on the pulmonary node, not lung audio.
>
> The analysis in §1–§6 remains an accurate record of the reasoning applied at the time,
> and is retained for that reason. What it could not account for was a confound
> discovered afterwards during patient-disjoint evaluation:
>
> - ICBHI recordings are **77 % identifiable by recording device alone**.
> - A baseline classifier shown **only the recording device** scored **0.675**, against
>   **0.643** for the lung-audio CNN. The trivial model outperformed the trained one.
> - The published lung-audio accuracies cited in §1 are therefore not safely comparable
>   with our patient-disjoint protocol.
>
> The pulmonary task was moved to the **Kermany paediatric chest-radiograph corpus**,
> where both classes originate from a single institution and a single imaging pipeline,
> eliminating the source confound by construction. The delivered pulmonary node scores
> **115/120 = 95.83 %** on hardware with **98.33 %** agreement against the reference model.
>
> Full evidence: [reports/A1_pipeline_findings.md](../reports/A1_pipeline_findings.md).
> The specification in §5 of this document is, in substance, what was subsequently built.

---

## 1. The accuracy argument — X-ray does not win

This was the whole premise of the question, so it is worth settling with numbers.

### Published 4-class chest X-ray results
**COVID-19 Radiography Database** — 21,165 images; classes: COVID-19, Lung Opacity, Normal,
Viral Pneumonia. The most-cited multi-class results:

| Method | Input | 4-class accuracy |
|---|---|---|
| ResNet50 + SVM | 224×224 RGB | **95.3 %** |
| EfficientNet | 224×224 RGB | **92.8 %** |
| Explainable-AI framework | 224×224 | **93.5 %** |

### Published disease-level lung *audio* results
| Method | Data | Split | Accuracy |
|---|---|---|---|
| CNN + BDLSTM (Fraiwan et al.) | KAUH + ICBHI, 213 patients, 6 classes | **subject-level 10-fold** | **99.62 %** |
| CNN alone (same paper) | same | subject-level 10-fold | **99.04 %** |

**The audio path already reports higher numbers than the X-ray path**, under a *stricter*
evaluation protocol (patient-disjoint).

### And X-ray gets worse on our hardware, not better
Those 95 % X-ray figures come from ResNet50 / EfficientNet at 224×224 — models that are
**one to two orders of magnitude too large** for an EFR32MG26. Published TinyML work is
explicit that even *int8 MobileNetV2 is ~5.3× too large to fit a microcontroller* and needs
compression well beyond quantisation, with quantised MobileNet variants holding accuracy only
**≥ 0.85** in a 1 MB flash / 256 KB SRAM budget.

To fit our board we would run ~96×96 grayscale int8. Realistic expectation after that
downscale: **88–93 %** on the 4-class task — *below* our 95 % target and below the audio path.

> **Conclusion as recorded at the time:** switching to chest radiography would cost accuracy.
> *This conclusion did not survive the device-confound analysis described in the notice above;
> the lung-audio accuracies it relies on are not comparable under a patient-disjoint protocol.*

---

## 2. The practicality argument — X-ray loses harder

| Question a reviewer will ask | Audio | X-ray |
|---|---|---|
| How does data get into the MCU? | A microphone. We own an I2S MEMS mic; the Dev Kit has two ICS-43434s on board. | **There is no answer.** We have no image sensor, and X-rays come from a machine that already has a computer attached. |
| Why does this need to be on a $5 MCU? | Because it is a body-worn / handheld sensor that must work offline, on a coin cell, with no cloud. | It doesn't. A $50,000 X-ray machine has a PC. Running the model on an MCU next to it is theatre. |
| Where is the patient? | Anywhere — clinic, village, home. | In a radiology department, which is exactly the resource-limited thing our product claims to work around. |

The competition topic is **Smart Healthcare with Edge AI**, and the two heaviest criteria are
*Practical applicability* (20 %) and *Live/Video demo* (25–30 %). "Cheap acoustic sensor that
screens where there is no radiologist" is a coherent edge-AI story. "MCU classifies X-rays" is
not — the edge premise collapses under the first question.

---

## 3. The coherence argument

The heart model is auscultation audio. Keeping the lung model on audio means:

- **One product story:** a cardiopulmonary acoustic screening platform.
- **One shared codebase:** `chiron_common` DSP is written once and reused on both nodes. An
  X-ray node would need a completely separate front-end and share almost nothing.
- **One growth path:** both nodes upgrade to live microphone capture in Phase 2 with the same
  work.

Two unrelated modalities on two boards is a weaker story *and* more work.

---

## 4. The one real argument *for* X-ray, and why it is not enough

X-ray input removes the DSP feature-parity problem entirely — no FFT, no mel filterbank, no
fixed-point porting. Feature parity ([11_VERIFICATION_AND_TEST.md §3](11_VERIFICATION_AND_TEST.md))
is the single highest technical risk in the project, and an image pipeline is just
"resize, quantise, feed".

That is genuine. It is outweighed because:
1. We have to solve feature parity **anyway** for the heart model. Solving it once covers both.
2. Fixed-point audio DSP on an MCU *is* the technical challenge (10 % of the score). Image
   resize is not.
3. It buys us a lower accuracy ceiling on hardware we cannot feed with real data.

---

## 5. If we ever do want the X-ray model — the ready-made spec

Keep this on the shelf. Executable in ~2 days if the situation changes.

| Item | Choice |
|---|---|
| Dataset | **COVID-19 Radiography Database** (Kaggle), 21,165 images |
| Classes | 4 — COVID-19, Lung Opacity, Normal, Viral Pneumonia |
| Input | 96×96 grayscale, int8 |
| Model | Depthwise-separable CNN, ~150 k params, same operator whitelist as [06_MODEL_DESIGN.md §1](06_MODEL_DESIGN.md) |
| Expected | 88–93 % on-device |
| Split | **Patient-disjoint if patient IDs are available**; this dataset is assembled from several sources, so also stratify by source and report a source-confusion matrix |
| Golden vectors | 24 held-out images as int8 arrays in flash — same harness as audio |
| Storage | 96×96 = 9,216 B/image → 24 images = 221 kB. Trivial. |

**Assessment recorded at the time of where this would be the right call:** if we pivot the product from a wearable sensor
to a **gateway-side triage assistant on the Raspberry Pi** — the Pi is a legitimate compute
node with a screen, and "rural clinic X-ray triage with no radiologist" is a strong story
*there*. That is a different product, not an upgrade to this one. Not a semi-final decision.

---

## 6. What actually gets us to a defensible high number

Not a modality change — a **task-level** choice, which we already have:

| Rung | Lung task | Expected | Notes |
|---|---|---|---|
| **C (now primary)** | **KAUH-only, 4-class**: Normal / Asthma / Heart Failure / COPD, 97 patients, single device | **95 %+** | Cleanest story, no merge confound, no 4 GB ICBHI download |
| A | KAUH + ICBHI, 6-class, 213 patients | 95–99 % | Better headline, needs the device-confound analysis |
| D | ICBHI event-level (crackle/wheeze/both/normal) | ~65 % | Secondary scientific result only |

Rung C was the plan of record when this document was written. **It was subsequently
abandoned**: the device-confound analysis applies to KAUH as well, and the pulmonary task
moved to chest radiography. See [05_DATASET_SELECTION.md](05_DATASET_SELECTION.md) for the
current specification.

---

## Sources
- [COVID-19 Detection Using Deep Learning on Chest X-ray (PMC8614951)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8614951/)
- [COVID-19 Classification from Chest X-Ray: Explainable AI framework (PMC9284325)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9284325/)
- [Detecting COVID-19 and pneumonia from chest X-rays with deep CNNs (PMC9117408)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9117408/)
- [TinyissimoYOLO — MCU memory limits for quantised MobileNet](https://arxiv.org/pdf/2306.00001)
- [Deploying TinyML for energy-efficient edge AI (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-27818-9)
- [Fraiwan et al., CNN+LSTM pulmonary disease recognition, 99.62 % (PMC8019351)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8019351/)
