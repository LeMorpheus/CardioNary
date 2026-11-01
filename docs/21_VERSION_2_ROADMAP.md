# CardioNary Version 2 — Design Roadmap

**Status:** Forward-looking design document. **Nothing in this file has been built or
measured.** Every V1 figure quoted here is a real measurement with a citation; every V2
figure is labelled **target**, **budget**, or **estimate** and is a thing to go and
measure, not a claim.

**Written:** 2026-08-24, at the close of V1 development.
**Predecessor:** V1 as verified in [reports/A5_hardware_bringup.md](../reports/A5_hardware_bringup.md).

---

## 0. How to read this document

V1 reported only measured results. The lung-audio task was terminated on that basis
([15_LUNG_MODALITY_DECISION.md](15_LUNG_MODALITY_DECISION.md)), and the same standard of
evidence applies throughout this document.

Statements are therefore classified as follows:

| Marker | Meaning |
|---|---|
| **[MEASURED]** | A V1 result on real hardware. Traceable to a report. |
| **[TARGET]** | A V2 goal. Chosen deliberately; not yet achieved. |
| **[BUDGET]** | An engineering allocation derived from a V1 measurement. |
| **[OPEN]** | A question we cannot answer yet and must not pretend to. |

The governing distinction: **V2 is not an incremental improvement on V1. V2 is the
release in which the V1 result is established on a patient.**

---

## 1. The boundary V1 does not cross

V1 is complete and it works. It is also, precisely, a **bench-verified substrate** — and
the difference matters more than any accuracy figure.

Here is the entire V1 data path:

```
golden vector in flash  →  front-end  →  INT8 CNN  →  seal  →  BLE  →  Pi  →  browser
```

Every input V1 has classified was a held-out clip from a public dataset, compiled into the
firmware image. **No sensor has ever fed either model.**
This was a deliberate non-goal, stated up front in [00_PROJECT_BRIEF.md](00_PROJECT_BRIEF.md),
and it removed every analogue, mechanical and biological variable from a system whose
digital path had to be verified first.

What V1 therefore proved, conclusively:

| Question | Answer | Source |
|---|---|---|
| Does INT8 quantisation cost accuracy on this silicon? | No. Agreement **100.00%** cardiac, **98.33%** pulmonary — the chip reproduces the reference model. | **[MEASURED]** [A5](../reports/A5_hardware_bringup.md) |
| Do the models fit? | Yes. Arenas **48,424 B** and **177,248 B** against 512 kB RAM. | **[MEASURED]** [A5](../reports/A5_hardware_bringup.md) |
| Is inference fast enough? | Yes. **57 ms** cardiac, **644 ms** pulmonary. | **[MEASURED]** [A5](../reports/A5_hardware_bringup.md) |
| Does the BLE + gateway + dashboard chain hold up? | Yes, two nodes concurrently, cold boot to connected in **57 s**. | **[MEASURED]** [A5](../reports/A5_hardware_bringup.md) |
| Can we seal results cheaply? | Yes. 28-byte record, **952 bytes** of flash. | **[MEASURED]** [14_PRIVACY_AND_SECURITY.md](14_PRIVACY_AND_SECURITY.md) |

And what V1 did **not** prove — this list is the specification for V2:

1. That the models survive contact with a real chest, a real room, and a real operator.
2. That a screening result correlates with a clinical reference standard on real patients.
3. That the confidence number means anything in a probabilistic sense.
4. That the security model survives leaving the bench.
5. That the pulmonary node has a coherent field deployment path (see §5).

---

## 2. What V2 is

> **V2 is the release in which the system acquires signals from patients.**

Three theses, in priority order:

**Thesis 1 — Acquisition is the whole problem now.** Everything downstream of the ADC is
solved and measured. The remaining risk is concentrated in the ~200 ms between a
diaphragm touching skin and a tensor entering the network. V2 spends most of its budget
there.

**Thesis 2 — Our own data is the only way to retire the confound.** V1 rejected lung audio
because public corpora are contaminated by recording device — a device-only baseline
scored **0.675** against our CNN's **0.643** **[MEASURED]** ([A1](../reports/A1_pipeline_findings.md)).
That confound is a property of *multi-source datasets*, not of lung audio. One device, one
protocol, one site removes it by construction — exactly the argument that justified moving
to Kermany. V2 collects.

**Thesis 3 — A confidence value without a coverage guarantee is not a probability.** The V1
threshold control is a useful operational feature, but a 0.99 softmax output does not denote
99 % probability. V2 replaces the heuristic floor with conformal prediction, which carries
negligible runtime cost and converts abstention into a formal coverage guarantee.

---

## 3. What carries over unchanged

V1 was built as a substrate rather than a demonstration. The middle of the pipeline is
therefore unchanged in V2.

| Component | V1 state | V2 |
|---|---|---|
| TFLM runtime + `ml_model` codegen | Working, both nodes | **Unchanged** |
| INT8 quantisation flow | Agreement verified | **Unchanged** |
| MVP accelerator path | Working (optimizer passes disabled — see below) | **Unchanged** |
| Cardiac CNN topology | 13,509 params, 27.7 kB | **Unchanged**, retrained on new data |
| CMSIS-DSP mel front-end | 201 ms, float32 | **Reused**, now fed live |
| BLE GATT service | Custom service, working | **Extended**, not replaced |
| Seal format | `counter(4) ‖ ct(16) ‖ tag(8)` | **Extended to 32 B**, see §8 |
| Gateway (FastAPI + SQLite + WS) | Working on Pi 4 | **Extended**, same stack |
| Dashboard | 6 languages, case detail | **Extended**, same codebase |
| Derived indicators (murmur phase, attention) | 49/52 physiology match | **Promoted** to load-bearing, see §4.4 |

**Carried-over constraints that must not be quietly dropped:**

- The two `.mlconf` graph-optimizer passes stay **disabled**. Enabling them silently
  corrupted the INT8 model to exactly chance (50.00% at 0.981 confidence). Cost of
  keeping them off: +20 ms, −112 B arena. This trade is accepted and is not revisited
  in V2 without a full agreement re-run.
- `slc generate` never overwrites a destination `config/`. Any V2 build script must
  copy-then-wipe. The tell is ~2 translation units rebuilding instead of ~100.
- The bootloader offset trap ([16_HARDWARE_BRINGUP_RUNBOOK.md](16_HARDWARE_BRINGUP_RUNBOOK.md))
  is re-entered deliberately in V2 §8.4 — this time with secure boot actually configured.

---

## 4. Track A — Live acquisition

The largest track, and the primary determinant of V2 feasibility.

### 4.1 The signal chain

**V1:** `const int8_t vector[]` in flash → mel front-end → CNN.

**V2 [TARGET]:**

```
INMP441 MEMS  →  EUSART-I2S  →  DMA ring buffer  →  DC block  →  decimate 16k→8k
                                                                        ↓
      CNN  ←  mel front-end  ←  window selection  ←  SQI gate  ←  AGC / clip detect
```

Note the MG26 has **no PDM peripheral** — this was a documented correction to the original
Writeups plan. The path is EUSART in I2S mode, same as the Dev Kit's onboard ICS-43434.

### 4.2 The five things stored vectors hide

Each of these is a failure mode that cannot appear in V1 and will appear on day one of V2:

| # | Problem | V2 mitigation |
|---|---|---|
| 1 | **Contact noise** — diaphragm rubbing on skin or clothing produces broadband transients that look like S1 | High-pass + transient rejection in the SQI gate |
| 2 | **Ambient speech** — the clinic is not quiet; speech energy overlaps the murmur band | Spectral flatness + voicing detector in SQI |
| 3 | **Breath contamination** — lung sound bleeds into every cardiac recording | Envelope-periodicity check; prefer end-expiratory windows |
| 4 | **Placement variance** — an operator who misses the mitral area gets a different signal | On-screen placement guide + SQI feedback loop |
| 5 | **Clipping and gain** — MEMS output level varies with coupling pressure | Clip counter; reject and re-record rather than classify |

### 4.3 Signal Quality Index — rejection prior to inference

**This is the highest-value, lowest-cost item in all of V2.**

A screening device that classifies a bad recording is worse than one that says
"record again". V1 already established the pattern — at threshold 0.99 the pulmonary node
declined **114 of 120** cases **[MEASURED]** rather than guess. V2 pushes that decision
*earlier*, to before the model even runs.

**[BUDGET]** SQI must cost < 10 ms and < 4 kB of arena. Justification: the cardiac
front-end already costs 201 ms **[MEASURED]**; a 5% addition is invisible to the operator
and saves a 57 ms inference plus a wasted clinical interaction.

Proposed composite (all cheap, all already available from the existing DSP):

| Sub-score | Signal | Cost |
|---|---|---|
| Clip fraction | samples at full scale / total | negligible |
| SNR proxy | envelope peak-to-trough ratio | reuses murmur DSP |
| Periodicity | autocorrelation peak in 40–200 bpm band | one small FFT |
| Band energy ratio | cardiac band / total | reuses mel bins |
| Stationarity | frame-to-frame energy variance | negligible |

**[TARGET]** SQI gate rejects ≥ 90% of deliberately-bad recordings (rubbed, talked-over,
off-site) while rejecting ≤ 5% of clean ones, measured on a purpose-built adversarial set.

### 4.4 Window selection — promoting the murmur DSP

V1's murmur-phase detector was a *derived indicator*: a bonus output, validated at
**49/52 against physiology** **[MEASURED]** ([18_DERIVED_INDICATORS.md](18_DERIVED_INDICATORS.md)).

In V2 it becomes structural. Live capture produces 10–15 s of audio; the model eats 3 s.
Something must choose. The S1/S2 detection that already powers murmur phasing is exactly
the machinery needed to select whole cardiac cycles with the best SQI — so the feature we
built for the dashboard turns out to be the acquisition system's window selector, at zero
additional code.

**[TARGET]** Multi-window voting: classify 3 non-overlapping windows, aggregate.
**[BUDGET]** 3 × 258 ms = 774 ms total cardiac path, still under 1 s. Derived from the
V1 measurement of 201 ms front-end + 57 ms inference.

### 4.5 The transfer question

**[OPEN]** — and it is the single largest risk in V2.

The cardiac model was trained on Yaseen, a clean curated corpus. Live MEMS-through-
stethoscope audio has different noise, different frequency response, different coupling.
**There is no guarantee the model transfers at all.**

Established practice is to test this *before* committing to it and to publish a negative
result if that is the outcome. Concretely — record 30 subjects on the V2 chain, run the **unmodified V1
model**, and compare against simultaneous reference recordings. If agreement collapses,
that is the finding, and the response is domain adaptation or retraining on our own data,
not quiet threshold tuning.

**Gate A (see §13): do not build the enclosure until transfer is characterised.**

---

## 5. Track B — The pulmonary modality decision

The uncomfortable question, stated plainly.

### 5.1 The tension

V1's pulmonary node runs a chest X-ray model on a microcontroller. It works — **115/120,
agreement 98.33%** **[MEASURED]**. But in the field, a chest radiograph does not come from
a $21 node. It comes from a radiology department that already has a workstation. The
pulmonary node, as delivered in V1, demonstrates that a 412k-parameter vision CNN executes
on an MG26, but the architecture has no corresponding sensor.

V2 must resolve this. Two paths.

### 5.2 Path A — Move CXR up a tier

Keep the model, move it to where images actually exist: the Pi, or a clinic workstation,
ingesting DICOM or a digital photograph of a film. The MG26 pulmonary node is then freed
for a different job.

- **Pro:** Zero model risk. Runs today. Honest about where images come from.
- **Con:** Abandons the "inference on the endpoint" story for half the product.
- **Effort:** Low. The model is already INT8 and portable.

### 5.3 Path B — Own the lung audio problem

Return to lung audio, but with our own data. V1's rejection was specific and it was
**about the dataset, not the modality**: ICBHI is **77% identifiable by recording device
alone** **[MEASURED]**. One device, one protocol, one site eliminates that confound the
same way Kermany eliminated it for imaging.

- **Pro:** Restores a coherent sensor→node→result story. Uses the acquisition chain V2 is
  building anyway. Genuinely novel.
- **Con:** Requires collecting a dataset large enough to train on. That is the long pole.
- **Effort:** High, and gated on §7.

**[OPEN]** How many patients does an honest binary lung-audio model need? Unknown, and the
literature is contaminated by exactly the confound we are avoiding, so it is a poor guide.
**Do not guess this number — derive it from a pilot.**

### 5.4 Recommendation

**Do both, sequenced.** Path A immediately — it is nearly free and it makes V2 shippable.
Path B as the research track running underneath, gated on data volume, with an explicit
willingness to publish "we could not do this honestly" a second time.

The device ships with CXR at the gateway tier and lung audio marked *investigational* and
**not displayed to clinical users** until validated.

---

## 6. Track C — Calibrated uncertainty

### 6.1 What is wrong with the V1 confidence number

Nothing, as a product feature. Everything, as a probability.

The dashboard slider is genuinely good: raise it and the device defers instead of guessing.
But the underlying quantity is a softmax output from a network trained with cross-entropy,
and such networks are systematically overconfident. "0.99" does not mean "wrong one time in
a hundred". The V1 dashboard never claims it does — but a clinician will read it that way.

### 6.2 Conformal prediction

**[TARGET]** Replace the raw floor with split conformal prediction.

The mechanism, briefly: hold out a calibration set, compute a nonconformity score for each
example, take the (1−α) quantile as a threshold. At inference, emit the *set* of classes
whose score falls below it. That set contains the true class with probability ≥ 1−α, under
exchangeability. No retraining. No architectural change.

**On-device cost [BUDGET]:**

| Item | Cost |
|---|---|
| Stored quantile | 1 float, or 1 int8 per class |
| Runtime | one comparison per class — 5 comparisons cardiac |
| Flash | < 64 B |
| Arena | 0 |

This is the best defensibility-per-byte in the entire project. It fits V1's memory
envelope with room to spare and it upgrades every existing claim.

### 6.3 What it changes on screen

| V1 | V2 |
|---|---|
| "Mitral Stenosis — 0.94" | "Mitral Stenosis (90% prediction set: Mitral Stenosis, Mitral Regurgitation)" |
| "UNCERTAIN — below threshold" | "Set includes all 5 classes — recording is uninformative" |

The second row is the important one. A conformal set that contains *everything* is a
mathematically principled way of saying "this told us nothing", which is exactly what the
V1 dashboard's *"what this screen cannot tell you"* panel expresses qualitatively.

---

## 7. Track D — Clinical validation

V1 holds zero patient data and claims zero clinical validity. V2's headline claim, if it
earns one, comes from here.

### 7.1 The metric change

Accuracy is the wrong metric for a screening device and V2 should stop reporting it as the
primary figure.

| Metric | Why it matters here |
|---|---|
| **Sensitivity** | Missing valvular disease is the harm we are trying to prevent |
| **Specificity** | False positives consume scarce echo/referral capacity |
| **PPV at realistic prevalence** | **This is the number that will disappoint, and must be stated anyway** |
| **Inter-rater agreement** | The clinician reference is itself noisy; κ bounds what we can claim |

**The prevalence trap, worked:** a test at 95% sensitivity and 95% specificity, applied to a
population with 2% disease prevalence, yields a PPV of roughly **28%** — nearly three in four
positives are false. This is arithmetic, not pessimism, and it applies to every screening
device ever built. V2 must present it in the writeup *before* a reviewer presents it for us.

### 7.2 Study shape [TARGET]

- **Design:** prospective, paired. Device and clinician assess the same patient in the same
  session. Blinded — the clinician does not see the device output.
- **Reference standard:** echocardiography where available for valvular disease; consultant
  auscultation otherwise, with the limitation stated.
- **Sample size:** derive from the pilot's observed sensitivity and the width of the
  confidence interval we are willing to publish. **[OPEN]** — do not pre-commit a number.
- **Pre-registration:** write the analysis plan before collecting. This is the cheapest
  possible protection against the appearance of fishing.

### 7.3 What we may claim, and when

| Evidence held | Permitted claim |
|---|---|
| Bench agreement only *(V1 today)* | "The chip reproduces the reference model." Nothing about patients. |
| Transfer study on healthy volunteers | "Signal acquisition is viable." Still nothing diagnostic. |
| Pilot, n small, unblinded | "Preliminary, hypothesis-generating." Confidence intervals mandatory. |
| Powered blinded study | Sensitivity/specificity with CIs, and the PPV caveat. |

**The V1 posture is not relaxed by any of this.** "A screening aid, not a diagnosis. Every
result is for a clinician to review." That sentence stays in V2 regardless of how good the
numbers get.

---

## 8. Track E — Security for a fleet

### 8.1 Honest assessment of V1 crypto

V1's crypto is real, it works on the wire, and it cost 952 bytes. It is also a *demo*
security model, and V2 must say why rather than inherit it silently:

| V1 property | Why it does not deploy |
|---|---|
| Single 16-byte PSK, same on every node and the gateway | One extraction compromises the entire fleet |
| Key compiled into `chiron_secret.h` in flash | Readable over SWD while the debug port is open |
| Per-boot counter | Replay across a power cycle is possible |
| **CMAC for model integrity** | **Symmetric — anything that can verify can also forge a valid model tag** |
| No key rotation | A compromise is permanent |

The fourth row is the sharpest. The device currently holds a secret capable of *minting*
valid model signatures. That is acceptable for a build-time integrity pin; it is not a
provenance mechanism.

### 8.2 V2 key hierarchy [TARGET]

```
Factory master key (HSM, never on device)
        ↓ KDF over device unique ID, at provisioning
Per-device identity key  ──────────────┐
        ↓ ECDH at pairing              │
Session keys (rotate per session)      │
        ↓                              ↓
   AES-CTR + CMAC              attestation to gateway
```

Consequence: compromising one node compromises one node.

### 8.3 Model provenance: CMAC → signature [TARGET]

Replace the symmetric model tag with **ECDSA P-256 verification** against a public key in
device flash. The private key never leaves the build server. The device can verify a model
is ours; it cannot forge one.

**[BUDGET]** ~1.5–3 kB additional flash for verification, and a one-time boot cost of tens
of milliseconds. Against a 3200 kB part with the pulmonary model occupying 610 kB
**[MEASURED]**, this is comfortably affordable. Both PSA Crypto and the radio stack's
existing primitives are candidates — **[OPEN]** confirm which are available without
pulling in a second crypto backend.

### 8.4 Secure boot — re-entering the trap deliberately

V1 removed `in_place_ota_dfu` and `bootloader_interface` from both `.slcp` files because a
bootloader-offset image flashed and verified clean but never booted. That was the correct
emergency fix and the diagnosis (reading the vector table at `0x08000000`) is in the runbook.

V2 must bring the bootloader back — secure boot and signed OTA are non-negotiable for a
fleet — and it must do so with the offset, the linker script and the bootloader image
configured together rather than one at a time. **Budget real time for this.** It has already
cost the project once.

### 8.5 Debug lock and Secure Vault

**[OPEN] — verify before designing to it.** The EFR32MG26 family offers Secure Vault at
different tiers, and tier is an **OPN-level** property. The project has already been bitten
by exactly this class of assumption: the MVP accelerator was documented as xG24/xG28-only in
one workshop deck, when in fact it is an OPN-level option that our `…B510…` parts happen to
carry.

**Do not assume our OPN has Secure Vault High.** Read the ordering information for
`EFR32MG26B510F3200IM68` and `…IM48` and design to what is actually there.

### 8.6 Data protection

V2 stores patient-linked records; V1 stored none. That is a categorical change in
obligation, not an incremental one.

- **[TARGET]** Encryption at rest on the gateway, not just in transit.
- **[TARGET]** Append-only audit log — who viewed which record, when.
- **[TARGET]** Configurable retention and a real deletion path.
- **[TARGET]** Pseudonymisation by default; the identity mapping lives separately.
- **[OPEN]** India's DPDP Act 2023 obligations for health data. **Get advice; do not
  improvise this from a blog post.**

---

## 9. Track F — Form factor and power

### 9.1 Two product shapes, and why V2 picks one

The Writeups vision is a wearable chest patch: hydrogel coupling, coin cell, continuous
monitoring. It is a good vision and it is **not V2**.

| | Clinic screening station | Wearable patch |
|---|---|---|
| Interaction | ~30 s, operator-held | Continuous, hours |
| Power | USB / mains | Coin cell — the hard constraint |
| Coupling | Standard stethoscope chest piece | Hydrogel, unsolved |
| Mechanical | Enclosure | Skin adhesion, comfort, biocompatibility |
| Distance from V1 | **Short — it is V1 with a microphone** | Long |

**V2 is the screening station. The patch is V3.** This follows V1's own governing rule:
a complete working system first, upgrades second. Attempting the patch in V2 imports
adhesion, biocompatibility and coin-cell power into a release that has not yet proven it
can hear a heartbeat.

### 9.2 Power [BUDGET — all figures to be measured, none claimed]

The screening station's power problem is *duty cycle*, not average draw. A 30-second
interaction dominated by a 201 ms front-end and a 57 ms inference **[MEASURED]** means the
radio and the idle state, not the CNN, set battery life.

Measurement plan, in priority order:

1. Profile actual current in each state on the kit's energy profiler — advertising,
   connected-idle, capture, inference, transmit.
2. Only then compute a battery figure.
3. Publish the profile, not just the headline number.

**Do not put a battery-life number in any deck until step 1 is done.** An unmeasured
power claim is exactly the kind of number this project has spent V1 refusing to make.

### 9.3 Consolidation

**[TARGET]** One custom board carrying the MG26, the MEMS front-end, and the power path —
replacing two dev kits. **[OPEN]** Whether the pulmonary node remains a separate MCU
depends entirely on §5's resolution.

---

## 10. Track G — Gateway, fleet, and records

| Concern | V1 | V2 [TARGET] |
|---|---|---|
| Nodes | 2, hardcoded by BLE name prefix | N, discovered and enrolled |
| Storage | SQLite, results only | Patient-linked records, encrypted at rest |
| Sync | None — single Pi | Offline-first, reconciles when connectivity appears |
| Identity | None | Operator login, role-based access |
| Audit | Event log | Append-only, tamper-evident |
| Model versioning | **`model_version` / `build_id` currently empty** | **Mandatory** — see below |
| Languages | 6 | Extend as deployment demands |

### 10.1 The model-version gap is now load-bearing

V1's `NodeInfo` reports empty `model_version` and `build_id`. In V1 this was cosmetic — a
known, documented gap.

In V2 it is a **correctness requirement**. A stored patient record whose model version is
unknown cannot be audited, cannot be re-analysed when a model is found to be faulty, and
cannot support a recall. Every V2 result must carry:

```
model_version ‖ build_id ‖ calibration_set_id ‖ SQI ‖ conformal_alpha
```

This is a small change that cannot be retrofitted once records exist, and is therefore
scheduled early.

### 10.2 Dashboard evolution

The V1 dashboard's design principles were chosen with clinical staff in mind and they hold:
full disease names, never abbreviations; no ratings or star scales; numbers where numbers
help; no emojis; per-case detail with urgency and next actions; six languages.

V2 adds, without disturbing them:

- **Signal quality shown next to every result** — the operator learns to record better.
- **Prediction sets** rather than a single label plus a number (§6.3).
- **Longitudinal view** — the same patient across visits, where a screening device provides
  the greatest clinical value.
- **Referral tracking** — did the positive get an echo? Without this the device cannot
  demonstrate outcome impact.

---

## 11. Regulatory posture

**[OPEN] throughout. These are the questions to put to someone qualified, not answers.**

- India: CDSCO, Medical Device Rules 2017. Software producing a disease-associated output
  is likely to attract classification. **Establish the class before designing the label.**
- The V1 stance — screening aid, clinician review, no diagnostic claim — is the correct
  posture and is also the posture with the lightest regulatory burden. **Drift here is
  expensive.** Every feature that nudges the output toward "diagnosis" should be
  interrogated for what it costs in classification.
- Quality system, design history file, traceability from requirement to test: V1 already has
  an unusually good documentation trail. **That is a genuine asset — keep the habit.**

---

## 12. Explicit non-goals for V2

V1's non-goals list was one of its most useful documents. Continuing it:

1. **No autonomous diagnosis.** The output is always for clinician review.
2. **No cloud inference.** Edge execution is the architecture, not an optimisation.
3. **No on-device training.** Per-site normalisation statistics are acceptable; gradient
   updates on a medical device are not.
4. **No wearable patch.** That is V3 (§9.1).
5. **No new disease classes on public multi-source data.** The confound argument that
   killed lung audio applies to every future class. Our data or no class.
6. **No clinical claim before the study reads out.** Not "early results suggest". Nothing.
7. **No accuracy figure as the headline metric.** Sensitivity, specificity, PPV at
   prevalence (§7.1).
8. **No re-enabling the graph optimizer passes** without a full agreement re-run.

---

## 13. Phased plan with gates

V1's Track A / Track B split worked because no Track B item started until Track A was
green. The same structure is retained.

### Phase 1 — Acquisition viability

- I2S capture on the MG26, DMA ring buffer, decimation
- SQI implementation and adversarial test set
- Window selection from the existing S1/S2 detector
- **Transfer study: unmodified V1 cardiac model on live audio, ~30 volunteers**

> **Gate A — does the V1 model work on live audio at all?**
> If agreement against the reference chain is poor, stop and characterise before
> proceeding. This gate exists to be failed honestly.

### Phase 2 — Defensibility

- Conformal calibration (§6)
- Model version / build ID plumbed end to end (§10.1)
- Per-device keys and ECDSA model provenance (§8.2, §8.3)
- Secure boot restored correctly (§8.4)

> **Gate B — can every result on screen be traced to a model, a calibration set, and a
> signal quality score?** No patient data is collected before this gate.

### Phase 3 — Clinical pilot

- Ethics approval, pre-registered analysis plan
- Paired blinded collection
- Pulmonary Path A shipped; Path B data accumulating, not displayed (§5.4)

> **Gate C — does the pilot support a powered study, and is the observed effect large
> enough to be worth one?**

### Phase 4 — Consolidation

- Custom board, measured power profile
- Fleet gateway, records, audit
- Powered validation study

**Note the ordering.** Enclosure and custom hardware come *last*. This is deliberate and it
is the same reasoning that made V1 use dev kits: do not commit to physical form until the
signal chain is characterised. Mechanical iterations are slow and expensive; firmware
iterations are not.

---

## 14. Risk register

| # | Risk | Severity | Signal it is happening | Response |
|---|---|---|---|---|
| 1 | **Live audio does not transfer** | **Critical** | Agreement collapses at Gate A | Domain adaptation, or retrain on collected data. Publish the negative result. |
| 2 | Own lung-audio dataset too small to train honestly | High | Pilot yields few positives | Ship Path A only; say so plainly (§5.4) |
| 3 | PPV at real prevalence disappoints | High | Pilot PPV low despite good sens/spec | **Expected, not a surprise.** Present the arithmetic proactively (§7.1) |
| 4 | Secure boot re-integration burns the schedule again | Medium | Board flashes clean, does not boot | Runbook has the diagnosis. Budget the time up front (§8.4) |
| 5 | Secure Vault tier is lower than assumed | Medium | Ordering info contradicts the design | **Check first** (§8.5) — this is why it is [OPEN] |
| 6 | Regulatory class higher than expected | Medium | Advice says Class C | Re-scope the output, not the claim |
| 7 | Scope creep toward the wearable patch | Medium | Enclosure discussions before Gate A | §12 item 4 exists for this |
| 8 | Ethics/approval timeline exceeds the project window | Medium | Approval pending at Phase 3 | Start the paperwork during Phase 1 |

---

## 15. Cost model

| | V1 **[MEASURED]** | V2 **[ESTIMATE]** |
|---|---|---|
| BOM per node, at volume | ₹1,799.00 at 1,000 units | **Costed** — itemised in [24_BILL_OF_MATERIALS.md](24_BILL_OF_MATERIALS.md). MEMS, custom board and enclosure are included; the MG26 module is 34% of it and is not the cost driver |
| Subscription | None | None — architectural, not a pricing choice |
| Per-study fee | None | None |
| Connectivity required | None | None |

The zero-marginal-cost story survives V2 intact, and it survives *because* inference is on
the endpoint. That is the commercial argument for the architecture and it should be stated
as such: there is no per-inference cost to pass on, because there is no per-inference cost.

**Do not put a V2 BOM figure in a deck until a real board is quoted.**

---

## 16. What would make us stop

A roadmap requires explicit termination criteria. V1 terminated its own lung-audio task on
evidence; V2 applies the same standard.

**Stop, or fundamentally re-scope, if:**

- Live acquisition cannot reach a signal quality where the models transfer, and domain
  adaptation on realistic data volumes does not close it.
- The pilot shows sensitivity too low to be clinically useful at any operating point on
  the ROC curve.
- The regulatory class makes an academic project non-viable, and no re-scoping of the
  output preserves clinical usefulness.
- We cannot collect data under an ethics framework we are comfortable defending in public.

Three of the four are research findings rather than engineering failures, and are
publishable as such.

---

## 17. Summary — the one-paragraph version

V1 proved that two INT8 CNNs run correctly on EFR32MG26 silicon, that quantisation costs
nothing measurable (**100.00%** and **98.33%** agreement), that the BLE and dashboard chain
holds, and that results can be sealed for 952 bytes of flash. It proved all of this on
stored vectors, because that removed every variable that was not digital. V2 puts the
variables back, in a deliberate order: first a microphone and a signal-quality gate that
rejects unusable recordings, then calibrated uncertainty that converts abstention into a
coverage guarantee rather than a heuristic, then per-device keys and signed models so the
security model survives deployment, and only then patients, a pre-registered study, and a custom
board. The middle of the pipeline — the models, the runtime, the seal, the gateway, the
dashboard — is unchanged, a direct consequence of V1 having been built as a substrate
rather than a demonstration.

---

## Appendix — open questions to resolve before committing

| # | Question | Blocks | Owner |
|---|---|---|---|
| 1 | Does our OPN carry Secure Vault High? | §8.5, key storage design | Firmware |
| 2 | Does the V1 cardiac model transfer to live MEMS audio? | **Everything** (Gate A) | Signal |
| 3 | What CDSCO class does the output attract? | Labelling, claims, §11 | External advice |
| 4 | Are ECDSA verify primitives available without a second crypto backend? | §8.3 | Firmware |
| 5 | How many subjects for an honest lung-audio pilot? | §5.3 Path B | Study design |
| 6 | Measured current in each device state? | §9.2, any battery claim | Hardware |
| 7 | Is echocardiography available as a reference standard, and for what fraction? | §7.2 | Clinical |
| 8 | DPDP Act obligations for stored health records? | §8.6 | External advice |

---

## Related documents

| Document | Relevance to V2 |
|---|---|
| [00_PROJECT_BRIEF.md](00_PROJECT_BRIEF.md) | V1 scope and the non-goals V2 inherits |
| [reports/A5_hardware_bringup.md](../reports/A5_hardware_bringup.md) | Every V1 measurement quoted here |
| [reports/A1_pipeline_findings.md](../reports/A1_pipeline_findings.md) | The device-confound evidence underpinning §5 |
| [15_LUNG_MODALITY_DECISION.md](15_LUNG_MODALITY_DECISION.md) | Why lung audio was rejected, and the conditions to revisit it |
| [16_HARDWARE_BRINGUP_RUNBOOK.md](16_HARDWARE_BRINGUP_RUNBOOK.md) | The bootloader trap V2 §8.4 re-enters deliberately |
| [17_HOSPITAL_DATA_COLLECTION.md](17_HOSPITAL_DATA_COLLECTION.md) | Protocol groundwork for §7 |
| [18_DERIVED_INDICATORS.md](18_DERIVED_INDICATORS.md) | The murmur DSP promoted to window selection in §4.4 |
| [14_PRIVACY_AND_SECURITY.md](14_PRIVACY_AND_SECURITY.md) | V1 crypto, honestly assessed in §8.1 |
| [13_REAL_WORLD_VALIDATION.md](13_REAL_WORLD_VALIDATION.md) | V1's own limitations analysis |
| [09_DASHBOARD_SPEC.md](09_DASHBOARD_SPEC.md) | The design principles §10.2 preserves |

---

*CardioNary — Let Yourself Heal Now.*
