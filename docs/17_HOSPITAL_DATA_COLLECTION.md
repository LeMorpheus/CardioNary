# Hospital Data Collection Protocol — Primary Health Centre

> ## Status — 2026-08-24
>
> **The system is finished and running**, which changes what this visit is for.
> The clinician is not asked to imagine a device; a working one is placed in front of them.
>
> | | |
> |---|---|
> | Cardiac node (BRD2608A) | **53/53 = 100%** on held-out cases, 57 ms per case |
> | Pulmonary node (BRD2709A) | **115/120 = 95.83%**, 644 ms per case |
> | Gateway | Raspberry Pi, cold-boots to a working dashboard in **57 s** |
> | Interface | Clinical case view in **six languages** |
>
> **Take the demo.** It runs from a portable wireless access point and a power bank — two boards,
> a gateway, and any browser on the network. A doctor who has pressed the button asks
> far better questions than one who has heard a description, and the questions in
> §10 are worth more after he has seen it work.
>
> ### What is new since this protocol was written
>
> **The case view now shows derived indicators**, not just a class — and this is
> the part worth showing a clinician, because it is checkable:
>
> - **Murmur detected, and its timing in the cardiac cycle.** Measured by signal
>   processing, not by the network. It matches physiology on **49 of 52** cases:
>   aortic stenosis 10/10 systolic, mitral regurgitation 11/11 systolic, mitral
>   stenosis 7/9 **diastolic**, normal 11/11 none.
> - Murmur duration, relative intensity, dominant frequency, S1/S2 ratio,
>   systole/diastole timing, heart rate, rhythm regularity.
> - For the X-ray: **which region of the image drove the result** (mid-lung in
>   43/60 pneumonia cases vs upper zones in 36/60 normals).
>
> **Ask him to check these against his own ear.** If he listens to a case and
> agrees the murmur is systolic, that is external validation costing one minute.
> If he disagrees, that is more valuable still.
>
> **What we deliberately do NOT claim** — worth stating before he asks, because
> it builds more credibility than any accuracy figure: severity or valve area
> from sound; cardiomegaly, pleural effusion, pneumothorax or oedema from the
> radiograph. Full reasoning in
> [18_DERIVED_INDICATORS.md](18_DERIVED_INDICATORS.md).
>
> ### Two questions to add to §10
>
> - *"The device says this murmur is systolic and occupies 41% of systole — does
>   that match what is heard?"* (direct validation of a measured output)
> - *"Would a device that reports **not conclusive** rather than guessing be more
>   useful, or less?"* (demonstrate it live: raise the threshold to 0.99
>   and watch 114 of 120 become not conclusive)
>
> ### Timing
>
> Development closes **24 August**, semi-final **25 August**. Data collected now
> cannot retrain anything in time, and should not try to. **Do not take the demo
> hardware to a hospital before the 25th** — losing or damaging it days before
> the semi-final would jeopardise the demonstration. After 25 August, all hardware is available.

**Purpose:** collect real-world clinical data to measure **where our models fail**, not to prove
they work. A documented failure analysis is worth more to this project than a flattering number,
and it is the strongest available answer to the question *"does this survive outside the training
dataset?"*

> Note: **This is not a clinical study and produces no diagnosis for any patient.** Nothing recorded
> here affects anyone's care. Every recording is a research artefact for engineering evaluation.

---

## 1. Collection priorities

Our two models are **different modalities**, and this changes the priority order completely.

| Priority | What to collect | Which model it tests | Why |
|---|---|---|---|
| **1 — highest** | **Anonymised chest X-rays** | **Pulmo node — directly** | Our pulmo model *is* a chest X-ray classifier. This is a true test of a deployed model. |
| **2** | **Heart sound audio** | **Cardio node — directly** | Our cardio model *is* an audio model (Yaseen, 5-class). |
| **3** | Lung sound audio | *No current model* | Valuable for future development, but it cannot test any currently deployed model. |

**Do not let the microphone become the point of the visit.** If the visit yields only X-rays and a
good conversation with the doctor, the trip was a success.

### One caveat that must be stated up front

Our pulmo model was trained on **Kermany paediatric chest X-rays (ages 1–5)**. Adult chest X-rays
are **out of distribution** and the model will very likely degrade badly on them.

That is not a reason to skip them — a measured, honest degradation curve is exactly the kind of
result that earns credibility. But if the PHC sees **paediatric** chest cases, those are the
direct, like-for-like test and are worth far more.

---

## 2. Ethics, consent and privacy — non-negotiable

Do this part properly or do not collect at all. It also happens to be what a serious technical review requires
about.

### Before any recording

- [ ] **Written permission from the Chief Medical Officer**, stating what may be collected, from
      whom, and what may be done with it. Email is fine if signed/acknowledged.
- [ ] Ask the CMO **whether the facility has an ethics committee (IEC)** and whether one is
      required. A student engineering evaluation with fully anonymised data is usually handled
      under institutional permission — but **the CMO decides, not the visiting team.** If there is any intention to
      *publish*, say so now, because that generally does require formal IEC approval.
- [ ] **Written informed consent from every patient**, in the language they actually speak
      (Hindi/local), not English-only. For anyone under 18: **guardian consent plus the child's
      assent.**
- [ ] Relevant law: **India's DPDP Act 2023** treats health data as personal data. Follow the
      CMO's guidance on handling and retention.

### What the consent must say, in plain language

1. What is being recorded and approximately how long it takes
2. That it is **research on a student engineering project**, not part of their treatment
3. That **their care is identical whether they agree or refuse**
4. That they can **stop at any time**, without giving a reason
5. That the recording is **anonymous** — no name, no hospital number, no date of birth
6. That it produces **no diagnosis** and they will not receive a result from it
7. Who to contact to have their data deleted

### Anonymisation rules — apply at the point of capture, never afterwards

- **Never** put a name, medical record number, telephone number or date of birth in a filename, folder or metadata field.
- Use a **pseudonymous ID only**: `P001`, `P002`, …
- Record an **age band** (`0-5`, `6-17`, `18-40`, `41-65`, `65+`), never an exact DOB.
- If a re-identification key is unavoidable, it stays **on paper, at the clinic, with the CMO** —
  never in the project repository, never on the working machine, never in cloud storage.
- **Never commit clinical data to the git repository.** Add it to `.gitignore` before departure.

### Managing speech — a commonly overlooked failure mode

A chest recording will happily capture a conversation, and **a voice is identifying data.**

- **Total silence during recording.** Operator, clinician, patient and attendant. Say this out
  loud before each take.
- **Announce the patient ID to the logbook *before* recording begins — never into the microphone.**
- If anyone speaks mid-take, **discard it and redo it.** Do not "trim it later"; the detail will be lost.
- Before any file leaves the clinic, **listen to every recording end to end** and delete any with
  intelligible speech.
- Never say the patient's name aloud in the room while recording.

---

## 3. Equipment

### Already available

| Item | Note |
|---|---|
| Raspberry Pi 4 + 64 GB SD + official PSU | recording host |
| **I2S MEMS microphone breakout** | **Check what's printed on it first** — document 02 lists one as unused. If it is an INMP441 or SPH0645, **no microphone purchase is required.** |
| MicroSD card reader, USB cables | |
| ESD mat, tweezers | |

### Required purchases

| Item | Why | Approx |
|---|---|---|
| **Cheap acoustic stethoscope ×2** | **The most important item on this list.** Its chest piece couples chest-wall vibration into the mic and rejects room noise. **Buy a dual-head (bell + diaphragm) model — see below.** Two: one to cut, one kept intact as backup. | ₹200–500 ea |
| INMP441 module (×2 if buying) | Buy the **bare module**, not the MAX98357A combo — that amplifier is a speaker driver and is useless for recording. Two units are specified, because a single failed module ends a session that cannot be repeated. | ₹150–300 ea |
| Female–female jumper wires | Pi header → mic | ₹50 |
| **Power bank (5 V, ≥3 A)** | The Pi must run untethered in a clinic | ₹800+ |
| Spare microSD | See §9 — do **not** take the demo card into the field | ₹400 |

### Stethoscope selection: a **bell** is required

Ignore the marketing about ear tips and binaural angles — we are cutting those off. **Only the
chest piece matters**, and one feature decides it:

- The **diaphragm** (flat side) is a stretched membrane that *deliberately attenuates low
  frequencies* to emphasise higher-pitched sounds like breath sounds.
- The **bell** (open cup, usually with a non-chill rim) transmits **low frequencies** — which is
  where heart sounds live: S1/S2 fundamentals at 20–150 Hz, S3/S4, and the murmur of mitral
  stenosis.

Our INMP441 already rolls off below ~60 Hz. Pairing it with a diaphragm-only stethoscope
**attenuates the same band twice**, in exactly our weakest region. A **dual-head chest piece
(bell + diaphragm, rotating)** lets us record both and compare — the comparison is itself a
result worth reporting.

So: **reject single-head "diaphragm only" models for cardiac work.** A product listing that
mentions a *"non-chill rim on the bell"* has a bell; one that says *"single side diaphragm"* does
not, regardless of any "dual frequency" claim in the same description.

### Stated limitation of the INMP441

Its response rolls off below **~60 Hz**. Heart sounds S1/S2 carry strong energy at **20–150 Hz**,
and our cardio front-end band-passes 20–800 Hz — so **we lose the bottom of our own band.**

It remains usable: murmurs (the actual pathology signal for AS/MS/MR/MVP) sit higher, around
100–600 Hz. **Record this limitation in the writeup as a known constraint**, and note that a piezo
contact mic or an electret would reach lower if this becomes the bottleneck.

---

## 3b. Equipment the clinic may already hold — establish before construction

**Call ahead and ask.** Some of this would replace our DIY rig outright, and all of it is free.

| Equipment | Likelihood at a PHC | What it would mean for us |
|---|---|---|
| **Acoustic stethoscopes** | **Certain** | Clinical-grade chest piece — better acoustics than a ₹300 one. See the non-destructive coupling below. |
| **Pulse oximeter** | **Very likely** (post-COVID) | **Free ground truth for heart rate** — see below, this is genuinely useful |
| **ECG machine** | **Likely** — 12-lead is standard under Indian Public Health Standards | Gold-standard cardiac timing; not used by our model, but excellent future work |
| **X-ray: digital (CR/DR) vs film** | **They have a machine — the question is the format** | **The single most important question of the visit.** Digital → anonymised DICOM/JPEG export. Film → photograph it, with measurable quality loss. |
| **Telemedicine kit** (eSanjeevani / ReMeDi / similar) | **Possible — worth asking** | **These kits often bundle a digital stethoscope.** If one exists here, it beats everything we would build. |
| **Digital/electronic stethoscope** (Littmann 3200/CORE, Eko) | **Unlikely standalone** — ₹15k–40k | Would replace our entire recording chain. Ask anyway; donations and CSR programmes do put these in PHCs. |
| **Desktop PC for records** | Likely | **Do not use it** — see §3c |

### The two highest-value asks

**1. "Do you have a telemedicine kit, and does it include a digital stethoscope?"**
Under Ayushman Bharat / eSanjeevani, some centres received kits containing a digital stethoscope,
ECG and other peripherals. If one is sitting in a cupboard here, it records clinical-quality audio
straight to a file and our coupling problem disappears. Ask what **format** it exports — some are
proprietary; Littmann's software exports WAV.

**2. "Is your X-ray digital or film, and can it export anonymised images?"**
This decides whether the highest-value data (§1) is a five-minute export or a photography session.

### Zero-cost ground truth: the pulse oximeter

Clip it on **during** each cardiac recording and write its heart rate into `recordings.csv`.

Then count cardiac cycles in our audio and compare. If our recording says 78 bpm and the oximeter
says 76, the chain is capturing real cardiac events. If they disagree wildly, we are recording
noise — **and we find that out in the clinic while we can still fix it**, rather than a week later
at home. Costs nothing and takes no extra time.

### Non-destructive use of clinic-owned stethoscopes

A clinic-owned stethoscope outperforms any low-cost substitute, and a clinician's personal
instrument is better still. **Do not cut it.** Instead:

1. Remove **one ear tip**.
2. Slip a short length of silicone tubing over the bare ear tube.
3. Connect that to the microphone chamber (§4).

Fully reversible, nothing damaged, and a clinical chest piece is obtained. The other ear tube can be
pinched closed, or left open — try both, it changes the acoustics.

**A low-cost stethoscope is still purchased.** Construction and rehearsal use that unit (§8). Turning up with nothing
and hoping to borrow is how a session gets wasted.

---

## 3b-ii. Borrowing from college labs

Two categories, and they serve different purposes. **Check the borrowing policy first** — many
labs will not let equipment leave campus, which rules out most bench instruments for the clinic
but not for calibration.

### Portable equipment — potential replacement for the purpose-built rig

| Equipment | Where to look | Why it's good |
|---|---|---|
| **Handheld field recorder** — Zoom H1n/H4n, Tascam DR-05 | **Media / AV / film club, journalism dept** | **The best possible borrow.** 24-bit/96 kHz WAV straight to SD, battery powered, 3.5 mm mic input, no drivers, no setup. Plug in a mic, press record. |
| **USB audio interface** — Focusrite Scarlett Solo/2i2, Behringer UMC22 | Media lab, music room, language lab | Clean 24-bit capture to a host machine via Audacity, phantom power for a condenser microphone. Removes all DIY risk. |
| **Condenser / lapel microphone** | Same labs | Pair with either of the above; usually better low-frequency response than the INMP441 |
| **Piezo disc / buzzer element** | Electronics component store | Nearly free **contact microphone**. Responds to chest-wall motion rather than air, so it reaches *below* 60 Hz — precisely where the INMP441 fails. Worth trying even as a curiosity. |

### Fixed-location equipment — for chain characterisation

| Equipment | Where | Use |
|---|---|---|
| **Function generator + small speaker** | Electronics / PCB lab | **Measure the actual frequency response of the stethoscope-and-microphone chain.** Sweep 20 Hz–2 kHz, record the output, plot the curve. |
| **Sound level meter (SPL)** | Instrumentation / environmental lab | Document the clinic's ambient noise floor in dBA — a real number for the writeup |
| **Piezo accelerometer + charge amplifier** | **Vibration / mechanical lab** | Technically the *right* sensor for heart sounds; real phonocardiography research uses accelerometers. Needs a charge amp and DAQ, so it's a campus experiment, not a clinic one. |
| **NI myDAQ / USB-6009 DAQ** | Instrumentation, measurements lab | 200 kS/s analog input; clean capture with LabVIEW or MATLAB |
| **DSP kit with audio codec** — TMS320C6713 DSK (AIC23) | Signal processing / communication lab | Good 24-bit codec, but an awkward workflow for plain recording |
| **DSO** | Any electronics lab | Poor for 20–30 s continuous capture (memory depth), but effective for *verifying* that the microphone produces a plausible signal |

### The characterisation experiment is worth doing regardless

With a function generator and a small speaker a qualitative claim becomes a measurement:

1. Sweep 20 Hz → 2 kHz through the speaker at constant amplitude
2. Record through the **complete** chain — stethoscope chest piece, tubing, microphone, gateway
3. Plot magnitude vs frequency

That converts *"the INMP441 rolls off below 60 Hz"* (a datasheet claim about one component) into
**"our acquisition chain measures −X dB at 40 Hz"** — a measured property of the actual system,
including whatever the stethoscope chamber does to it.

**This is a genuine technical contribution and costs one afternoon.** It also directly answers the
sharpest question a technical reviewer can ask: *"how is it established that the acquisition
hardware can capture what the model requires?"*

---

## 3c. Prohibition on storage using hospital equipment

Even if they offer a PC:

- Patient audio on their machine creates a **data-governance problem for them**, not solely for the visiting team.
- Do not plug project USB drives into clinical systems, or theirs into the working machine.
- Keep the entire chain on **project-owned equipment**, anonymised at capture (§2).

The clean story — *"nothing identifying ever left the room, and nothing belonging to the team touched clinic
systems"* — is materially easier to defend to the CMO and under technical review.

---

## 4. Building the acoustic coupling — performed off-site, not at the clinic

The mic must sit in a sealed air column behind the stethoscope diaphragm.

1. Cut the stethoscope tubing about **8–10 cm** from the chest piece. Keep the chest piece and
   that stub of tube; discard the earpieces.
2. Seat the INMP441 so its **port hole faces into the tube bore**, and seal the joint with tape,
   hot glue or Blu-Tack. **Any air leak destroys the low frequencies** — this is the single most
   common mistake.
3. Strain-relieve the wires so tugging the cable cannot move the mic inside the chamber.
4. **Test before the visit** (§8). If the operator's own heartbeat is not clearly audible under
   controlled conditions, nothing usable will be obtained in a clinical environment.

### Wiring — INMP441 → Raspberry Pi 4

| INMP441 | Pi 4 physical pin | Function |
|---|---|---|
| VDD | 1 (3.3 V) | power — **3.3 V, never 5 V** |
| GND | 6 | ground |
| SCK | 12 (GPIO18) | bit clock |
| WS | 35 (GPIO19) | word select |
| SD | 38 (GPIO20) | data out |
| L/R | 6 (GND) | selects the left channel |

Enable I2S in `/boot/firmware/config.txt`:

```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

> The `googlevoicehat-soundcard` overlay is the most commonly working capture path for an INMP441
> on current Raspberry Pi OS, but **treat it as unverified until we test it together** — the
> correct overlay varies by kernel version. We will confirm this before the visit, not during it.

---

## 5. Recording chain

### Primary — Raspberry Pi + I2S mic

Why not a phone as primary: Consumer handsets apply **automatic gain control and noise suppression** to
microphone input, and both actively destroy the low-frequency content we need. The I2S path has
no such processing.

- Record at the mic's native **48 kHz, 24-bit**, mono.
- **Never record at 8 kHz.** Keep the original and downsample offline — it is not possible to recover what was not captured, and the archive should outlive this model.
- Format: **WAV**. Never MP3 — lossy compression mangles exactly the low-amplitude detail here.

**Run it headless with a button and an LED** so no screen is required in a consulting room:
press once to record a fixed duration, LED illuminated while recording, file automatically
named. This script is written and tested before the visit rather than on the day.

### Backup — handset voice recorder

Always have this ready. Press the stethoscope chest piece against the handset microphone and use an app
that can record **WAV with processing disabled** (look for an "unprocessed"/"raw" source option).
Quality is worse, but a mediocre recording beats a failed rig and no data at all.

---

## 6. Where to place the microphone

### Cardiac — 5 sites

Our cardio model's classes are Normal, **AS** (aortic stenosis), **MS** (mitral stenosis),
**MR** (mitral regurgitation), **MVP** (mitral valve prolapse).

| Code | Site | Anatomy | Why |
|---|---|---|---|
| `AORT` | Aortic | **2nd ICS, right sternal border** | **AS** is loudest here |
| `PULM` | Pulmonic | 2nd ICS, left sternal border | completeness |
| `ERB` | Erb's point | 3rd ICS, left sternal border | murmurs radiate here |
| `TRIC` | Tricuspid | 4th–5th ICS, left lower sternal border | completeness |
| `MITR` | **Mitral / apex** | **5th ICS, midclavicular line** | **MS, MR and MVP are all loudest here** |

*ICS = intercostal space. Find the 2nd ICS from the sternal angle (the ridge below the notch).*

**If time only allows two sites, take `MITR` and `AORT`** — between them they cover four of our
five classes.

**Positioning:** sitting or supine. For suspected mitral pathology (MS/MR/MVP), rolling the
patient into the **left lateral decubitus** position brings the apex against the chest wall and
markedly increases murmur audibility — worth doing for `MITR`.

**Breathing:** normal quiet breathing. If lung noise is swamping the heart, ask for a brief
**end-expiratory breath hold** — but never longer than comfortable, and never with a breathless
patient.

### Pulmonary — 6 sites minimum

Patient **sitting upright, arms crossed in front** — this rotates the scapulae out of the way.
Breathing **deeper than normal, in and out through an open mouth**. Stop if they get dizzy.

| Code | Site |
|---|---|
| `PRU` / `PLU` | Posterior right/left **upper** — above the scapular spine |
| `PRM` / `PLM` | Posterior right/left **mid** — interscapular, ~T4–T6 |
| `PRL` / `PLL` | Posterior right/left **lower** — below the scapula, ~T8–T10 |
| `LAR` / `LAL` | Lateral right/left — mid-axillary, ~5th–6th ICS *(optional)* |
| `ARU` / `ALU` | Anterior right/left upper — 2nd ICS midclavicular *(optional)* |

Always compare **left against right at the same level** — asymmetry is the clinically interesting
signal.

### Duration

| Type | Per site | Why |
|---|---|---|
| Heart | **20–30 s** | ~20–30 cardiac cycles; our model uses 3 s windows, so this yields 6+ independent samples |
| Lung | **20–30 s** | must contain **at least 3 full breath cycles** |

Budget roughly **8–10 minutes of contact time per patient** including positioning and consent.

---

## 7. Naming and metadata

### Filename

```
P{NNN}_{SITE}_{SEQ}_{YYYYMMDD}.wav
```

Example: `P007_MITR_01_20260822.wav`

`SEQ` increments if a site is retaken the same site. **Never overwrite a take** — keep the retake and
flag the bad one in the log; "why did that fail" is itself useful data.

### `recordings.csv` — one row per file

| Column | Example | Note |
|---|---|---|
| `pseudo_id` | `P007` | **never a name or MRN** |
| `file` | `P007_MITR_01_20260822.wav` | |
| `date` | `2026-08-22` | |
| `age_band` | `41-65` | **band, never DOB** |
| `sex` | `M` / `F` / `other` | |
| `site_code` | `MITR` | from §6 |
| `position` | `left-lateral` | |
| `duration_s` | `25` | |
| `device` | `INMP441+steth / Pi4` | |
| **`doctor_impression`** | `systolic murmur, ?MR` | **see below** |
| `referral_diagnosis` | `—` | if already known |
| `xray_available` | `yes` / `no` | links to the X-ray set |
| `quality_flag` | `good` / `noisy` / `discard` | operator judgement at the time |
| `notes` | `patient coughing` | |
| `consent` | `yes` | must be `yes` for every row |

### The governing column is `doctor_impression`

**Unlabelled audio is nearly worthless to us.** We cannot measure where a model fails without
knowing the truth. So for every recording, ask the doctor for their impression *at the time* —
even "sounds normal to me" is a label, and even an uncertain one is far better than none.

Ask explicitly whether they'd be willing to **listen to the recordings afterwards and label
them**, and whether a cardiologist contact could confirm the interesting cases (§10).

---

## 8. Rehearsal — prior to the visit

Do not debug hardware in front of a patient.

- [ ] Coupling built and sealed (§4)
- [ ] I2S overlay confirmed; `arecord -l` lists the capture device
- [ ] Record **the operator's own heartbeat at the apex** — S1/S2 ("lub-dub") must be clearly audible
- [ ] Record **the operator's own breathing** — breath sounds clearly audible
- [ ] Inspect a spectrogram: energy where expected, no mains hum dominating
- [ ] Confirm the 20 s button-triggered script produces a correctly named file
- [ ] Pi boots from the power bank with **no monitor and no keyboard**
- [ ] Backup handset recorder installed and tested
- [ ] Consent forms printed in the local language
- [ ] `recordings.csv` template + a **paper** logbook and pen (batteries fail)
- [ ] Clinical data path added to `.gitignore`

---

## 9. Timing and risk

**Development closes 24 August 2026; the semi-final is 25 August 2026.**

Data collected at this stage **cannot retrain the models within the remaining schedule**, and
is not intended to. Its value is the failure analysis it supports and the evidential basis it
provides under technical questioning.

**The demonstration gateway and its SD card are not taken into a clinical setting before
25 August 2026.** Where a visit precedes that date, a **second SD card** is used for the
recorder so that the demonstration image remains unmodified.

**A visit with no recording at all is still worth making** — the doctor conversation (§10) and
the X-ray question are the high-value, low-risk parts.

---

## 10. Questions for the doctor

He is a general physician and a CMO, not a cardiopulmonary specialist — so **do not ask him to
be one.** His real expertise is *how medicine actually works in a resource-limited setting*, which
is exactly what our project claims to address. He has served in the forces and in remote regions,
which makes him unusually well qualified on the deployment question.

### A. Clinical reality — what actually walks in

1. What cardiac and respiratory presentations do you see most often here?
2. At what stage do patients typically present — early, or already advanced?
3. How much of your cardiac/respiratory assessment rests on the stethoscope alone?
4. Which conditions do you feel confident detecting by auscultation, and which do you not?
5. How often are you genuinely uncertain and refer mainly to be safe?

### B. Deployment — would this help, honestly

6. If a device flagged "possible murmur, please refer", would that change what you do?
7. **Which error hurts more here — a missed case, or a false alarm?** *(This directly sets our
   confidence threshold, so listen carefully.)*
8. How many false alarms per week before you'd stop trusting it?
9. Where would a screening device be most useful — here, at a sub-centre, or with field workers?
10. From your time in remote postings: **what makes a device survive in the field, and what makes
    people abandon it?**
11. Who would realistically operate it — doctor, nurse, ASHA worker? How much training is realistic?

### C. Equipment — established by telephone, before construction begins (§3b)

12a. **Do you have a telemedicine kit — eSanjeevani, ReMeDi or similar? Does it include a digital
     stethoscope?** *(If yes, it likely replaces our entire recording rig.)*
12b. Do you have an **electronic/digital stethoscope** of any kind? Can it save or export files?
12c. May we borrow a stethoscope and a **pulse oximeter** during recording? *(Non-destructive
     coupling in §3b; the oximeter gives free heart-rate ground truth.)*
12d. Do you have an **ECG machine**? *(Not used by our model — a strong future-work angle.)*

### D. Data — the concrete asks

12. May we obtain **anonymised chest X-rays**? Are they **digital (DICOM/JPEG) or film**?
    *(If film, can they be photographed? Quality will suffer — worth measuring.)*
13. **Do you see paediatric chest cases?** *(Our model is trained on ages 1–5 — this is the
    like-for-like test.)*
14. Are the X-rays accompanied by a **radiologist report** we could use as ground truth?
15. What permission or ethics process do we need for both audio and images?
16. Is there a quiet room we could use for recording?

### E. Ground truth — without this the data is unusable

17. Would you be willing to give a **one-line impression at the time of each recording**?
18. Would you later **listen to the recordings and label them**?
19. Could you connect us to a **cardiologist or pulmonologist** to confirm the interesting cases?
    *(This is the highest-value ask in the entire visit — say it plainly.)*

### F. Their perspective

20. What would make you personally trust a machine's opinion on a heart sound?
21. What would you want it to do that we haven't thought of?

**Bring the working demo.** Let him press Self-Test and watch it classify. A specific question
after seeing it run is worth more than any question on this list.

---

## 11. After the visit

1. **Listen to every file end to end.** Delete anything with intelligible speech (§2).
2. Complete `recordings.csv` the **same day**, while the details are fresh.
3. Store clinical data **outside the git repo**; confirm `.gitignore` covers it.
4. Run the recordings through `core/features.py` and compare the spectrograms against Yaseen —
   **expect them to look different**, and quantify how.
5. Score the cardio model against the doctor's labels. Report **per-class recall**, not just
   accuracy — with a handful of samples, accuracy alone is meaningless.
6. Write up the failure analysis: *what broke, and why*. Noise? Coupling? The 60 Hz roll-off?
   Population mismatch?

> **This last document is the deliverable.** "We tested our model on real patients and it achieved
> X" is a weaker claim than "we tested it on real patients, it failed in these specific ways, and
> here is what each failure tells us." The second is what technical review actually requires —
> and it's true.
