# Dataset Research — Cardiac and Respiratory Audio

Survey of every serious candidate, with the evidence needed to justify the choices made in
[05_DATASET_SELECTION.md](05_DATASET_SELECTION.md). Research date: **2026-08-12**.

---

## 0. Principal finding

> **Respiratory audio tasks split into two families with completely different accuracy ceilings.**
>
> | Family | Example labels | Realistic ceiling |
> |---|---|---|
> | **Event / adventitious-sound level** | normal, crackle, wheeze, both, rhonchi, stridor | **~65–80 %** |
> | **Disease / diagnosis level** | healthy, asthma, COPD, pneumonia, heart failure | **95–99 %** |
>
> The ≥95 % requirement is achievable **only** in the disease-level family. Any plan that
> targets ICBHI's official 4-class event task and promises 95 % is promising something that
> no published method has delivered.

The reason for the gap is discussed honestly in §6 — it is not purely that disease
classification is "easier".

---

## 1. Cardiac datasets

### 1.1 Yaseen 2018 — *Classification of Heart Sound Signal Using Multiple Features* 

| Property | Value |
|---|---|
| Size | **1000 recordings** |
| Classes | **5**, perfectly balanced at **200 each** |
| Class list | Normal (N), Aortic Stenosis (AS), Mitral Stenosis (MS), Mitral Regurgitation (MR), Mitral Valve Prolapse (MVP) |
| Sample rate | **8 kHz**, mono |
| Duration | 1–4 s per clip |
| Format | `.wav` |
| Source | <https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-> |
| Paper | *Applied Sciences* 8(12):2344, 2018 — <https://dx.doi.org/10.3390/app8122344> |
| Licence | Paper is **CC BY** (open access). Dataset distributed with the paper's repository. |

**Published accuracy on this dataset:**

| Method | Accuracy |
|---|---|
| Lightweight CNN | **98.6 %** |
| Hybrid CNN-LSTM-GRU | **98.0 %** (test) |
| Hybrid CNN + GRU | **99.3 %** |

**Why it fits us almost perfectly:**
- 5 genuine pathology classes — comfortably clears "not normal-vs-abnormal".
- **Already 8 kHz mono** — our exact target rate. No resampling artefacts, no decisions to make.
- 1–4 s clips fit a 3 s inference window with trivial pad/crop.
- Perfectly balanced → accuracy is a meaningful metric, no class-weighting games.
- 98–99 % is reported by *many independent groups with small CNNs*, which is the signal that
  matters: it is not one lucky paper.

**Honest limitation (must be stated in the report):** the authors assembled the clips
"from myriad online sources" — largely curated teaching/reference recordings, downsampled and
mono-mixed. The model therefore learns **textbook murmur morphology**, not the messy
variability of real bedside auscultation. It is an excellent, reproducible, high-accuracy
*classifier demonstration*; it is not evidence of clinical readiness. Say this out loud in the
report and it becomes a strength (scientific honesty) rather than a gap a reviewer can identify.

---

### 1.2 PhysioNet/CinC 2016 Challenge

| Property | Value |
|---|---|
| Size | ~3,126 phonocardiograms, 5 source databases |
| Classes | **2** — normal vs abnormal (+ signal-quality annotation) |
| Sample rate | 2 kHz |

**Rejected.** Binary only, which requirement R2 explicitly excludes. Its value is as an
optional *robustness* corpus: after the 5-class model works, testing it on CinC-2016 normals
shows how it behaves on heterogeneous real-world recordings. Nice-to-have, not the base.

---

### 1.3 CirCor DigiScope

| Property | Value |
|---|---|
| Size | 5,272 recordings, 1,568 paediatric subjects (0–21 y) |
| Labels | Murmur present/absent/unknown; murmur timing, shape, pitch, grading, quality, location; S1/S2 segmentation |
| Sample rate | 4 kHz |
| Auscultation sites | 4 (AV, PV, TV, MV) |

**Not chosen as primary.** Richest annotation of any public PCG corpus, but:
- The headline task is murmur **present / absent / unknown** — 3 classes, one of which is
  "unknown". State-of-the-art weighted accuracy on the official split is well below 95 %.
- Multi-site recording introduces position as a confound needing explicit handling.
- Paediatric-only.

Excellent for a future *"murmur grading and localisation"* extension. Wrong tool for a
95 %-accuracy hardware demo.

---

## 2. Respiratory datasets

### 2.1 KAUH / Fraiwan — *Lung sounds recorded from the chest wall using an electronic stethoscope* 

| Property | Value |
|---|---|
| Subjects | **112** (35 healthy, 77 unhealthy), age 21–90 (50.5 ± 19.4), 69 M / 43 F |
| Files | **336** `.wav` = 112 subjects × **3 filter renderings each** |
| Device | 3M Littmann Electronic Stethoscope **model 3200** |
| Duration | 5–30 s per recording |
| Licence | **CC BY 4.0** |
| Data | <https://data.mendeley.com/datasets/jwyy9np4gv/3> (DOI `10.17632/jwyy9np4gv.3`) |
| Paper | *Data in Brief* — <https://pmc.ncbi.nlm.nih.gov/articles/PMC7937981/> |

**Subject counts per diagnosis:**

| Diagnosis | Subjects |
|---|---|
| Normal | 35 |
| Asthma | 32 |
| Heart failure | 21 |
| COPD | 9 |
| Lung fibrosis | 5 |
| Pneumonia | 5 |
| Bronchitis (BRON) | 3 |
| Pleural effusion | 2 |

**Filter renderings** (this is the trap — see §6.2):

| Code | Filter | Band | Emphasis |
|---|---|---|---|
| `B` | Bell | 20–1000 Hz | 20–200 Hz |
| `D` | Diaphragm | 20–2000 Hz | **100–500 Hz** |
| `E` | Extended | 20–1000 Hz | 50–500 Hz |

**Filename convention:**
`[Filter][P][PatientNo]_[Diagnosis],[SoundType],[ChestZone],[Age],[Gender]`
e.g. `BP60_heart failure,Crep,P L L,83,F`
Sound types: `I`nspiratory, `E`xpiratory, `W`heezes, `C`rackles, `N`ormal, `Crep`itations.

---

### 2.2 ICBHI 2017 Respiratory Sound Database

| Property | Value |
|---|---|
| Size | **920 recordings**, **126 subjects**, **6,898** annotated respiratory cycles, ~5.5 h |
| Sample rates | 4 kHz – 44.1 kHz (heterogeneous) |
| Duration | 10–90 s |
| Event labels | crackle / wheeze / both / normal (per cycle) |
| Diagnosis labels | Healthy, COPD, Pneumonia, Bronchiectasis, Bronchiolitis, URTI, LRTI, Asthma |
| Collection sites | Portugal and Greece, multiple devices |
| Access | <https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge> — free for research |

**Severe class imbalance at diagnosis level:** COPD dominates (64 of 126 subjects,
~793 of 920 recordings). A classifier that predicts "COPD" for everything already scores
~86 % recording-level accuracy. Any COPD-inclusive accuracy figure must be reported alongside
a **confusion matrix and macro-F1**, or it is meaningless.

**Event-level (the official challenge task):** the 4-class crackle/wheeze/both/normal task
with the official 60/40 patient-independent split is *hard*. Published ICBHI Scores cluster
around 60–65 %. **This task cannot reach 95 %.**

---

### 2.3 SPRSound (SJTU paediatric)

| Property | Value |
|---|---|
| Size | 2,683 recordings, 292 children (1 month – 18 y) |
| Events | 9,089 annotated |
| Classes | normal, rhonchi, wheeze, stridor, fine crackle, coarse crackle, wheeze+crackle |
| Sample rate | 8 kHz |

Finest event granularity available and natively 8 kHz. But it is an **event-level** task —
same ceiling problem as ICBHI-4. Also heavily imbalanced (stridor is rare). Keep as a possible
Phase-3 extension for paediatric event detection; not a route to 95 %.

---

## 3. Reported performance summary (respiratory, disease-level)

| Study | Data | Classes | Split | Accuracy |
|---|---|---|---|---|
| Fraiwan et al., *Recognition of pulmonary diseases from lung sounds using CNN + LSTM* ([PMC8019351](https://pmc.ncbi.nlm.nih.gov/articles/PMC8019351/)) | **KAUH (103 pt) + ICBHI (110 pt) = 213 pt, 1,483 recordings**, all resampled to **4 kHz**, 5 s segments | **6** (Normal, Asthma, Pneumonia, BRON, COPD, Heart Failure) | **subject-level 10-fold CV** | **CNN+BDLSTM 99.62 %**, CNN alone **99.04 %**, BDLSTM alone **98.16 %** |
| TinyML lung disease classification, *Future Internet* 16(11):391 | public lung sound dataset | multi-class | not subject-disjoint (as reported) | custom CNN **96 %**, quantised for MCU deployment |
| Multi-feature lightweight CNN (IEEE Access) | KAUH-family | multi-class | — | ~91.7 % overall |
| CNN-RNN fusion, *Sci Rep* 2025 | ICBHI | — | — | 94.0 % acc, F1 0.94 |
| Transformer fusion (MFITN-E2NetGA) | ICBHI 2017 | multi-class | — | 98.75 % |

**Class distribution of the Fraiwan 6-class merged corpus** (this is the **Track B / B6 upgrade**
task; Track A uses the KAUH-only 4-class subset — see [05 §1](05_DATASET_SELECTION.md)):

| Class | KAUH | ICBHI | Total |
|---|---|---|---|
| Normal | 35 | 26 | **61** |
| Asthma | 32 | 1 | **33** |
| Pneumonia | 5 | 6 | **11** |
| BRON | 3 | 13 | **16** |
| COPD | 9 | 64 | **73** |
| Heart Failure | 19 | — | **19** |
| **Total** | 103 | 110 | **213** |

---

## 4. Datasets considered and set aside

| Dataset | Why not |
|---|---|
| PhysioNet CinC 2016 | binary only (R2) |
| CirCor DigiScope | 3-class with an "unknown" class; SOTA < 95 % |
| SPRSound | event-level ceiling |
| ICBHI event task (4-class) | event-level ceiling (~65 %) |
| Chest X-ray "lung disease" Kaggle sets | wrong modality — this is an **audio** project; also several are silently merged/duplicated collections |
| R.A.L.E. | licensing/access friction, small |

> Note: Several highly-cited "lung disease classification, 98 % accuracy" papers turn out to use
> **chest X-ray images**, not audio. When collecting citations for the report, confirm the
> modality before citing.

---

## 5. Availability, licence and citation obligations

| Dataset | Access | Licence | Must cite |
|---|---|---|---|
| Yaseen 2018 | GitHub, direct download | CC BY (paper) | Yaseen, Son & Kwon, *Appl. Sci.* 8(12):2344, 2018 |
| KAUH / Fraiwan | Mendeley Data, direct download | **CC BY 4.0** | Fraiwan et al., *Data in Brief*, DOI `10.17632/jwyy9np4gv.3` |
| ICBHI 2017 | bhichallenge.med.auth.gr, free for research | research use, attribution | Rocha et al., ICBHI 2017 |

All three are compatible with an **Apache 2.0** source repository as long as the *data* is not
re-hosted under Apache 2.0. **Do not commit dataset audio into the Git repository.** Ship a
`scripts/download_datasets.py` that fetches from the official sources and verifies checksums,
plus a `DATA_LICENSES.md` recording each licence and citation. This also satisfies the
competition's repository-hygiene rule about not committing large generated/binary assets.

---

## 6. Why the disease-level numbers are so high — and how we avoid fooling ourselves

This section exists because a technical reviewer asking *"why is the reported lung accuracy 99 % when
the ICBHI challenge SOTA is 65 %?"* is a question we must be able to answer in one sentence.

### 6.1 Leakage vector 1 — the label is a per-patient constant
Diagnosis is a property of the **patient**, and every recording from that patient shares the
same room, stethoscope, operator, background noise and skin coupling. Split at the
*segment* or *recording* level and the network can score near-perfectly by recognising the
**recording session**, never learning pathology at all.

**Mitigation (mandatory):** every split — train/val/test and every CV fold — is
**patient-disjoint**. Patient ID is derived from the filename before any windowing.

### 6.2 Leakage vector 2 — KAUH's three filter renderings
The 336 KAUH files are **112 recordings rendered through 3 stethoscope filters**. `BP60`,
`DP60` and `EP60` are the *same acoustic event*. Putting `BP60` in train and `DP60` in test is
straightforward leakage that inflates accuracy by a large margin.

**Mitigation (mandatory):** pick **one** filter (we use **Diaphragm `D`**, 20–2000 Hz, which
emphasises 100–500 Hz — the band where crackles and wheezes live), or keep all three but bind
them to the same fold as one unit. Never split across filter renderings.

### 6.3 Leakage vector 3 — device/site confound in the merged corpus
In the 6-class merged task, **Heart Failure comes only from KAUH** and **most COPD comes from
ICBHI**. A model can partly solve the task by identifying the recording device.

**Mitigations:**
- Resample everything to a single rate (8 kHz), band-limit identically, and
  amplitude-normalise **per recording** so absolute gain carries no information.
- Report a **source-stratified confusion matrix** (KAUH rows vs ICBHI rows).
- Run a **KAUH-only 4-class ablation** (Normal / Asthma / Heart Failure / COPD). If that also
  clears ~95 % patient-disjoint, the device confound is not doing the heavy lifting.

### 6.4 Small-N reality
213 patients is small. Some classes have 11–19 patients. A single test patient can move
accuracy by several points.

**Mitigation:** report **patient-disjoint 10-fold cross-validation mean ± std**, not a single
lucky holdout. Report **macro-F1 and per-class recall** alongside accuracy. A single number is
not a result.

Doing all four of these is what turns "we got 99 %" from a red flag into a defensible claim —
and it maps directly onto the *Technical challenge* (10 %) and *Accuracy* (15–20 %) scoring
criteria.

---

## Sources

- [Yaseen 2018 dataset (GitHub)](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-)
- [Yaseen et al., Applied Sciences 8(12):2344](https://dx.doi.org/10.3390/app8122344)
- [KAUH lung sound dataset (Mendeley Data)](https://data.mendeley.com/datasets/jwyy9np4gv/3)
- [Fraiwan et al., Data in Brief (dataset description)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7937981/)
- [Fraiwan et al., CNN+LSTM pulmonary disease recognition (99.62 %)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8019351/)
- [ICBHI 2017 Challenge / Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
- [Empowering Healthcare: TinyML for Precise Lung Disease Classification, Future Internet 16(11):391](https://doi.org/10.3390/fi16110391)
- [Lung Sound Classification with Multi-Feature Integration Utilizing Lightweight CNN Model (IEEE)](https://ieeexplore.ieee.org/document/10419195/)
- [Deep Learning-Driven Early Diagnosis of Respiratory Diseases using CNN-RNN Fusion, Sci Rep](https://www.nature.com/articles/s41598-025-28832-7)
- [Heart Sound Classification using a Hybrid of CNN and GRU (99.3 %)](https://www.sciencedirect.com/science/article/pii/S1877050924009724)
- [A review on deep learning methods for heart sound signal analysis, Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1434022/full)
