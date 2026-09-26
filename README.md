# CardioNary

**Dual-modality cardiopulmonary screening with edge-deployable AI — heart sounds and
chest X-rays, each read in the context of the patient.**

CS F407 Artificial Intelligence · BITS Pilani, Hyderabad Campus · First Semester 2026–27

> CardioNary is a screening aid for academic and educational use. It is not a
> diagnostic device, and every output is meant for review by a clinician.

---

## The idea

A doctor does not decide from one signal. They listen to the chest, look at the
film, and then talk to the patient — age, symptoms, history — before arriving at a
judgement. CardioNary is built to follow the same shape:

- **A heart-sound model** classifies a heart recording into five classes:
  Normal, Aortic Stenosis, Mitral Stenosis, Mitral Regurgitation, Mitral Valve Prolapse.
- **A chest X-ray model** separates radiographs with no pneumonia pattern from those
  with a pattern consistent with pneumonia.
- **Each model is then combined with patient metadata and symptoms**, so the final
  decision reflects the clinical signal *and* the patient in front of you.
- **Each model adapts at test time** to the instruments and recording conditions of the
  site it is deployed at, so a model trained on one set of stethoscopes or scanners
  keeps working on another.

Everything is designed to stay small enough to run at the edge — on modest hardware,
without a cloud service in the loop.

### Why fusion, and not one big model

Feeding metadata straight into a single network invites it to lean on whatever is
easiest — age or sex, say — and to stop listening to the signal. Keeping the
clinical model and the patient context as separate modalities, and combining them
deliberately, lets us measure what each contributes and stop either one from
drowning out the other. Comparing **late fusion** with **early fusion** is one of the
central experiments of this project.

---

## Starting point: prior work

The baseline pipeline was developed by members of this team in an earlier project,
**Project Chiron**. It is imported here as prior work and extended; it is not
presented as new. Retained unchanged:

| Retained | What it does |
|---|---|
| Audio front-end | 8 kHz, 3 s windows, causal 20–800 Hz Butterworth band-pass, log-mel spectrogram (40 bands × 192 frames) |
| Image front-end | Greyscale, centre-crop to square, 96 × 96 |
| Spectrogram rendering | Display spectrogram for every heart-sound input |
| Derived measurements | S1/S2 segmentation, murmur timing (systolic vs diastolic), occlusion maps on X-rays |
| Anti-leakage rules | Patient-disjoint folds and leakage checks enforced as assertions and tests |
| Out-of-distribution gate | Flags recordings that do not resemble the training data |
| Compression | 8-bit post-training quantisation, judged by prediction agreement |

The code for all of this arrives in `core/` over the first week of the repository.
[`docs/BRANCHING.md`](docs/BRANCHING.md) shows the order.

## What this project adds

1. **Fusion with patient metadata** — late fusion and early fusion, compared on the
   same evaluation protocol, with metadata kept from dominating the decision.
2. **Test-time adaptation** — the model adjusts to a new site's data and measuring
   conditions without new labels, so it becomes instrument-agnostic.
3. **A clinical dashboard** — multi-language, suited to use at the edge.

---

## Teams

| Team | People | Owns | Branch lane |
|---|---|---|---|
| **Cardio** | 3 | Heart-sound model · its metadata fusion · its test-time adaptation | `cardio/dev` |
| **Lung** | 4 | Chest X-ray model · interpretability · its metadata fusion · its test-time adaptation | `lung/dev` |
| **Everyone** | 7 | Dashboard, built once the models, fusion and adaptation are complete | `app/dev` |

## Repository layout

```
core/       shared pipeline both teams import — front-ends, leakage rules, training,
            evaluation, and the contracts both models implement
cardio/     heart-sound team:  model/  fusion/  tta/  experiments/
lung/       chest X-ray team:  model/  fusion/  tta/  experiments/
app/        dashboard (later in the semester)
tests/      evaluation gates, including the anti-leakage tests
docs/       how we work, and the report
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/
```

Python 3.10. Datasets and trained weights are never committed; see
[`docs/BRANCHING.md`](docs/BRANCHING.md) for what is and is not tracked.

## How we work

Branching, naming, merge policy and releases are in
[`docs/BRANCHING.md`](docs/BRANCHING.md). The short version:

- Nobody commits to `main` or `develop` directly.
- Work happens on `<lane>/<workstream>/<name>-<topic>` branches.
- Changes land through pull requests with merge commits, so every person's history
  is kept.

## Use of AI tools

AI assistants are used in this project, as the course permits. Commits with AI
contribution carry a `Co-Authored-By` trailer. The final report lists every tool
used, what it was used for, and how its output was verified.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
