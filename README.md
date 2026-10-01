# CardioNary

**Ultra-compact dual-CNN late fusion for edge AI cardiopulmonary screening — heart
sounds and chest X-rays, each read in the context of the patient, with one
refer-or-not decision per visit.**

CS F407 Artificial Intelligence · BITS Pilani, Hyderabad Campus · First Semester 2026–27
· Track: Application

> CardioNary is a screening aid for academic and educational use. It flags patients
> for referral; it is not a diagnostic device, and every output is meant for review
> by a clinician.

---

## The idea

A doctor does not decide from one signal. They listen to the chest, look at the
film, and then weigh both against the patient's age, history and symptoms.
CardioNary follows the same shape, small enough to run offline on an ordinary CPU:

- **A heart-sound model** — a compact CNN built from depthwise-separable
  convolutions — classifies 3 seconds of heart audio (a 40 × 192 spectrogram) into
  Normal, Aortic Stenosis, Mitral Stenosis, Mitral Regurgitation or Mitral Valve
  Prolapse.
- **A chest X-ray model** — MobileNetV2 fine-tuned by transfer learning — reads a
  96 × 96 greyscale X-ray as normal or pneumonia.
- **Secondary characteristics.** Each input is also measured with classical signal
  and image processing (heart-cycle and murmur timing; X-ray density, texture and
  the region the CNN relied on), and the measurements are shown next to the answer.
- **Late fusion of three votes.** In each branch the CNN, a classifier on the
  secondary characteristics and a classifier on **patient metadata and symptoms**
  vote:
  `p = w1·p_CNN + w2·p_features + w3·p_metadata`, with the weights chosen by
  cross-validation and the metadata weight capped at 0.3.
- **Adapting to a new clinic.** Test-time adaptation re-estimates the batch
  normalisation statistics from the clinic's first few unlabelled cases, in seconds
  and without retraining, so the models stay **dataset-agnostic**.
- **One decision per visit.** The two branches combine as
  `p = 1 − (1 − p_heart)(1 − p_xray)`, and the patient is referred when
  `p · C_miss > C_refer` — the maximum-expected-utility choice for the clinic's
  costs of a miss and of a referral.
- **Edge-first.** Both models are compressed to 8 bits by post-training
  quantisation, and everything runs locally: patient data never leave the clinic.

### Why late fusion

Feeding metadata straight into the network invites it to lean on whatever is
easiest — age or sex, say — and stop listening to the signal. Keeping the CNN, the
secondary characteristics and the patient metadata as separate votes keeps each one
measurable and interpretable, lets the CNN be trained, compressed and tested once,
and lets a missing input simply drop out while the remaining weights are
renormalised. The metadata vote can shift a borderline case but never outweigh the
recording.

---

## Starting point: prior work

The audio and image front-ends, leakage rules, training, quantisation and
secondary-characteristic code were developed by members of this team in an earlier
project, **Project Chiron**. They are imported into `core/` as prior work and are not
presented as new. What this project adds:

1. **Late fusion** of the CNN with secondary characteristics and patient metadata.
2. **Test-time adaptation** to a new clinic from unlabelled cases.
3. **One cost-aware decision per visit** across both tests.
4. **Dataset-agnostic evaluation** on datasets the models were never trained on,
   including about 100 heart-sound clips we record ourselves with a stethoscope and
   a MEMS microphone.
5. **An offline, multi-language screening tool** for health workers.

## Datasets

Datasets are downloaded by scripts and never committed.

| Use | Heart sounds | Chest X-rays |
|---|---|---|
| Training and 5-fold testing | Yaseen | Kermany (paediatric) |
| Unseen datasets | BUET heart-sound dataset; PhysioNet 2022 challenge data | RSNA Pneumonia; VinDr-CXR if time allows |
| Patient metadata | PhysioNet 2022: age group, sex, height, weight | RSNA: age, sex |
| New-device test | Our own ~100 recordings, with a symptom checklist | — |

Every split is patient-disjoint, and automated tests reject any split that places
one patient in both training and testing.

## Teams

| Primary focus | People | Owns | Team branch |
|---|---|---|---|
| **Heart-sound model** | 3 | Heart CNN · secondary characteristics · feature and metadata classifiers · test-time adaptation · our own recordings | `cardio` |
| **Chest X-ray model** | 4 | X-ray CNN · secondary characteristics · feature and metadata classifiers · test-time adaptation · calibration | `lung` |
| **Both** | 7 | Visit decision, shared evaluation code, the offline tool | `dashboard` (later) |

## Repository layout

```
core/       shared pipeline both teams import — front-ends, leakage rules, training,
            quantisation, secondary characteristics, and the shared contracts
cardio/     heart-sound team:  model/  fusion/  tta/  experiments/
lung/       chest X-ray team:  model/  fusion/  tta/  experiments/
app/        offline screening tool (later in the semester)
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

The full guide is [`docs/BRANCHING.md`](docs/BRANCHING.md). The short version:

- Make your own branch from your team's branch: `<team>-<your name>-<topic>`.
- Open a pull request back into your team's branch; one teammate approves.
- Merge with **Create a merge commit**, never squash, so everyone's history is kept.
- Nobody pushes straight to `main` or `develop`.

## Use of AI tools

AI assistants are used in this project, as the course permits. Commits with AI
contribution carry a `Co-Authored-By` trailer. The final report lists every tool
used, what it was used for, and how its output was verified.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
