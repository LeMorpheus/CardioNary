# CardioNary v1: what the imported baseline contains

This is the README of the snapshot imported from our earlier project (Project Chiron /
CardioNary v1): the two trained models, the full ML pipeline, the backend and the
dashboard. Paths are updated to this repository's layout (`core/` instead of `ml/`).

> CardioNary is a screening aid for academic and educational use. It is not a
> diagnostic device.

---

## Where the project stands

### Heart-sound model — 5 classes

Normal · Aortic Stenosis · Mitral Stenosis · Mitral Regurgitation · Mitral Valve Prolapse

| | Result | Source file |
|---|---|---|
| 5-fold cross-validated accuracy | **98.89% ± 1.14%** | `reports/cv_yaseen_logmel_w16_b4_noaug.json` |
| Held-out fold, full precision → 8-bit | 98.06% → **98.06%** (no loss) | `reports/quantization_chiron_cardio_v1.json` |
| Full-precision vs 8-bit agreement | **100%** | same |
| Size | 13,509 parameters · 27.7 kB (8-bit) | `models/chiron_cardio_v1_int8.tflite` |
| Input | 3 s of audio at 8 kHz → log-mel spectrogram, 40 bands × 192 frames | `core/features.py` |

### Chest X-ray model — 2 classes

No pneumonia pattern · Pattern consistent with pneumonia

| | Result | Source file |
|---|---|---|
| Held-out fold, 8-bit, per image | **96.37%** (full precision 96.61%) | `reports/quantization_chiron_pulmo_v1.json` |
| Held-out fold, 8-bit, per patient | **97.85%** (full precision 98.03%) | same |
| Full-precision vs 8-bit agreement | **99.27%** | same |
| 5-fold cross-validation log | | `reports/kermany_mnet_cv.log` |
| Size | 412,773 parameters · 610 kB (8-bit) | `models/chiron_pulmo_v1_int8.tflite` |
| Model | MobileNetV2 (width 0.35), ImageNet-pretrained, fine-tuned | `core/train.py` |
| Input | greyscale, centre-cropped, 96 × 96 | `core/images.py` |

### Measurements beyond the class label

| What | Result | Code |
|---|---|---|
| Murmur timing (systolic vs diastolic) matches physiology | **49 of 52** held-out cases | `core/descriptors.py` |
| Where the X-ray model looks (3×3 occlusion) | Normal → upper field 36/60 · Pneumonia → middle field 43/60 | `core/descriptors.py` |
| Out-of-distribution check (energy above 1 kHz) | flags **0 of 53** reference recordings and **44 of 50 (88%)** field recordings | `gateway/chiron_live.py` |

### What we learned the hard way

- **We dropped a whole task on evidence.** The lung model began as lung-*sound*
  classification. A classifier that only reads which stethoscope made each recording
  scored 0.675; our network scored 0.643. The public lung-sound data was teaching the
  model the stethoscope, not the disease — so we moved to chest X-rays.
  See `reports/A1_pipeline_findings.md`.
- **The chest X-ray dataset's official split leaks:** 264 children (967 images) appear
  in both its train and test sets. We rebuilt the folds so no child is in both.
- **Every split is checked by tests**, not by trust. See `tests/test_no_leakage.py`.

---

## What is in this repository

```
core/            the ML pipeline: dataset download and manifests, audio and image
               front-ends, training with cross-validation, baselines, 8-bit
               quantisation, derived measurements, dashboard case data
models/        the four trained models (full-precision .keras and 8-bit .tflite)
gateway/       the backend (FastAPI + SQLite + WebSocket) and, in static/, the
               dashboard (plain HTML/CSS/JS, six languages)
tests/         leakage tests and model edge-case tests
data/manifests/ every dataset split, one row per recording or image, with its fold
reports/       every result file the numbers above come from, plus training logs
Test_Data/     53 heart sounds and 120 X-rays from the public held-out sets —
               no patient data (see Test_Data/README.md)
vectors/       the held-out predictions the dashboard's simulate mode replays
docs/          design and results documents
```

**Start reading with `docs/25_MACHINE_LEARNING.md`** — the whole ML side in plain
language. Then `docs/18_DERIVED_INDICATORS.md` for the secondary measurements, and
`docs/23_QNA_PREPARATION.md` for every figure with its reasoning.

---

## Run it

You need **Python 3.10**. Make a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 1. Check the ML pipeline

```bash
pip install -r requirements.txt
python -m pytest tests/
```

Expected: **23 passed, 5 skipped**. Four of the skips check feature caches, which only
exist once you build them (step 3). The fifth does not apply to the X-ray model.

### 2. Run the dashboard — no hardware needed

Simulate mode replays the real held-out predictions, so the dashboard behaves exactly
as it did with the original hardware.

```bash
pip install -r gateway/requirements.txt
```

macOS / Linux:

```bash
CHIRON_MEDIA_DIR=Test_Data python gateway/chiron_gateway.py --simulate
```

Windows PowerShell:

```powershell
$env:CHIRON_MEDIA_DIR = "Test_Data"; python gateway/chiron_gateway.py --simulate
```

Open <http://localhost:8000> and press **Run Screening Batch**.
`CHIRON_MEDIA_DIR` lets each case show its X-ray or play its heart sound.

### 3. Retrain from scratch (optional)

Download the data. The heart-sound set is fetched by script, which needs `unrar`
installed. Download the chest X-rays yourself from Mendeley Data
(doi `10.17632/rscbjbr9sj`, `ChestXRay2017.zip`), then unzip it into
`data/raw/kermany/img/`.

```bash
python -m core.download_datasets --dataset yaseen
python -m core.build_cardio_manifest
python -m core.build_kermany_manifest
```

Build the feature caches:

```bash
python -m core.preprocess --dataset yaseen
python -m core.preprocess --dataset kermany
```

Heart-sound model — cross-validation, then the final 8-bit model:

```bash
python -m core.train --dataset yaseen --no-augment --seeds 1
python -m core.quantize --dataset yaseen
```

Chest X-ray model — cross-validation, then the final 8-bit model:

```bash
python -m core.train --dataset kermany --arch mobilenet
python -m core.quantize --dataset kermany --arch mobilenet --augment
```

`train_pulmo.py` trains **both** models; the name is left over from the original
lung-sound task.

---

## What is deliberately not here

| Left out | Why |
|---|---|
| Firmware, hardware scripts and hardware test results | the course project is software-only |
| The real encryption key | `gateway/chiron_secret.py` is an all-zeros **placeholder**. The real key only protected results sent over Bluetooth from the old hardware |
| Field recordings from patients, and all photographs | collected under consent for the original project — never published |
| Raw datasets and feature caches | large, and downloadable (see above) |
| Documents about hardware, Bluetooth and the competition | not relevant to the software work; some included docs still link to them |

## Known corrections

- **The out-of-distribution figure is 44/50 (88%), not 82%.** Several documents and a
  code comment quote 82%. Re-measured on 28 Sep 2026 on the current field recordings,
  under every audio-preparation option, it is 88%. Both show the same thing: the check
  separates the two kinds of recording.
- **`gateway/requirements.txt` was incomplete.** It is missing `python-multipart`,
  without which the server does not start, and `numpy` and `Pillow`, which the upload
  and spectrogram code needs. All three are added here. Tested from a fresh virtual
  environment.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
