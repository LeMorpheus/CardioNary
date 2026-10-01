# core — the shared pipeline

Everything both teams use, defined once. Import from `core`; never copy it into your
team's folder, or the two copies will drift apart.

| Module | What it does |
|---|---|
| manifests + `preprocess` | dataset lists, patient-disjoint folds, leakage checks |
| `features`, `images` | audio processing (band-pass filter, log-mel) and image processing |
| `spectrogram`, `ood` | spectrogram picture; check for recordings unlike the training data |
| `train`, `baselines`, `quantize` | training, simple baselines to beat, 8-bit compression |
| `descriptors` | secondary characteristics: systole and diastole timing, murmur timing and loudness, band powers; X-ray density by zone, texture and occlusion maps |
| `contracts` | what each model, late fusion and the visit decision take in and give back, and patient metadata |

Both teams use this code, so changes go on a `shared-<topic>` branch and are reviewed
by one person from each team. The modules arrive in the order shown at the end of
`docs/BRANCHING.md`.
