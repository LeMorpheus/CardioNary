# core — the shared pipeline

Everything both teams use, defined once. Import from `core`; never copy it into your
team's folder, or the two copies will drift apart.

| Module | What it does |
|---|---|
| manifests + `preprocess` | dataset lists, patient-disjoint folds, leakage checks |
| `features`, `images` | audio processing (band-pass filter, log-mel) and image processing |
| `spectrogram`, `ood` | spectrogram picture; check for recordings unlike the training data |
| `train`, `baselines`, `quantize` | training, simple baselines to beat, 8-bit compression |
| `descriptors` | heart-cycle timing, murmur timing, X-ray occlusion maps |
| `contracts` | what each model takes in and gives back |

Both teams use this code, so changes go on a `shared-<topic>` branch and are reviewed
by one person from each team. The modules arrive in the order shown at the end of
`docs/BRANCHING.md`.
