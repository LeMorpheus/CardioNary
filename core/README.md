# core — the shared pipeline

Everything both teams depend on, defined once. Lanes import from `core`; they never
copy it. Every change here needs review from both team leads.

| Module | Responsibility |
|---|---|
| manifests + `preprocess` | dataset manifests, patient-disjoint folds, anti-leakage assertions |
| `features`, `images` | audio front-end (causal band-pass, log-mel) and image front-end |
| `spectrogram`, `ood` | display spectrogram; out-of-distribution gate |
| `train`, `baselines`, `quantize` | cross-validated training, hurdle baselines, 8-bit quantisation |
| `descriptors` | S1/S2 segmentation, murmur timing, occlusion maps |
| `contracts`, `eval` | the interface both models implement; shared metrics |

The modules arrive in this order during the first week — see `docs/BRANCHING.md` §6.
