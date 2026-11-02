# Test-time adaptation on the clinic recordings

Date: 1 Nov 2026 · Roadmap stage S3 for the heart branch (Aviral's part) · Proposal 4.3

## What was run

```bash
python -m cardio.eval.real_world --data-dir <Real_World_Data> --adapt
```

**Method.** The batch-normalisation statistics of the final heart model are re-estimated from clinic recordings:

- one forward pass with momentum 0 over the four batch-normalisation layers;
- no labels, no gradients, and the weights are unchanged;
- a fresh copy of the model is adapted each time, so the original model never changes.

**Split.** The 10 patients are split into two halves of 5 by patient number.

- Adapt on one half (25 recordings, 75 windows) and measure the other half.
- Then swap, so each recording is measured once by a model that never saw it.
- The collapse guard from proposal 4.3 keeps the original statistics if the adapted model puts more than 95% of the adaptation recordings in one class while the original did not.

**Mechanism check on public data first.** On Yaseen fold 0, the adapted copy's batch-normalisation statistics change, the original model's do not, and clean accuracy is unharmed: 99.0% before and 100% after on the held-out half.

The adaptation code is a small helper in `cardio/eval/real_world.py`. The team's shared adaptation module, `cardio/tta/bn_adapt.py`, did not exist yet; this analysis should switch to it once it does.

## Results (aggregates only; full numbers in `reports/heart_real_world.json`)

| Measure | Before | After |
|---|---|---|
| Recordings called abnormal | 50 of 50 | 50 of 50 |
| Predicted classes (Normal, AS, MS, MR, MVP) | 0, 0, 16, 3, 31 | 0, 20, 13, 14, 3 |
| Mean p(abnormal) | 0.927 | 0.863 |
| Mean top-class probability | 0.535 | 0.425 |
| Patients whose 5 positions agree on refer-or-not | 10 of 10 | 10 of 10 |
| Recordings passing the quality check called abnormal | 9 of 9 | 9 of 9 |
| Time to adapt, including loading the model | 0.20 to 0.28 s | |
| Collapse guard triggered | no, in either half | |

The quality check, segmentation and heart rate do not use the model, so they are unchanged: 9 of 50 pass, and 1 of 50 segments. 8-bit agreement for the adapted model needs re-quantisation, which is S4.

## What it means

Adaptation is fast (well under the proposal's 5 s), and it does change the model's behaviour. The pile-up on MVP disappears, and confidence falls. It does not move a single recording to Normal, so the refer-or-not answer, which is the one that matters, is unchanged.

Without diagnoses we cannot say whether the new class mix is closer to the truth. Every patient being abnormal at every position is still implausible, so batch-normalisation re-estimation alone does not undo this shift.

That fits the S2 picture. Most of these recordings fail the quality check and defeat heart-cycle segmentation. Handling noise and low-frequency energy are not just a change in the average level and spread of the model's internal features, which is all this kind of adaptation corrects.

Two cheap things to try next, outside this stage:

1. **Input-side fixes.** A stronger band-pass or noise gate for this microphone. The input normalisation also still uses Yaseen's mean and spread, while the clinic input sits 0.6 standard deviations away.
2. **The optional entropy-minimisation step** (proposal 4.3, "if time allows").

Both belong with the team's adaptation experiments.
