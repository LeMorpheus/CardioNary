# S3 adaptation with the reference model

The public-data run used the same float reference weights and training
normalisation as the provisional S2 run. It completed 140 draws across
BMD-HS, CirCor, 5 dB white noise, and 300 Hz low-pass Yaseen. Each site had
five draws at each of 8, 16, 32, and 64 recordings, plus five draws at each of
10%, 50%, and 90% abnormal at size 32. The fixed scoring panels contained
27 BMD-HS patients, 203 CirCor patients, and 50 Yaseen fold 0 clips. Patient
groups used for scoring stayed outside the adaptation pool.

The figures below are means of the five random draws at 32 recordings.
The aggregate report contains the other conditions, model hashes, per-draw
timings, and patient bootstrap intervals.
[Reference run](../../reports/heart_tta_reference.json)

- **BMD-HS** referral macro-F1 stayed at 0.449. AUROC fell from 0.773 to 0.571.
- **CirCor** referral macro-F1 stayed at 0.161. AUROC rose from 0.523 to 0.698,
  while clean Yaseen macro-F1 dropped from 1.000 to 0.444.
- **White noise** referral macro-F1 fell from 0.729 to 0.443. AUROC rose
  from 0.815 to 0.847, while clean Yaseen macro-F1 dropped to 0.444.
- **Low-pass Yaseen** had macro-F1 and AUROC of 1.000 before and after
  adaptation. The clean-to-shift gap was zero, so recovery was undefined.
  The 10% abnormal stress batches reduced shifted macro-F1 to 0.851.

No condition recovered 50% of a positive macro-F1 gap. The longest CPU
adaptation with the collapse check was 1.21 seconds. The five-class guard
never triggered on these adaptation batches. The decline on clean Yaseen
shows why the guard and the clean-score check are both needed.

These are provisional reference-model results. The official team-model run
uses the same runner with the confirmed checkpoint and paired report.
