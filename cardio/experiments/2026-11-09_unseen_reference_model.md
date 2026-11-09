# S2 unseen-dataset check with the reference model

The S2 runner evaluated the float reference model from `main` at `751d275` on
BMD-HS and CirCor. This is a provisional baseline. The model and normalisation
report hashes, cohort counts, and 95% patient bootstrap intervals are in
[`reports/heart_unseen_reference.json`](../../reports/heart_unseen_reference.json).

- BMD-HS had 108 eligible patients. Referral AUROC was 0.679 (95% CI 0.542–0.798).
  The 0.5 cutoff flagged all 108 for referral. Four-class macro-F1 was 0.152
  across 51 patients with one shared diagnosis.
- CirCor had 812 eligible patients. Referral AUROC was 0.535 (95% CI 0.487–0.588).
  The 0.5 cutoff flagged 808 patients for referral. The run excluded 56 patients
  with only unknown labels and four patients with conflicting known labels.

The reference model flagged almost everyone at the fixed cutoff, and the
CirCor AUROC interval included 0.5. Feature extraction ran on all 872 BMD-HS
and 3,163 CirCor recordings, with each CSV row matching the path and patient
ID in its manifest.

To repeat the run, use the reference model and matching report from the stated
`main` revision, or run the script from a checkout containing those files.
The default output is `scratch/heart_unseen_reference.json`. For the team run,
provide the confirmed float model and its paired training normalisation report:

```powershell
python -m cardio.eval.run_unseen --model 'C:/path/to/final-model.keras' --normalization-report 'C:/path/to/quantization-report.json' --out reports/heart_unseen.json
python -m cardio.model.features --dataset bmdhs --out scratch/heart_features_bmdhs.csv
python -m cardio.model.features --dataset physionet --out scratch/heart_features_physionet.csv
```
