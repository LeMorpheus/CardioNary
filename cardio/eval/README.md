# Metrics and single-clip corruption

```python
from cardio.eval.metrics import classification_metrics, patient_bootstrap

scores = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
interval = patient_bootstrap(
    [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], ['a', 'a', 'b', 'c'],
    metric=lambda y, p: classification_metrics(y, p)['accuracy'],
)
```

`classification_metrics` accepts integer class-column indices and binary
positive-class probabilities or an N-by-C probability matrix. For model outputs
from `core.contracts`, column order must match `CLASSES['heart']`. It returns
accuracy, macro-F1, per-class recall, AUROC, sensitivity at 90% specificity, and
15-bin top-label expected calibration error (ECE). Logits and unknown labels
are rejected. Evaluate BMD-HS disease labels as separate binary tasks, with
missing targets explicitly excluded. CirCor murmur labels are not five-class
valve diagnoses.

Binary ROC uses class 1. Multiclass ROC uses macro one-vs-rest, including every
declared probability column. A missing class gives NaN for ROC metrics; F1 and
recall use zero for absent classes. Sensitivity is the best empirical sensitivity
whose specificity is at least 90%, without interpolation or splitting tied
scores. Confirm this operating-point convention against the full proposal if
it specifies interpolation. ECE uses 15 equal-width confidence bins and includes
zero and one. Binary predictions use P(class 1) >= 0.5.

`patient_bootstrap` resamples whole patients with replacement, retaining all
recordings and the multiplicity of each selected patient. Defaults: 1,000
resamples, seed 0, 95% percentile interval. The point metric is recording-
weighted. Any scalar metric can be passed, including per-class recall.
Nonfinite resample values are excluded and the returned object reports valid
and total counts; all-undefined samples produce NaN bounds. Other errors
propagate. Prefix patient IDs with dataset names when pooling. Yaseen provides
clip IDs rather than real patient IDs: resampling those describes uncertainty
over clips rather than over patients.

## Corrupt a WAV clip

```powershell
python -m cardio.eval.corrupt input.wav white.wav --noise white --snr-db 10 --seed 0
python -m cardio.eval.corrupt input.wav pink.wav --noise pink --snr-db 10 --seed 0
python -m cardio.eval.corrupt input.wav lowpass.wav --cutoff-hz 400
```

Noise is scaled to exact aggregate RMS SNR. Pink noise has 1/f power shaping
and zero DC. Channels receive independent noise. Sample rate and channel shape
are preserved. Silence has undefined SNR and raises an error. Output is float32
WAV without clipping or normalization; peaks above one are preserved. Existing
files are rejected.

The corruption low-pass is an offline zero-phase Butterworth filter, order 4
by default, with an explicit cutoff below Nyquist. It does not replace or modify
the causal filter in `core.features`. Clips too short for zero-phase filtering
fail explicitly. With both options, noise is applied before filtering; the
requested SNR describes the addition before filtering.

Run all project tests with `python -m pytest tests/`.
