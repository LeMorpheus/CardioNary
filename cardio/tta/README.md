# cardio/tta — test-time adaptation

A model trained on one set of stethoscopes and recording conditions will meet
different ones at a new site. This workstream adapts the model at test time, from
that site's own unlabelled recordings, so it becomes instrument-agnostic.

Evaluated three ways: accuracy on shifted data before adaptation, after adaptation,
and on the original data after adaptation — to show nothing was forgotten.

Run the S3 public-data study from the repository root with the team model and
its matching quantization report available locally.

```powershell
python -m cardio.eval.run_tta --model-dir 'C:/path/to/models' --report-dir 'C:/path/to/reports'
```

The runner writes `reports/heart_tta.json` after all four sites finish. It uses
five draws per adaptation size (8, 16, 32, and 64 recordings), plus five draws
at each of 10%, 50%, and 90% abnormal at size 32. Labels select the stress
batches and score the results. The adaptation step receives only audio
features. All recordings of a scoring patient stay out of the adaptation
pool, and the scoring patients stay fixed across draws. Yaseen fold 0 uses
clip IDs because the dataset does not provide patient IDs.

The report includes macro-F1, AUROC, their recovery fractions, clean Yaseen
scores after adaptation, and the timing and collapse decision for each draw.
The recovery denominator is the clean score minus the shifted plain score.
For the two Yaseen corruptions, clean and shifted scores use the same clips.
For BMD-HS and CirCor, the clean score is a separate Yaseen benchmark. Recovery
is undefined when the clean score is no higher than the shifted plain score.

For a quick local check with reference weights:

```powershell
python -m cardio.eval.run_tta --model-dir scratch/s3-reference/models --reference-model --sizes 32 --draws 1 --imbalance-size 0 --max-score-patients 24
```

The reference run is marked provisional and defaults to a report in `scratch/`.
`bn_adapt.adapt_bn` returns the selected Keras model and collapse diagnostics.
The caller can save an accepted adapted copy with Keras `save` if needed.
