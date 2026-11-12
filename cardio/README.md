# cardio — heart-sound team (3 people)

Team branch: `cardio`. Your own branches: `cardio-<your name>-<topic>`.

| Work | Folder | Owner |
|---|---|---|
| Heart-sound model, 8-bit model, clinic recordings, tool | `model/`, `eval/` | Aviral |
| Secondary characteristics; feature and metadata votes; late fusion; calibration | `fusion/` | Anshuman |
| Unseen datasets; test-time adaptation; noise tests | `tta/`, `eval/` | Shivansh |

Write up every experiment, including the ones that fail, in `experiments/`.
New code goes in this folder; `core/` is only imported, never changed.

## Running what exists so far

```bash
# once: the Yaseen data and its spectrogram cache
python core/download_datasets.py --dataset yaseen
python -m core.preprocess --dataset yaseen

# base CNN, 5-fold cross-validation with out-of-fold probabilities
python -m cardio.model.cv_oof              # reports/heart_oof_yaseen.csv, heart_cv_yaseen.json
python -m cardio.model.cv_oof --from-oof   # only the 95% intervals, from the saved file

# final model (folds 1-4) and its 8-bit copy, into models/
python -m core.quantize --dataset yaseen --holdout-fold 0

# PhysioNet 2022 manifest and caches (once; see cardio/data/README.md)
python -m cardio.data.build_unseen --dataset circor

# 8-bit against float agreement: Yaseen fold 0, and PhysioNet 2022 by patient
python -m cardio.eval.agreement --manifest data/manifests/circor_manifest.csv --name physionet_2022

# clinic recordings, aggregates only; run only on the laptop that holds them
python -m cardio.eval.real_world --data-dir <path to Real_World_Data>

# the same, plus test-time adaptation on the clinic recordings (roadmap S3)
python -m cardio.eval.real_world --data-dir <path to Real_World_Data> --adapt

# re-quantisation for a new site (roadmap S4): shipped vs re-quantised vs adapted
python -m cardio.eval.requantise --site physionet_2022
python -m cardio.eval.requantise --site clinic --data-dir <path to Real_World_Data>

# edge budget on this laptop (roadmap S4, S5): size, time per case through
# cardio/predict.py, memory, and the share of public recordings passing the quality check
python -m cardio.eval.edge_budget          # reports/heart_edge_budget.json

# the heart result for one recording, as the dashboard shows it (public data only)
python -m cardio.predict data/raw/circor/training_data/<recording>.wav

# every result file again, in order; skips what this laptop has no code or data for
python -m cardio.run_all --list
python -m cardio.run_all [--clinic-dir <path to Real_World_Data>] [--final]
```

## For the dashboard

`cardio/predict.py` is the heart branch's one entry point (roadmap S5; frontend
guide, Section 3). The dashboard calls `analyse_heart(recording, metadata)` with an
upload or a path and a `PatientMetadata`, and gets back the contracts of
`core/contracts.py`: a `ModelOutput` (calibrated class probabilities and the 17
characteristics) and a `FusedOutput` (p_h, each vote and its weight). When
`withheld` is set (`"unreadable"`, `"too_short"` or `"quality_check"`), the
dashboard shows the reason and no heart result. `result.pcm16()` feeds
`core.spectrogram.spectrogram_png`, and `result.close_call` names the top two
classes when they are less than 0.05 apart.

Until `heart_fusion()` and `reports/heart_fusion_params.json` exist, the CNN vote
counts alone with weight 1, and T = 1 until `"T"` is in that file or in
`reports/heart_calibration.json`.

Every result carries a 95% bootstrap interval over patients, from
`cardio.eval.intervals`, which wraps `cardio.eval.metrics`.

`reports/heart_oof_yaseen.csv` has one row per clip: its fold, label, and the
mean softmax over its windows from the model that did not train on that fold.
Calibration and late fusion use these probabilities.
