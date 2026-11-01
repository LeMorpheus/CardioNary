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

# final model (folds 1-4) and its 8-bit copy, into models/ (not in Git)
python -m core.quantize --dataset yaseen --holdout-fold 0

# 8-bit against float agreement on Yaseen fold 0 (optionally a folder of recordings)
python -m cardio.eval.agreement            # reports/heart_agreement.json

# clinic recordings, aggregates only; run only on the laptop that holds them
python -m cardio.eval.real_world --data-dir <path to Real_World_Data>
```

`reports/heart_oof_yaseen.csv` has one row per clip: its fold, label, and the
mean softmax over its windows from the model that did not train on that fold.
Calibration and late fusion use these probabilities.
