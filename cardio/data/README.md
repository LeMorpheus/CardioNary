# Unseen heart-sound datasets

Run these commands from the repository root in the Python 3.10 `.venv`:

```powershell
$env:PYTHONUTF8 = '1'
git clone https://github.com/sani002/BMD-HS-Dataset data/raw/bmdhs
git -C data/raw/bmdhs lfs install --local
git -C data/raw/bmdhs lfs pull
python -m cardio.data.download_unseen
python -m cardio.data.build_unseen --dataset bmdhs
python -m cardio.data.build_unseen --dataset circor
```

The CirCor downloader uses the public S3 mirror linked on
[PhysioNet's official release page](https://www.physionet.org/content/circor-heart-sound/1.0.3/).
It requires the AWS CLI or uv, resumes existing files, and verifies every entry
in the release's published SHA256SUMS. Use `--verify-only` to recheck completed
downloads. No cloud account is needed. BMD-HS comes from the
[public dataset repository](https://github.com/sani002/BMD-HS-Dataset).

The importer defaults to `core.features.features_for_file(path, CARDIO_CONFIG)`.
Use `--manifest-only` to skip caches. Recording manifests go into
`data/manifests/{bmdhs,circor}_manifest.csv`, and per-recording compressed NPZ
caches go into `data/cache/unseen/`. `features` is a float32 array with shape
`(n_windows, 40, 192)`; a clip shorter than 1.5 seconds produces zero windows,
matching the shared pipeline. `*_config.json` records the front-end settings.
Caches are rebuilt on each invocation. Separate cache directories are needed
to retain multiple configurations. Individual outputs are replaced atomically;
an entire dataset run is not transactional.
The CLI extracts four recordings concurrently; use `--workers 1` for serial
extraction. Manifest ordering stays deterministic for any worker count.

Each manifest has one row per recording: dataset, repository-relative path,
patient ID, label scope, labels, metadata, and repository-relative cache path.
Labels and metadata are JSON objects inside CSV cells. No folds are assigned
to these held-out datasets. BMD-HS labels remain multilabel disease indicators,
with missing labels represented by null. CirCor patient-level murmur and outcome
labels remain separate; a patient's murmur need not be audible in every location.
Outcome stays null if absent from this release. No five-class Yaseen mapping is
invented. Metadata remain source values; a metadata classifier must convert age
groups explicitly rather than guessing precise measurements.

CirCor `Additional ID` links repeat visits. The importer groups linked visits
under one patient ID and preserves the original visit ID in metadata as
`subject_id`. Prefix patient IDs with dataset names when pooling cohorts.
Missing recordings, duplicate paths/subject descriptions, invalid labels, or
paths outside a dataset root fail explicitly. Default dataset locations produce
portable manifests. External `--root` paths are supported but generate absolute
paths and should not be committed.

Only manifests and code belong in Git. Raw datasets, caches, and trained weights
remain local. Downloaded source license files are retained with the datasets.
