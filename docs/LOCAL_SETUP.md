# Shivansh's local heart-sound setup

This checkout is the active `LeMorpheus/CardioNary` repository, started from
`cardio` at `59823e4385ff2cbd60ef7f3ae598c6229c31bae1`, on the local branch
`cardio-shivansh-unseen-metrics`. The downloaded `CardioNary_Baseline-main.zip`
is an older frozen baseline whose README points to this active repository.
The baseline archive was also extracted into a separate sibling directory.

Use the active checkout for the assignment:

```powershell
cd 'C:\Users\shiva\Documents\ChatGPT\Embedded presentation\CardioNary'
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = '1'
python --version
python -m pytest tests/
```

Python 3.10.20 is installed for this checkout. The environment uses the exact
versions in `requirements.txt`, plus Streamlit. Invoking the interpreter
directly also works if PowerShell's execution policy blocks activation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

To recreate the environment from scratch with the installed uv tool:

```powershell
uv python install 3.10
uv venv --seed --python 3.10 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt streamlit
```

## Reproduce the data preparation

```powershell
.\scripts\prepare_cardio.ps1
```

This checks CirCor's published SHA256 hashes, rebuilds the Yaseen manifest and
rejects a diff against Git, builds Yaseen/BMD-HS/CirCor feature caches through the
shared front-end, then runs the tests. Add `-Download` to fetch missing datasets,
or `-InstallDependencies` to reinstall `requirements.txt` and Streamlit.

The Yaseen source filenames contain Korean characters. `PYTHONUTF8=1` prevents
Windows console encoding errors. If raw.githubusercontent.com is unreachable,
the same archives can be fetched using the official repository's GitHub blob
API (base64 payloads), keeping their original filenames and bytes. That fallback
was used for the initial local download. The shared downloader then extracted
them using the installed 7-Zip.

The official PhysioNet server did not respond during initial setup, so CirCor
was fetched from its documented public S3 mirror using the AWS CLI in an
isolated uv tool environment. This does not need AWS credentials and does not
modify the user's global AWS configuration.

## Added implementation

- `cardio/data/build_unseen.py`: recording manifests and actual core feature caches.
- `cardio/data/download_unseen.py`: public CirCor download and SHA256 verification.
- `cardio/eval/metrics.py`: metrics and 95% patient bootstrap intervals.
- `cardio/eval/corrupt.py`: single-clip white noise, pink noise, and low-pass filter.
- Tests for metric values, clustering, SNR, filtering, dataset formats, real
  feature extraction, and download integrity.

See `cardio/data/README.md` and `cardio/eval/README.md` for APIs and conventions.
The full proposal was not provided; operating-point interpolation and a future
cross-dataset task mapping need confirmation if the proposal specifies them.
No raw data, caches, or model weights should be committed.

## AI assistance

OpenAI Codex assisted with local setup, implementation, tests, and documentation.
Validation uses known synthetic metric values, measured SNR/filter responses,
the existing core front-end, dataset checksums, and the project's test suite.
Commits carry a Co-Authored-By trailer as required by the repository README.

## Verified on 1 November 2026

- Python 3.10.20; exact pinned requirements installed, plus Streamlit 1.65.0.
  `python -m pip check` reports no broken requirements.
- Yaseen: 1,000 recordings; rebuilt manifest matches committed content and
  folds with an empty Git diff. Cache: 1,044 windows of shape `(40, 192)`.
  Exactly 10 recordings shorter than 1.5 seconds yield no window, as expected.
- BMD-HS: 872 real WAV recordings, 109 patient IDs, 872 feature caches.
  Source revision: `4745751855d15ff5fc9aaed29c32ad3ec7a5d409`.
- CirCor 1.0.3: all 10,434 published SHA256 entries verified; 3,163 recordings
  and 3,163 feature caches. Repeat-visit linkage yields 872 patient groups.
- Full test suite: **46 passed, 20 skipped, no failures**. Skips require
  KAUH/ICBHI datasets and caches, pretrained model assets, or exported vectors
  that are not supplied in this active checkout.
- All unseen manifest paths and cache references exist. Artifact tests read
  every cache and check its shape, float32 type, and finite values.
- Real-clip corruption examples are under ignored `scratch/noise_demo/`:
  white and pink noise at 10 dB SNR, plus a 400 Hz low-pass output.
- The PowerShell setup script parses successfully. Raw datasets, feature
  caches, and model weights are excluded from the local code commit.
