param([switch]$Download, [switch]$InstallDependencies)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $env:PYTHONUTF8 = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Create the Python 3.10 .venv first; see docs/LOCAL_SETUP.md.'
    }
    function Invoke-Python {
        & $pythonPath @args
        if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
    }
    if ($InstallDependencies) {
        Invoke-Python -m pip install -r requirements.txt streamlit
    }
    if ($Download) {
        Invoke-Python core/download_datasets.py --dataset yaseen
        if (-not (Test-Path -LiteralPath 'data/raw/bmdhs/.git')) {
            git clone https://github.com/sani002/BMD-HS-Dataset data/raw/bmdhs
            if ($LASTEXITCODE -ne 0) { throw 'BMD-HS clone failed.' }
        }
        git -C data/raw/bmdhs lfs install --local
        if ($LASTEXITCODE -ne 0) { throw 'Git LFS setup failed.' }
        git -C data/raw/bmdhs lfs pull
        if ($LASTEXITCODE -ne 0) { throw 'BMD-HS LFS download failed.' }
        Invoke-Python -m cardio.data.download_unseen
    }
    Invoke-Python -m cardio.data.download_unseen --verify-only
    Invoke-Python -m core.build_cardio_manifest
    git diff --exit-code HEAD -- data/manifests/yaseen_manifest.csv
    if ($LASTEXITCODE -ne 0) { throw 'Yaseen manifest differs from committed folds.' }
    Invoke-Python -m core.preprocess --dataset yaseen
    Invoke-Python -m cardio.data.build_unseen --dataset bmdhs
    Invoke-Python -m cardio.data.build_unseen --dataset circor
    Invoke-Python -m pytest tests/
} finally {
    Pop-Location
}
