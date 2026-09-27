"""Anti-leakage rules as executable tests (docs/05_DATASET_SELECTION.md section 5).

These are the tests that make the reported accuracies mean something. Run them
before believing any number:

    .venv-ml/Scripts/python -m pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
CACHE_DIR = REPO_ROOT / "data" / "cache"

PATIENT_DISJOINT = {"kauh", "icbhi"}      # corpora with real subject grouping
ALL_MANIFESTS = ["kauh", "yaseen", "icbhi", "kermany"]


def _manifest(name: str) -> pd.DataFrame:
    p = MANIFEST_DIR / f"{name}_manifest.csv"
    if not p.exists():
        pytest.skip(f"{p.name} not built yet")
    return pd.read_csv(p)


@pytest.mark.parametrize("name", ALL_MANIFESTS)
def test_rule_5_1_folds_are_unit_disjoint(name):
    """No patient (or clip, for corpora without subjects) spans two folds."""
    df = _manifest(name)
    spans = df.groupby("patient_id")["fold"].nunique()
    offenders = spans[spans > 1]
    assert offenders.empty, f"{name}: units in multiple folds: {offenders.index.tolist()[:10]}"

    for f in sorted(df.fold.unique()):
        train = set(df.loc[df.fold != f, "patient_id"])
        test = set(df.loc[df.fold == f, "patient_id"])
        overlap = train & test
        assert not overlap, f"{name} fold {f} leaks: {sorted(overlap)[:10]}"


def test_rule_5_2_single_kauh_filter_rendering():
    """KAUH B/D/E are the same acoustic event; only one may be used."""
    df = _manifest("kauh")
    assert set(df.filter_code) == {"D"}, (
        f"expected only the Diaphragm rendering, found {sorted(set(df.filter_code))}"
    )


@pytest.mark.parametrize("name", ["kauh", "yaseen", "icbhi"])
def test_no_duplicate_decoded_audio(name):
    """Hash the decoded PCM, not the file: catches re-encoded duplicates.

    This is the rule that found the DP100/DP101 label contradiction in KAUH.
    Audio corpora only -- for image corpora an exact duplicate is common and
    harmless; what matters there is conflicting labels (tested below).
    """
    df = _manifest(name)
    dupes = df[df.pcm_md5.duplicated(keep=False)]
    assert dupes.empty, (
        f"{name}: duplicate PCM content in "
        f"{dupes.filename.tolist()[:6]}"
    )


@pytest.mark.parametrize("name", ALL_MANIFESTS)
def test_no_conflicting_labels_for_identical_audio(name):
    """Identical audio must never carry two different labels."""
    df = _manifest(name)
    hashcol = "pcm_md5" if "pcm_md5" in df.columns else "md5"
    cls = "binary_class" if "binary_class" in df.columns else "diagnosis"
    conflict = df.groupby(hashcol)[cls].nunique()
    bad = conflict[conflict > 1]
    assert bad.empty, f"{name}: {len(bad)} identical items carry conflicting labels"


@pytest.mark.parametrize("name", ALL_MANIFESTS)
def test_labels_are_contiguous_and_match_diagnosis(name):
    df = _manifest(name)
    cls = "binary_class" if "binary_class" in df.columns else "diagnosis"
    mapping = df.groupby(cls)["label"].nunique()
    assert (mapping == 1).all(), f"{name}: a diagnosis maps to multiple label ids"
    labels = sorted(df.label.unique())
    assert labels == list(range(len(labels))), f"{name}: labels not contiguous: {labels}"


@pytest.mark.parametrize("name,feature", [("kauh", "logmel"), ("kauh", "mfcc"),
                                          ("yaseen", "logmel"), ("icbhi", "logmel")])
def test_rule_5_4_windows_never_straddle_a_fold(name, feature):
    """Every window of a recording sits in exactly one fold."""
    p = CACHE_DIR / f"{name}_{feature}.npz"
    if not p.exists():
        pytest.skip(f"{p.name} not built yet")
    c = np.load(p, allow_pickle=True)
    df = pd.DataFrame({"rec": c["rec"], "fold": c["fold"], "pid": c["patient_id"]})
    assert (df.groupby("rec")["fold"].nunique() == 1).all(), \
        f"{name}/{feature}: windows of one recording landed in different folds"
    assert (df.groupby("pid")["fold"].nunique() == 1).all(), \
        f"{name}/{feature}: windows of one patient landed in different folds"


@pytest.mark.parametrize("node", ["cardio", "pulmo"])
def test_rule_5_6_golden_vectors_come_from_a_held_out_fold(node):
    """Vectors flashed to the device must be samples the model never trained on.

    The corpus is read from the vector manifest rather than hard-coded, so this
    keeps working when a node changes dataset or modality.
    """
    import json

    vm = REPO_ROOT / "vectors" / f"chiron_{node}_vectors_manifest.json"
    qr = REPO_ROOT / "reports" / f"quantization_chiron_{node}_v1.json"
    if not vm.exists() or not qr.exists():
        pytest.skip(f"{node} vectors not exported yet")

    v = json.loads(vm.read_text())
    q = json.loads(qr.read_text())
    assert v["holdout_fold"] == q["holdout_fold"], (
        f"{node}: vectors from fold {v['holdout_fold']} but model held out "
        f"fold {q['holdout_fold']}"
    )

    man = _manifest(v["dataset"])
    held = set(man.loc[man.fold == v["holdout_fold"], "filename"])
    for entry in v["vectors"]:
        assert entry["source_file"] in held, (
            f"{node}: golden vector {entry['source_file']} is NOT in held-out "
            f"fold {v['holdout_fold']} of {v['dataset']}"
        )
