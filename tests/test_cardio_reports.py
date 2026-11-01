"""Checks on the heart branch's committed result files.

Each test is skipped until its file exists, so the suite passes on a fresh clone.

Run:
    python -m pytest tests/test_cardio_reports.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
CLASSES = ["Normal", "AS", "MS", "MR", "MVP"]


def _load_csv(name):
    p = REPORTS / name
    if not p.exists():
        pytest.skip(f"{name} not produced yet")
    return pd.read_csv(p)


def test_oof_file_covers_every_clip_once_with_valid_probabilities():
    oof = _load_csv("heart_oof_yaseen.csv")
    manifest = pd.read_csv(REPO_ROOT / "data" / "manifests" / "yaseen_manifest.csv")
    assert oof.patient_id.is_unique
    assert set(oof.patient_id) <= set(manifest.patient_id)
    # every clip long enough for one 3-second window is scored exactly once
    assert len(oof) >= 0.98 * len(manifest)
    assert sorted(oof.fold.unique()) == [0, 1, 2, 3, 4]
    probs = oof[["p_" + c for c in CLASSES]]
    assert ((probs >= 0) & (probs <= 1)).all().all()
    assert (probs.sum(axis=1) - 1).abs().max() < 1e-3


def test_oof_folds_and_labels_match_the_manifest():
    oof = _load_csv("heart_oof_yaseen.csv").set_index("patient_id")
    manifest = pd.read_csv(REPO_ROOT / "data" / "manifests" / "yaseen_manifest.csv").set_index("patient_id")
    m = manifest.loc[oof.index]
    assert (oof.fold == m.fold).all()
    assert (oof.label == m.label).all()


def test_clinic_report_holds_aggregates_only():
    p = REPORTS / "heart_real_world.json"
    if not p.exists():
        pytest.skip("heart_real_world.json not produced yet")
    text = p.read_text()
    # no patient IDs (p1 ... p10) or file names may appear anywhere in the file
    assert not re.search(r"\bp\d+(_|\b)", text)
    assert ".wav" not in text
    out = json.loads(text)
    assert out["recordings"] == out["quality_check"]["passed"] + out["quality_check"]["failed"]
