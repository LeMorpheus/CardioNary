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


def _load_json(name):
    p = REPORTS / name
    if not p.exists():
        pytest.skip(f"{name} not produced yet")
    return p.read_text()


def _intervals_bracket_estimates(node):
    """Every interval anywhere in a report must contain its point estimate."""
    if isinstance(node, dict):
        if {"estimate", "lower_95", "upper_95"} <= node.keys():
            assert node["lower_95"] - 1e-9 <= node["estimate"] <= node["upper_95"] + 1e-9
        for v in node.values():
            _intervals_bracket_estimates(v)
    elif isinstance(node, list):
        for v in node:
            _intervals_bracket_estimates(v)


def test_cv_intervals_present_and_consistent():
    cv = json.loads(_load_json("heart_cv_yaseen.json"))
    block = cv["with_intervals"]
    assert set(block["five_class"]["per_class_recall"]) == set(CLASSES)
    assert block["five_class"]["accuracy"]["resamples"] == 1000
    _intervals_bracket_estimates(block)


def test_requantisation_report_is_consistent_and_clinic_part_aggregate_only():
    text = _load_json("heart_requantisation.json")
    out = json.loads(text)
    for site in ("physionet_2022", "clinic"):
        if site not in out:
            continue
        variants = out[site]["variants"]
        assert set(variants) == {"A_original", "B_requantised", "C_adapted_requantised"}
        for v in variants.values():
            assert 0 <= v["recording_agreement"] <= 1
            assert 0 <= v["refer_or_not_agreement"] <= 1
            assert 0 <= v["share_of_inputs_clipped"] <= 1
            binned = sum(b["recordings"] for b in v["agreement_by_float_margin"].values())
            assert binned == out[site]["measured_recordings"]
        _intervals_bracket_estimates(out[site])
    if "clinic" in out:
        clinic = json.dumps(out["clinic"])
        assert not re.search(r"\bp\d+(_|\b)", clinic) and ".wav" not in clinic


def test_edge_budget_report_covers_every_stage():
    from cardio.predict import STAGES

    out = json.loads(_load_json("heart_edge_budget.json"))
    for timing in out["timing_per_case"].values():
        for stage in (*STAGES, "total"):
            assert timing[f"{stage}_ms"]["median"] >= 0
        assert 0 <= timing["passed_quality_check"] <= timing["recordings"]
    assert out["sizes"]["int8_bytes"] > 0
    assert set(out["stages"]) == set(STAGES)
    for survey in out["quality_check_on_public_data"].values():
        assert 0 <= survey["share_passing"]["estimate"] <= 1
    _intervals_bracket_estimates(out["quality_check_on_public_data"])


def test_interval_helpers_on_toy_data():
    import numpy as np
    from cardio.eval.intervals import agreement_interval, share_abnormal, statistic_interval

    prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    groups = np.array(["a", "a", "b", "c"])
    a = agreement_interval(np.array([0, 1, 0, 0]), prob, groups)
    assert a["estimate"] == pytest.approx(0.75)
    assert a["lower_95"] <= 0.75 <= a["upper_95"]
    s = statistic_interval(prob, groups, share_abnormal)
    assert s["estimate"] == pytest.approx(0.5)
