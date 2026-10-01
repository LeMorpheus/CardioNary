"""The shared contracts reject bad values where they are made.

Run:
    python -m pytest tests/test_contracts.py -v
"""

from __future__ import annotations

import pytest

from core.contracts import (
    METADATA_WEIGHT_CAP,
    FusedOutput,
    ModelOutput,
    PatientMetadata,
    VisitDecision,
)

HEART = {"Normal": 0.7, "AS": 0.1, "MS": 0.1, "MR": 0.05, "MVP": 0.05}


def test_model_output_reports_probability_of_abnormal():
    out = ModelOutput("heart", HEART, secondary={"heart_rate_bpm": 78.0})
    assert out.p_abnormal == pytest.approx(0.3)


def test_model_output_needs_exactly_the_modality_classes():
    with pytest.raises(ValueError, match="classes"):
        ModelOutput("xray", HEART)
    with pytest.raises(ValueError, match="classes"):
        ModelOutput("heart", {"Normal": 0.5, "AS": 0.5})


def test_model_output_probabilities_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ModelOutput("xray", {"Normal": 0.6, "Pneumonia": 0.6})


def test_unmeasured_secondary_characteristic_is_left_out_not_nan():
    with pytest.raises(ValueError, match="finite"):
        ModelOutput("heart", HEART, secondary={"heart_rate_bpm": float("nan")})


def test_patient_metadata_is_optional_and_checked():
    PatientMetadata()                                   # nothing recorded is valid
    PatientMetadata(age_years=67, sex="female", symptoms={"breathlessness"})
    with pytest.raises(ValueError, match="age_years"):
        PatientMetadata(age_years=-1)
    with pytest.raises(ValueError, match="unknown symptoms"):
        PatientMetadata(symptoms={"headache"})


def test_fusion_weights_sum_to_one_and_metadata_is_capped():
    FusedOutput("heart", 0.4, votes={"cnn": 0.5, "features": 0.3, "metadata": 0.2},
                weights={"cnn": 0.5, "features": 0.3, "metadata": 0.2})
    with pytest.raises(ValueError, match="sum to 1"):
        FusedOutput("heart", 0.4, votes={"cnn": 0.5, "features": 0.3},
                    weights={"cnn": 0.5, "features": 0.3})
    with pytest.raises(ValueError, match="exceeds the cap"):
        FusedOutput("xray", 0.4, votes={"cnn": 0.5, "metadata": 0.2},
                    weights={"cnn": 1 - METADATA_WEIGHT_CAP - 0.1,
                             "metadata": METADATA_WEIGHT_CAP + 0.1})


def test_a_missing_vote_is_left_out_but_the_cnn_vote_is_required():
    FusedOutput("xray", 0.5, votes={"cnn": 0.5}, weights={"cnn": 1.0})
    with pytest.raises(ValueError, match="CNN vote"):
        FusedOutput("xray", 0.5, votes={"features": 0.5}, weights={"features": 1.0})


def test_visit_decision_needs_at_least_one_usable_test():
    VisitDecision(refer=True, p_any=0.44, cost_ratio=4.0, p_heart=0.2, p_xray=0.3)
    VisitDecision(refer=False, p_any=0.1, cost_ratio=4.0, p_xray=0.1)
    with pytest.raises(ValueError, match="at least one"):
        VisitDecision(refer=False, p_any=0.0, cost_ratio=4.0)


def test_metadata_has_no_acquisition_fields():
    # View position and device describe how the input was acquired, not the
    # patient; a classifier given them learns a shortcut.
    fields = set(PatientMetadata.__dataclass_fields__)
    assert not fields & {"view_position", "device", "stethoscope", "scanner"}
