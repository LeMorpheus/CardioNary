"""Tests for lung late fusion and metadata vote contracts."""

import numpy as np
import pandas as pd
import pytest

from core.contracts import FusedOutput, METADATA_WEIGHT_CAP
from lung.fusion.late_fusion import fuse_dataset
from lung.fusion.meta_vote import run_meta_vote


def test_fused_output_enforces_metadata_cap():
    # Attempting metadata weight > 0.3 should raise ValueError in FusedOutput
    with pytest.raises(ValueError, match="exceeds the cap"):
        FusedOutput(
            modality="xray",
            p_abnormal=0.5,
            votes={"cnn": 0.5, "features": 0.5, "metadata": 0.5},
            weights={"cnn": 0.3, "features": 0.3, "metadata": 0.4},
        )


def test_fused_output_requires_cnn():
    with pytest.raises(ValueError, match="CNN vote is always present"):
        FusedOutput(
            modality="xray",
            p_abnormal=0.5,
            votes={"features": 0.5},
            weights={"features": 1.0},
        )


def test_fused_output_valid_kermany_fusion():
    fo = FusedOutput(
        modality="xray",
        p_abnormal=0.85,
        votes={"cnn": 0.9, "features": 0.6},
        weights={"cnn": 0.95, "features": 0.05},
    )
    assert fo.p_abnormal == 0.85
    assert sum(fo.weights.values()) == pytest.approx(1.0)
    assert "metadata" not in fo.weights


def test_fuse_dataset_mock(tmp_path):
    # Create synthetic OOF and feature vote DataFrames
    n = 100
    df_oof = pd.DataFrame({
        "index": np.arange(n),
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "fold": [i % 5 for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "p_normal": np.linspace(0.8, 0.2, n),
        "p_pneumonia": np.linspace(0.2, 0.8, n),
    })
    df_plr = pd.DataFrame({
        "index": np.arange(n),
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "fold": [i % 5 for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "pLR": np.linspace(0.3, 0.7, n),
    })

    out_csv = tmp_path / "fused.csv"
    out_json = tmp_path / "fused.json"

    fused_outputs = fuse_dataset(
        df_oof=df_oof,
        df_plr=df_plr,
        df_pm=None,
        dataset_name="synthetic",
        out_csv=out_csv,
        out_json=out_json,
    )

    assert len(fused_outputs) == n
    assert out_csv.exists()
    assert out_json.exists()
    for fo in fused_outputs:
        assert isinstance(fo, FusedOutput)
        assert fo.modality == "xray"
        assert 0.0 <= fo.p_abnormal <= 1.0
        assert sum(fo.weights.values()) == pytest.approx(1.0)
        assert fo.weights.get("metadata", 0.0) <= METADATA_WEIGHT_CAP
