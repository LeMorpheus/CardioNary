"""Check actual unseen manifests/caches when datasets have been prepared locally."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from core.features import CARDIO_CONFIG

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('dataset', ['bmdhs', 'circor'])
def test_unseen_manifest_and_cache_contract(dataset):
    manifest = ROOT / f'data/manifests/{dataset}_manifest.csv'
    if not manifest.exists():
        pytest.skip(f'{dataset} has not been prepared locally')
    if not (ROOT / 'data/raw' / dataset).is_dir():
        pytest.skip(f'{dataset} raw data are local-only and absent on this machine')
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert len({row['path'] for row in rows}) == len(rows)
    for row in rows:
        assert row['dataset'] == dataset
        assert row['patient_id']
        assert row['label_scope'] == 'patient'
        assert not Path(row['path']).is_absolute()
        assert (ROOT / row['path']).is_file()
        assert isinstance(json.loads(row['labels']), dict)
        assert isinstance(json.loads(row['metadata']), dict)
        if row['cache_path']:
            assert not Path(row['cache_path']).is_absolute()
            cache_path = ROOT / row['cache_path']
            if not cache_path.exists():
                pytest.skip('Feature caches are local-only and not present on this machine')
            with np.load(cache_path, allow_pickle=False) as cache:
                features = cache['features']
                assert features.ndim == 3
                assert features.shape[1:] == CARDIO_CONFIG.shape
                assert features.dtype == np.float32
                assert np.isfinite(features).all()
