import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cardio.data.build_unseen import bmdhs_rows, circor_rows, build_caches, write_manifest


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def bmd_fixture(self):
        (self.root/'train').mkdir()
        for name in ['a.wav', 'b.wav']:
            (self.root/'train'/name).touch()
        (self.root/'train.csv').write_text(
            'patient_id,AS,AR,MR,MS,N,recording_1,recording_2\np1,1,1,N/A,0,0,a,b\n')
        (self.root/'additional_metadata.csv').write_text('patient_id,Age,Gender\np1,30-39,M\n')

    def test_bmd_expands_recordings_preserves_multilabel_missing_and_metadata(self):
        self.bmd_fixture()
        rows = bmdhs_rows(self.root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['patient_id'], rows[1]['patient_id'])
        self.assertEqual(rows[0]['labels'], dict(AS=1, AR=1, MR=None, MS=0, N=0))
        self.assertEqual(rows[0]['metadata']['Age'], '30-39')
        self.assertEqual(rows[0]['path'], 'train/a.wav')

    def test_missing_recording_fails(self):
        self.bmd_fixture()
        (self.root/'train/a.wav').unlink()
        with self.assertRaises(FileNotFoundError):
            bmdhs_rows(self.root)

    def test_duplicate_recording_fails(self):
        self.bmd_fixture()
        path = self.root/'train.csv'
        path.write_text(path.read_text().replace(',a,b', ',a,a'))
        with self.assertRaises(ValueError):
            bmdhs_rows(self.root)

    def test_circor_format_and_missing_labels(self):
        (self.root/'123_AV.wav').touch()
        (self.root/'123_MV.wav').touch()
        description = self.root/'123.txt'
        description.write_text('123 2 4000\nAV 123_AV.hea 123_AV.wav 123_AV.tsv\nMV 123_MV.hea 123_MV.wav 123_MV.tsv\n#Age: Child\n#Murmur: Present\n#Outcome: Abnormal\n')
        rows = circor_rows(self.root)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['labels'], dict(murmur='Present', outcome='Abnormal'))
        self.assertEqual(rows[0]['metadata']['location'], 'AV')
        self.assertEqual(rows[0]['label_scope'], 'patient')
        description.write_text('123 1 4000\nAV 123_AV.hea 123_AV.wav\n')
        self.assertEqual(circor_rows(self.root)[0]['labels'], dict(murmur=None, outcome=None))

    def test_caches_call_extractor_and_roundtrip_manifest(self):
        self.bmd_fixture()
        config, calls = {'sample_rate': 4000}, []
        def features(path, received_config):
            self.assertIs(received_config, config)
            self.assertTrue(Path(path).exists())
            calls.append(path)
            return {'values': np.array([1, 2, 3])}
        rows = build_caches(bmdhs_rows(self.root), self.root, self.root/'cache', config, features)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(rows[0]['cache_path'], rows[1]['cache_path'])
        with np.load(self.root/'cache'/rows[0]['cache_path'], allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached['values'], [1, 2, 3])
        output = self.root/'manifest.csv'
        write_manifest(rows, output)
        with output.open(newline='') as stream:
            written = list(csv.DictReader(stream))
        self.assertEqual(json.loads(written[0]['labels']), rows[0]['labels'])
        self.assertFalse(Path(written[0]['path']).is_absolute())

    def test_object_cache_rejected(self):
        self.bmd_fixture()
        with self.assertRaises(TypeError):
            build_caches(bmdhs_rows(self.root), self.root, self.root/'cache', {}, lambda p,c: {'x': object()})

    def test_real_frontend_cache_shape_and_short_tail(self):
        from core.features import CARDIO_CONFIG
        from scipy.io import wavfile
        self.bmd_fixture()
        for name, n in [('a.wav', 80000), ('b.wav', 1000)]:
            signal = (1000*np.sin(2*np.pi*100*np.arange(n)/4000)).astype(np.int16)
            wavfile.write(self.root/'train'/name, 4000, signal)
        rows = build_caches(bmdhs_rows(self.root), self.root, self.root/'cache', CARDIO_CONFIG, workers=2)
        with np.load(self.root/'cache'/rows[0]['cache_path']) as cache:
            self.assertEqual(cache['features'].shape[1:], CARDIO_CONFIG.shape)
            self.assertGreater(cache['features'].shape[0], 0)
            self.assertEqual(cache['features'].dtype, np.float32)
        with np.load(self.root/'cache'/rows[1]['cache_path']) as cache:
            self.assertEqual(cache['features'].shape, (0, *CARDIO_CONFIG.shape))
            self.assertEqual(cache['features'].dtype, np.float32)

    def test_circor_repeat_visit_ids_share_one_patient(self):
        for patient, other in [('100', '200'), ('200', 'nan')]:
            (self.root/f'{patient}_AV.wav').touch()
            (self.root/f'{patient}.txt').write_text(
                f'{patient} 1 4000\nAV x.hea {patient}_AV.wav\n#Additional ID: {other}\n')
        rows = circor_rows(self.root)
        self.assertEqual({r['patient_id'] for r in rows}, {'100'})
        self.assertEqual({r['metadata']['subject_id'] for r in rows}, {'100', '200'})

    def test_path_traversal_rejected(self):
        self.bmd_fixture()
        path = self.root/'train.csv'
        path.write_text(path.read_text().replace(',a,b', ',../../outside,b'))
        with self.assertRaises(ValueError):
            bmdhs_rows(self.root)


if __name__ == '__main__':
    unittest.main()
