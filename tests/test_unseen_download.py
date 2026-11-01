import hashlib
from pathlib import Path
import tempfile
import unittest

from cardio.data.download_unseen import verify


class DownloadIntegrityTests(unittest.TestCase):
    def test_published_hashes_accept_valid_and_reject_corrupt_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'audio.wav').write_bytes(b'fixture audio payload')
            digest = hashlib.sha256(b'fixture audio payload').hexdigest()
            (root / 'SHA256SUMS.txt').write_text(f'{digest} audio.wav\n')
            self.assertEqual(verify(root), 1)
            (root / 'audio.wav').write_bytes(b'changed payload')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                verify(root)

    def test_checksum_cannot_reference_files_outside_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'SHA256SUMS.txt').write_text('0'*64 + ' ../outside.wav\n')
            with self.assertRaisesRegex(ValueError, 'Unsafe checksum path'):
                verify(root)
