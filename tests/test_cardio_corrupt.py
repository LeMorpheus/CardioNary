import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from cardio.eval.corrupt import add_noise, low_pass, main


class CorruptTests(unittest.TestCase):
    def setUp(self):
        self.rate = 4000
        self.t = np.arange(8000) / self.rate
        self.x = 0.5 * np.sin(2*np.pi*100*self.t)

    def test_exact_snr_mono_and_stereo(self):
        for kind in ['white', 'pink']:
            for x in [self.x, np.column_stack([self.x, self.x/2])]:
                y = add_noise(x, 12, kind)
                measured = 10*np.log10(np.mean(x*x) / np.mean((y-x)**2))
                self.assertAlmostEqual(measured, 12, places=10)
                np.testing.assert_array_equal(y, add_noise(x, 12, kind))

    def test_pink_noise_has_more_low_frequency_power(self):
        noise = add_noise(self.x, 0, 'pink') - self.x
        spectrum = abs(np.fft.rfft(noise))**2
        freq = np.fft.rfftfreq(len(noise), 1/self.rate)
        self.assertGreater(spectrum[(freq > 20) & (freq < 200)].mean(),
                           3*spectrum[(freq > 1000) & (freq < 1500)].mean())

    def test_filter_passes_low_and_rejects_high(self):
        high = np.sin(2*np.pi*1200*self.t)
        y = low_pass(self.x + high, self.rate, 300)
        self.assertLess(np.sqrt(np.mean((y[200:-200]-self.x[200:-200])**2)), 0.01)
        self.assertEqual(low_pass(np.column_stack([self.x, high]), self.rate, 300).shape, (8000, 2))

    def test_invalid_audio_and_parameters(self):
        for x in [np.zeros(30), [np.nan, 0], [1]]:
            with self.assertRaises(ValueError):
                add_noise(x, 10)
        with self.assertRaises(ValueError):
            low_pass(self.x, self.rate, 2000)
        with self.assertRaises(ValueError):
            low_pass([0, 1], self.rate, 300)

    def test_wav_cli_preserves_rate_channels_and_input(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory)/'in.wav', Path(directory)/'out.wav'
            wavfile.write(src, self.rate, np.column_stack([self.x, self.x]).astype(np.float32))
            original = src.read_bytes()
            main([str(src), str(dst), '--noise', 'white', '--snr-db', '10', '--cutoff-hz', '300'])
            rate, samples = wavfile.read(dst)
            self.assertEqual(rate, self.rate)
            self.assertEqual(samples.shape, (8000, 2))
            self.assertEqual(src.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
