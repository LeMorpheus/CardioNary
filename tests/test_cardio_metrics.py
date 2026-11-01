import unittest
import numpy as np
from cardio.eval.metrics import (
    classification_metrics, expected_calibration_error,
    sensitivity_at_specificity, patient_bootstrap,
)


class MetricsTests(unittest.TestCase):
    def test_known_binary_results(self):
        m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
        self.assertEqual(m['accuracy'], 0.5)
        self.assertEqual(m['macro_f1'], 0.5)
        self.assertEqual(m['recall_0'], 0.5)
        self.assertEqual(m['recall_1'], 0.5)
        self.assertEqual(m['auroc'], 0.75)
        self.assertEqual(m['sensitivity_at_90_specificity'], 0.5)
        self.assertAlmostEqual(m['ece'], 0.35)

    def test_multiclass_perfect(self):
        m = classification_metrics([0, 1, 2], np.eye(3))
        for key in ('accuracy', 'macro_f1', 'auroc', 'recall_0', 'recall_1', 'recall_2', 'sensitivity_at_90_specificity'):
            self.assertEqual(m[key], 1)
        self.assertEqual(m['ece'], 0)

    def test_ece_edges_and_weighting(self):
        self.assertEqual(expected_calibration_error([0, 1], [0, 1]), 0)
        self.assertEqual(expected_calibration_error([1, 0], [0, 1]), 1)
        self.assertAlmostEqual(expected_calibration_error([0, 0, 1, 1], [0.25]*4), 0.25)

    def test_sensitivity_ties_do_not_interpolate(self):
        self.assertEqual(sensitivity_at_specificity([0, 1], [0.5, 0.5]), 0)
        self.assertEqual(sensitivity_at_specificity([0, 1], [0.5, 0.5], 0), 1)

    def test_exact_specificity_threshold(self):
        # One of ten negatives can be accepted at specificity 0.9.
        self.assertEqual(sensitivity_at_specificity([0]*10 + [1], [0.95]+[0.1]*9+[0.9]), 1)

    def test_missing_class(self):
        m = classification_metrics([0, 0], [0.1, 0.2])
        self.assertTrue(np.isnan(m['auroc']))
        self.assertTrue(np.isnan(m['sensitivity_at_90_specificity']))
        self.assertEqual(m['recall_1'], 0)
        self.assertEqual(m['macro_f1'], 0.5)

    def test_invalid_inputs(self):
        cases = [([], []), ([0], [np.nan]), ([0], [1.1]), ([2], [0.5]),
                 ([0.0], [0.5]), ([0, 1], [0.5]), ([0], [[0.2, 0.2]])]
        for y, p in cases:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                classification_metrics(y, p)

    def test_bootstrap_known_interval_and_determinism(self):
        def accuracy(y, p):
            return np.mean(y == (p[:, 1] >= 0.5))
        a = patient_bootstrap([0, 1], [0.1, 0.1], ['a', 'b'], accuracy)
        b = patient_bootstrap([0, 1], [0.1, 0.1], ['a', 'b'], accuracy)
        self.assertEqual(a, b)
        self.assertEqual((a.estimate, a.lower, a.upper), (0.5, 0, 1))
        self.assertEqual((a.valid_resamples, a.total_resamples), (1000, 1000))

    def test_resamples_keep_patients_together_with_multiplicity(self):
        seen = []
        def metric(y, p):
            seen.append(y.copy())
            return y.mean()
        patient_bootstrap([0, 0, 1], [0.1, 0.2, 0.8], ['a', 'a', 'b'], metric, n_resamples=100)
        for sample in seen[:-1]:
            self.assertEqual(np.sum(sample == 0) % 2, 0)
            self.assertIn(len(sample), [2, 3, 4])
        self.assertTrue(any(len(sample) == 4 for sample in seen))

    def test_reports_undefined_bootstrap_draws(self):
        def auc(y, p):
            return classification_metrics(y, p)['auroc']
        interval = patient_bootstrap([0, 1], [0.1, 0.9], ['a', 'b'], auc, n_resamples=40)
        self.assertLess(interval.valid_resamples, 40)
        self.assertGreater(interval.valid_resamples, 0)
        self.assertEqual((interval.lower, interval.upper), (1, 1))

    def test_invalid_bootstrap(self):
        for ids in (['a'], ['a', None], ['a', '']):
            with self.assertRaises(ValueError):
                patient_bootstrap([0, 1], [0.2, 0.8], ids, lambda y, p: 1)


if __name__ == '__main__':
    unittest.main()
