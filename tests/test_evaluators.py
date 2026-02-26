import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Adjust this import to match the actual name/path of your module!
from censored_regressors.metrics.evaluators import (
    _get_classification_labels, MAE, MAE_c, Accuracy, Precision,
    Recall, Jaccard, evaluate_observed, evaluate_latent
)


class TestEvaluators(unittest.TestCase):

    def setUp(self):
        # Setup dummy data
        self.limits = np.array([2.0, 5.0, 0.0])
        self.latent = np.array([2.0, 7.0, -2.0])
        self.censoring = np.array([0, 1, -1])  # Obs, Right, Left

        # Perfect predictions for the classification boundaries
        self.pred_good = np.array([2.0, 6.0, -1.0])
        # Poor predictions (fails to predict the censored regions)
        self.pred_bad = np.array([2.0, 4.0, 1.0])

        self.X = np.array([[1], [2], [3]])
        self.gp_lower = np.array([1.0, 5.0, -3.0])
        self.gp_upper = np.array([3.0, 8.0, 1.0])
        self.model = MagicMock()

    # ==========================================
    # --- HELPER TESTS ---
    # ==========================================

    def test_get_classification_labels(self):
        """Test the internal helper correctly maps continuous predictions to binary flags."""

        # Test 1: Good predictions (model correctly guesses >5.0 and <0.0)
        y_true_class, y_pred_class = _get_classification_labels(self.limits, self.pred_good, self.censoring)
        np.testing.assert_array_equal(y_true_class, [0, 1, 1])
        np.testing.assert_array_equal(y_pred_class, [0, 1, 1])

        # Test 2: Bad predictions (model guesses 4.0 which is NOT >5.0)
        y_true_class, y_pred_class = _get_classification_labels(self.limits, self.pred_bad, self.censoring)
        np.testing.assert_array_equal(y_true_class, [0, 1, 1])
        np.testing.assert_array_equal(y_pred_class, [0, 0, 0])

    # ==========================================
    # --- METRIC TESTS ---
    # ==========================================

    def test_metaclass_callable(self):
        """Verify the metaclass allows calling classes directly like functions."""
        result = MAE(self.latent, self.pred_good)
        self.assertIsInstance(result, float)

    def test_mae_c(self):
        """Test MAE_c calculates error ONLY on censored points, and handles empty masks."""

        # 1. Normal behavior: Error should only be calculated on indices 1 and 2
        # Latent: [7.0, -2.0], Pred: [6.0, -1.0] -> Errors: [1.0, 1.0] -> MAE_c: 1.0
        mae_c_val = MAE_c(self.latent, self.pred_good, self.censoring)
        self.assertEqual(mae_c_val, 1.0)

        # 2. Edge Case behavior: No censored points in the batch
        all_uncensored = np.array([0, 0, 0])
        mae_c_nan = MAE_c(self.latent, self.pred_good, all_uncensored)
        self.assertTrue(np.isnan(mae_c_nan))

    def test_classification_metrics(self):
        """Test that wrapped sklearn metrics execute correctly."""

        # Perfect predictions -> 1.0 scores
        self.assertEqual(Accuracy(self.limits, self.pred_good, self.censoring), 1.0)
        self.assertEqual(Precision(self.limits, self.pred_good, self.censoring), 1.0)

        # Bad predictions -> 0 correct in censored classes
        self.assertAlmostEqual(Accuracy(self.limits, self.pred_bad, self.censoring), 0.3333333, places=5)
        self.assertEqual(Recall(self.limits, self.pred_bad, self.censoring), 0.0)

    # ==========================================
    # --- SUMMARY FUNCTION TESTS ---
    # ==========================================

    # We patch calc_nlpd and calc_latent_nlpd to avoid needing a real PyTorch/GPy model in these tests
    @patch('censored_regressors.metrics.evaluators.calc_nlpd', return_value=1.5)
    def test_evaluate_observed(self, mock_calc_nlpd):
        """Test that the evaluate_observed dictionary populates with the correct keys."""
        res = evaluate_observed(
            self.X, self.limits, self.pred_good,
            censoring=self.censoring, model=self.model
        )

        expected_keys = {
            'NLPD', 'RMSE', 'MAE', 'MAE_c', 'Hinge_MAE',
            'Accuracy', 'Precision', 'Recall', 'Jaccard'
        }
        self.assertEqual(set(res.keys()), expected_keys)
        self.assertEqual(res['NLPD'], 1.5)

        mock_calc_nlpd.assert_called_once()

    @patch('censored_regressors.metrics.evaluators.calc_nlpd', return_value=1.5)
    @patch('censored_regressors.metrics.evaluators.calc_latent_nlpd', return_value=0.5)
    def test_evaluate_latent(self, mock_calc_latent_nlpd, mock_calc_nlpd):
        """Test that the evaluate_latent dictionary populates with the extended keys."""
        res = evaluate_latent(
            self.X, self.limits, self.latent, self.pred_good,
            self.gp_lower, self.gp_upper, self.censoring, self.model
        )

        expected_keys = {
            'NLPD_obs', 'NLPD_latent', 'RMSE', 'MAE', 'MAE_c', 'Hinge_MAE',
            'Coverage', 'Accuracy', 'Precision', 'Recall', 'Jaccard'
        }
        self.assertEqual(set(res.keys()), expected_keys)
        self.assertEqual(res['NLPD_obs'], 1.5)
        self.assertEqual(res['NLPD_latent'], 0.5)

        mock_calc_nlpd.assert_called_once()
        mock_calc_latent_nlpd.assert_called_once()


if __name__ == '__main__':
    unittest.main()