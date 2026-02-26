import unittest
import numpy as np
import torch
import gpytorch
from unittest.mock import MagicMock, patch

# Import the metrics to test
from censored_regressors.metrics.metrics import calc_nlpd, hinge_mae, calc_latent_nlpd


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1.0], [2.0], [3.0]])
        self.y_true = np.array([0.5, 1.5, 2.5])
        self.censoring = np.array([0, 1, -1])  # Obs, Right, Left

    # ==========================
    # --- OBSERVED NLPD TESTS ---
    # ==========================

    def test_nlpd_gpytorch_censored(self):
        """
        Test NLPD calculation for the custom censored likelihood path.
        Checks if it uses Monte Carlo integration (LogSumExp).
        """
        # 1. Mock the GPyTorch Model
        model = MagicMock(spec=torch.nn.Module)
        model.model = model

        dummy_param = torch.nn.Parameter(torch.empty(0))
        model.parameters.return_value = iter([dummy_param])

        del model.log_predictive_density

        # 2. Mock the Likelihood
        likelihood = MagicMock()
        likelihood.log_prob_density = MagicMock()

        # 3. Setup Mock Returns
        q_f = MagicMock()
        model.return_value = q_f

        f_samples = torch.zeros(1000, 3)
        q_f.sample.return_value = f_samples

        # Return constant -1.0 for all samples
        likelihood.log_prob_density.return_value = torch.ones(1000, 3) * -1.0

        # 4. Run Calc
        nlpd = calc_nlpd(model, self.X, self.y_true, likelihood=likelihood)

        self.assertAlmostEqual(nlpd, 1.0, places=5)
        likelihood.log_prob_density.assert_called_once()

    def test_nlpd_gpytorch_standard(self):
        """
        Test fallback to standard GPyTorch metrics for standard likelihoods.
        """
        model = MagicMock(spec=torch.nn.Module)
        dummy_param = torch.nn.Parameter(torch.empty(0))
        model.parameters.return_value = iter([dummy_param])
        del model.log_predictive_density

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        q_f = gpytorch.distributions.MultivariateNormal(
            torch.zeros(3), torch.eye(3)
        )
        model.side_effect = lambda x: q_f

        nlpd = calc_nlpd(model, self.X, self.y_true, likelihood=likelihood)

        self.assertFalse(np.isnan(nlpd))
        self.assertTrue(isinstance(nlpd, float))

    @patch('censored_regressors.metrics.metrics.gpytorch.metrics.negative_log_predictive_density')
    def test_nlpd_gpytorch_clamping(self, mock_nlpd_metric):
        """
        Test that out-of-bounds y_true values are correctly clamped to the distribution's support bounds.
        """
        model = MagicMock(spec=torch.nn.Module)
        dummy_param = torch.nn.Parameter(torch.empty(0))
        model.parameters.return_value = iter([dummy_param])
        del model.log_predictive_density

        likelihood = MagicMock()

        del likelihood.log_prob_density

        # Mock a predictive distribution with strict bounds [0.0, 1.0]
        pred_dist = MagicMock()
        pred_dist.support.lower_bound = torch.tensor(0.0)
        pred_dist.support.upper_bound = torch.tensor(1.0)
        likelihood.return_value = pred_dist

        # Mock the final metric calculation to prevent actual GPyTorch execution
        mock_nlpd_metric.return_value = torch.tensor([1.5])

        # Run calc (y_true = [0.5, 1.5, 2.5])
        calc_nlpd(model, self.X, self.y_true, likelihood=likelihood)

        # Check what targets were actually passed to the NLPD calculator
        args, _ = mock_nlpd_metric.call_args
        clamped_y_passed = args[1]

        # The inputs 1.5 and 2.5 should be clamped down to the upper bound of 1.0
        expected_clamped = torch.tensor([0.5, 1.0, 1.0])
        torch.testing.assert_close(clamped_y_passed, expected_clamped)

    # ==========================
    # --- LATENT NLPD TESTS ---
    # ==========================

    def test_latent_nlpd_gpy(self):
        """
        Test that Latent NLPD on a GPy model forces the censoring metadata to all zeros.
        """
        model = MagicMock()
        # Mock the GPy lpd function to return dummy log probabilities
        model.model.log_predictive_density = MagicMock(return_value=np.array([-1.0, -2.0, -3.0]))

        # Run Latent Calc
        nlpd = calc_latent_nlpd(model, self.X, self.y_true)

        # Verify it calculated the mean of [-1.0, -2.0, -3.0] correctly (which is 2.0 after negating)
        self.assertEqual(nlpd, 2.0)

        # Verify the Y_metadata passed was an array of zeros
        call_args = model.model.log_predictive_density.call_args
        kwargs = call_args[1]
        self.assertIn('Y_metadata', kwargs)
        self.assertIn('censoring', kwargs['Y_metadata'])
        np.testing.assert_array_equal(
            kwargs['Y_metadata']['censoring'],
            np.zeros((3, 1))
        )

    @patch('censored_regressors.metrics.metrics.gpytorch.likelihoods.GaussianLikelihood')
    @patch('censored_regressors.metrics.metrics.gpytorch.metrics.negative_log_predictive_density')
    def test_latent_nlpd_gpytorch(self, mock_nlpd_metric, MockGaussianLikelihood):
        """
        Test that Latent NLPD on a GPyTorch model correctly extracts the custom noise parameter
        and substitutes it into a standard Gaussian Likelihood.
        """
        model = MagicMock(spec=torch.nn.Module)
        dummy_param = torch.nn.Parameter(torch.empty(0))
        model.parameters.return_value = iter([dummy_param])
        del model.log_predictive_density

        # Mock custom censored likelihood with a trained noise of 0.42
        custom_lik = MagicMock()
        custom_lik.noise = torch.tensor([0.42])

        # Mock the internal substitution GaussianLikelihood
        mock_standard_lik_instance = MagicMock()
        MockGaussianLikelihood.return_value.to.return_value = mock_standard_lik_instance

        mock_nlpd_metric.return_value = torch.tensor([3.14])

        # Run Latent Calc
        res = calc_latent_nlpd(model, self.X, self.y_true, likelihood=custom_lik)

        # Verify the custom noise parameter was injected into the standard likelihood
        self.assertAlmostEqual(mock_standard_lik_instance.noise.item(), 0.42, places=5)

        # Verify the final value was extracted
        self.assertAlmostEqual(res, -3.14, places=5)

    # ==========================
    # --- HINGE MAE TESTS ---
    # ==========================

    def test_hinge_mae_uncensored(self):
        err = hinge_mae([1.0], [1.0], [0])
        self.assertEqual(err, 0.0)
        err = hinge_mae([1.0], [0.5], [0])
        self.assertEqual(err, 0.5)

    def test_hinge_mae_right_censored(self):
        err = hinge_mae([0.5], [0.8], [1])
        self.assertEqual(err, 0.0)
        err = hinge_mae([0.5], [0.2], [1])
        self.assertAlmostEqual(err, 0.3)

    def test_hinge_mae_left_censored(self):
        err = hinge_mae([0.5], [0.2], [-1])
        self.assertEqual(err, 0.0)
        err = hinge_mae([0.5], [0.8], [-1])
        self.assertAlmostEqual(err, 0.3)


if __name__ == '__main__':
    unittest.main()