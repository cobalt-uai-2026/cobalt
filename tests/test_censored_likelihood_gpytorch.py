import unittest
import torch
import math
import numpy as np
import sys
import os
from scipy import integrate
from scipy.stats import norm
from torch.distributions import MultivariateNormal
from unittest.mock import patch, MagicMock

# --- IMPORTS ---
try:
    from censored_regressors.likelihoods.censored_likelihood_gpytorch import CensoredGaussianLikelihoodAnalytic
    from censored_regressors.distributions.censored_normal import CensoredNormal
except ImportError:
    print("Package import failed. Attempting local import...")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.censored_regressors.likelihoods.censored_likelihood_gpytorch import CensoredGaussianLikelihoodAnalytic
    from src.censored_regressors.distributions.censored_normal import CensoredNormal


class TestCensoredLikelihood(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tolerance = 1e-4

    # --- HELPER: SCIPY GROUND TRUTH ---
    def scipy_reference_integration(self, a, b):
        """Computes E_{z~N(0,1)} [ log Phi(a + b*z) ] using SciPy."""

        def integrand(z):
            pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            arg = a + b * z
            log_phi = norm.logcdf(arg)
            return log_phi * pdf

        val, error = integrate.quad(integrand, -20, 20)
        return val

    # --- CRITICAL BUG REPRODUCTION TESTS ---

    def test_censored_loss_is_nonzero_and_gradients_flow(self):
        """
        CRITICAL TEST:
        If we have a Right Censored point at y=1.0, and our model predicts mean=-5.0,
        it is VERY unlikely that the data is > 1.0.
        Therefore, the Log-Likelihood should be a large NEGATIVE number (high loss).

        If this returns 0.0, your masking or integration logic is broken.
        """
        # Right Censored at 1.0
        high_bound = 1.0
        model = CensoredGaussianLikelihoodAnalytic(variance=1.0, high=high_bound)

        # 1. Create a "Bad" Model Prediction
        # The latent function f is at -5.0.
        # The probability that y > 1.0 given f=-5.0 is tiny.
        # So log P(y > 1.0 | f=-5.0) should be large negative.
        mean = torch.tensor([-5.0], requires_grad=True)
        cov = torch.tensor([[1.0]])
        post = MultivariateNormal(mean, cov)

        # Target must be >= high_bound to trigger censoring logic in most implementations
        # or exactly high_bound depending on how you generate labels.
        # Let's assume standard survival analysis: observed y is the bound.
        target = torch.tensor([high_bound])

        # 2. Compute Loss
        log_prob = model.expected_log_prob(target, post)

        print(f"\n[Loss Test] Mean=-5.0, Bound=1.0. LogProb: {log_prob.item()}")

        # 3. Assertions
        self.assertNotEqual(log_prob.item(), 0.0,
                            "Log Probability is 0.0! The censored term is being ignored/masked out.")
        self.assertLess(log_prob.item(), -1.0, "Log Probability should be negative for poor predictions.")

        # 4. Check Gradients
        loss = -log_prob.sum()
        loss.backward()
        self.assertIsNotNone(mean.grad)
        self.assertNotEqual(mean.grad.item(), 0.0, "Gradient is 0.0! The model won't learn.")
        print(f"[Gradient Test] dLoss/dMean: {mean.grad.item()}")

    def test_quadrature_atoms_usage(self):
        """
        Verifies that the integration actually uses the quadrature 'atoms' (nodes).
        We patch the internal math function to see what inputs it receives.
        """
        n_points = 10
        model = CensoredGaussianLikelihoodAnalytic(
            integration_type='gauss_hermite',
            n_points=n_points,
            high=1.0
        )

        mean = torch.tensor([0.0])
        cov = torch.tensor([[1.0]])
        post = MultivariateNormal(mean, cov)
        target = torch.tensor([1.0])  # Right censored

        # Patch the robust log phi function which is called inside the integral
        with patch.object(model, '_log_phi_robust', side_effect=model._log_phi_robust) as mock_math:
            _ = model.expected_log_prob(target, post)

            # Check calls
            self.assertTrue(mock_math.called)

            # Inspect arguments of the first call
            args, _ = mock_math.call_args
            input_tensor = args[0]  # The 'z' or 'arg' tensor passed to log_phi

            print(f"\n[Atoms Test] Integration Input Shape: {input_tensor.shape}")

            # The input shape should reflect [Batch_Size, n_points]
            # Batch size is 1, so we expect [1, 10]
            self.assertEqual(input_tensor.shape[-1], n_points,
                             f"Expected {n_points} quadrature atoms, but got {input_tensor.shape[-1]}")

    def test_masking_correctness(self):
        """
        Ensure that the model correctly identifies Left, Right, and Observed data.
        """
        low = -1.0
        high = 1.0
        model = CensoredGaussianLikelihoodAnalytic(variance=0.1, low=low, high=high)

        # Batch of 3: [Left Censored, Observed, Right Censored]
        targets = torch.tensor([-1.0, 0.0, 1.0])
        mean = torch.zeros(3)
        cov = torch.eye(3)
        post = MultivariateNormal(mean, cov)

        # We spy on _expected_log_prob_terms to check internal masks
        with patch.object(model, '_integrate_log_phi', side_effect=model._integrate_log_phi) as mock_int:
            _ = model.expected_log_prob(targets, post)

            # If masking works, _integrate_log_phi should be called.
            # However, since pytorch operations are vectorized, it might compute integrals
            # for all and then mask results.
            # Instead, let's check the output values.

            terms = model._expected_log_prob_terms(targets, post)

            # Check Normal Part: Should be non-zero ONLY for the middle element (index 1)
            normal_part = terms['normal_part']
            self.assertEqual(normal_part[0].item(), 0.0)
            self.assertNotEqual(normal_part[1].item(), 0.0)
            self.assertEqual(normal_part[2].item(), 0.0)

            # Check Upper Censored Part: Should be non-zero ONLY for last element (index 2)
            upper = terms['upper_censored_part']
            self.assertEqual(upper[0].item(), 0.0)
            self.assertEqual(upper[1].item(), 0.0)
            self.assertNotEqual(upper[2].item(), 0.0)

    # --- STANDARD NUMERICAL TESTS ---

    def test_numerical_accuracy_censored(self):
        """Compare PyTorch GH integration against SciPy Quad."""
        high_bound = 1.0
        model = CensoredGaussianLikelihoodAnalytic(variance=1.0, high=high_bound,
                                                   integration_type='gauss_hermite', n_points=40)
        mu_val = 1.5
        std_val = 0.5
        target = torch.tensor([2.0])  # > high
        post = MultivariateNormal(torch.tensor([mu_val]), torch.tensor([[std_val ** 2]]))

        with torch.no_grad():
            terms = model._expected_log_prob_terms(target, post)
            py_val = terms['upper_censored_part'].item()

        a = (mu_val - high_bound) / 1.0
        b = std_val / 1.0
        scipy_val = self.scipy_reference_integration(a, b)

        self.assertAlmostEqual(py_val, scipy_val, delta=self.tolerance)

    def test_extreme_tails_robustness(self):
        """Test asymptotic stability."""
        model = CensoredGaussianLikelihoodAnalytic(high=0.0)
        a = torch.tensor([-30.0], dtype=torch.float64)  # Far from bound
        b = torch.tensor([1.0], dtype=torch.float64)
        res = model._integrate_log_phi(a, b)

        self.assertFalse(torch.isnan(res).any())
        # log Phi(-30) approx -30^2/2 = -450
        self.assertTrue(res.item() < -400.0)


if __name__ == '__main__':
    unittest.main()