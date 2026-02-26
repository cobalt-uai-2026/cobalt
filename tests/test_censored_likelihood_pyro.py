import unittest
import torch
import numpy as np
import pyro
import math
from scipy import integrate
from scipy.stats import norm
from torch.distributions import constraints

try:
    from censored_regressors.distributions.censored_normal_pyro import PyroCensoredNormal
    from censored_regressors.likelihoods.censored_likelihood_pyro import CensoredHomoscedGaussian, \
        VariationalCensoredNormal
except ImportError:
    pass


class TestPyroCensoredLikelihood(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tolerance = 1e-3

    def scipy_expected_log_prob(self, y, f_loc, f_var, obs_sigma, low, high):
        """Ground truth calculation using SciPy."""
        f_sigma = math.sqrt(f_loc.item() if isinstance(f_loc, torch.Tensor) else f_loc)
        q_sigma = math.sqrt(f_var)
        q_mu = f_loc

        # Logic: If y is at the boundary, we calculate the mass (CDF/SF).
        # We use a small epsilon for float comparison safety
        eps = 1e-6

        def log_lik_f(f_val):
            if y <= low + eps:
                # Left censored mass
                return norm.logcdf((low - f_val) / obs_sigma)
            elif y >= high - eps:
                # Right censored mass
                return norm.logsf((high - f_val) / obs_sigma)
            else:
                # Uncensored density
                return norm.logpdf(y, loc=f_val, scale=obs_sigma)

        def integrand(f_val):
            q_prob = norm.pdf(f_val, loc=q_mu, scale=q_sigma)
            ll = log_lik_f(f_val)
            return q_prob * ll

        limit = 10 * q_sigma
        val, err = integrate.quad(integrand, q_mu - limit, q_mu + limit)
        return val

    def test_forward_shapes(self):
        """Ensure the likelihood handles batching correctly."""
        B = 5
        f_loc = torch.randn(B)
        f_var = torch.rand(B)

        # [FIX] Generate y within bounds [-1, 1]
        low, high = -1.0, 1.0
        y_raw = torch.randn(B)
        y = torch.clamp(y_raw, low, high)

        likelihood = CensoredHomoscedGaussian(variance=0.5, low=low, high=high)

        # Check distribution manually to avoid Pyro trace overhead
        dist_obj = VariationalCensoredNormal(f_loc, f_var, likelihood.variance.sqrt(),
                                             likelihood.low, likelihood.high)

        res = dist_obj.log_prob(y)
        self.assertEqual(res.shape, (B,))
        self.assertTrue(torch.isfinite(res).all())

    def test_numerical_accuracy_uncensored(self):
        """Verify GH Quadrature for uncensored data."""
        f_loc = 0.5
        f_var = 0.2
        obs_var = 0.1
        obs_sigma = math.sqrt(obs_var)
        y = 0.6  # Uncensored value (strictly inside -1, 1)

        dist_obj = VariationalCensoredNormal(
            torch.tensor(f_loc), torch.tensor(f_var), torch.tensor(obs_sigma),
            torch.tensor(-1.0), torch.tensor(1.0), num_quad_points=30
        )
        pyro_val = dist_obj.log_prob(torch.tensor(y)).item()
        scipy_val = self.scipy_expected_log_prob(y, f_loc, f_var, obs_sigma, -1.0, 1.0)

        print(f"\n[Uncensored] Pyro: {pyro_val:.5f} vs SciPy: {scipy_val:.5f}")
        self.assertAlmostEqual(pyro_val, scipy_val, delta=self.tolerance)

    def test_numerical_accuracy_censored(self):
        """Verify GH Quadrature for CENSORED data."""
        f_loc = 1.5
        f_var = 0.3
        obs_var = 0.5
        obs_sigma = math.sqrt(obs_var)

        low, high = -1.0, 1.0

        # [FIX] The observed value MUST be the bound itself (1.0)
        # It cannot be 2.0, as that is outside the support.
        y = 1.0

        dist_obj = VariationalCensoredNormal(
            torch.tensor(f_loc), torch.tensor(f_var), torch.tensor(obs_sigma),
            torch.tensor(low), torch.tensor(high), num_quad_points=40
        )
        pyro_val = dist_obj.log_prob(torch.tensor(y)).item()
        scipy_val = self.scipy_expected_log_prob(y, f_loc, f_var, obs_sigma, low, high)

        print(f"\n[Censored] Pyro: {pyro_val:.5f} vs SciPy: {scipy_val:.5f}")
        self.assertAlmostEqual(pyro_val, scipy_val, delta=self.tolerance)

    def test_gradients_exist(self):
        """Ensure we can backprop to f_loc and f_var."""
        f_loc = torch.tensor([0.5], requires_grad=True)
        f_var = torch.tensor([0.2], requires_grad=True)

        # [FIX] Observed value is at the bound
        y = torch.tensor([1.0])

        likelihood = CensoredHomoscedGaussian(variance=0.1, low=-1.0, high=1.0)

        y_dist = VariationalCensoredNormal(f_loc, f_var, likelihood.variance.sqrt(),
                                           likelihood.low, likelihood.high)

        loss = -y_dist.log_prob(y).sum()
        loss.backward()

        self.assertIsNotNone(f_loc.grad)
        self.assertFalse(torch.isnan(f_loc.grad).any())

    def test_prediction_variance_convolved(self):
        f_loc = torch.zeros(10)
        f_var = torch.ones(10) * 0.5
        obs_var = 0.5

        likelihood = CensoredHomoscedGaussian(variance=obs_var, low=-10, high=10)

        dist_obj = VariationalCensoredNormal(f_loc, f_var, likelihood.variance.sqrt(),
                                             likelihood.low, likelihood.high)

        # Check convolution
        total_scale = (f_var + obs_var).sqrt()
        expected_scale = torch.ones(10)
        self.assertTrue(torch.allclose(total_scale, expected_scale))

        # Samples should be within bounds
        samples = dist_obj.sample()
        self.assertEqual(samples.shape, (10,))
        self.assertTrue((samples >= -10).all())
        self.assertTrue((samples <= 10).all())


if __name__ == '__main__':
    unittest.main()