import unittest
import numpy as np
from scipy.stats import norm
import GPy
from GPy.likelihoods import link_functions
from censored_regressors.likelihoods.censored_likelihood_gpy import CensoredGaussian

class TestCensoredGaussian(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Initialize likelihood with variance=1.0
        self.variance = 1.5
        self.likelihood = CensoredGaussian(variance=self.variance)

        # Test Data
        self.N = 10
        self.f = np.random.randn(self.N, 1)  # Latent function values
        self.y = np.random.randn(self.N, 1)  # Observed values / thresholds

    def test_uncensored_logpdf_matches_scipy(self):
        """
        Case c=0: Should match standard Gaussian PDF exactly.
        """
        # Metadata for uncensored
        meta = {'censoring': np.zeros(self.N)}

        # 1. GPy Implementation
        gpy_res = self.likelihood.logpdf_link(self.f, self.y, Y_metadata=meta)

        # 2. Scipy Ground Truth
        # log N(y | f, sigma^2)
        scale = np.sqrt(self.variance)
        scipy_res = norm.logpdf(self.y, loc=self.f, scale=scale)

        np.testing.assert_allclose(gpy_res, scipy_res, atol=1e-6,
                                   err_msg="Uncensored logpdf mismatch")

    def test_right_censored_logpdf_matches_scipy(self):
        """
        Case c=1 (Right Censored):
        Observation y is a lower bound. True Y > y.
        Log Likelihood = log(1 - CDF((y-f)/sigma)) = log(CDF((f-y)/sigma))
        """
        meta = {'censoring': np.ones(self.N)}

        gpy_res = self.likelihood.logpdf_link(self.f, self.y, Y_metadata=meta)

        scale = np.sqrt(self.variance)
        # P(Y > y) = SF(y) = 1 - CDF(y)
        scipy_res = norm.logsf(self.y, loc=self.f, scale=scale)

        np.testing.assert_allclose(gpy_res, scipy_res, atol=1e-6,
                                   err_msg="Right censored logpdf mismatch")

    def test_left_censored_logpdf_matches_scipy(self):
        """
        Case c=-1 (Left Censored):
        Observation y is an upper bound. True Y < y.
        Log Likelihood = log(CDF((y-f)/sigma))
        """
        meta = {'censoring': np.full(self.N, -1)}

        gpy_res = self.likelihood.logpdf_link(self.f, self.y, Y_metadata=meta)

        scale = np.sqrt(self.variance)
        # P(Y < y) = CDF(y)
        scipy_res = norm.logcdf(self.y, loc=self.f, scale=scale)

        np.testing.assert_allclose(gpy_res, scipy_res, atol=1e-6,
                                   err_msg="Left censored logpdf mismatch")

    def test_vectorized_mixed_censoring(self):
        """
        Ensure the class handles a batch with mixed [0, 1, -1] correctly.
        """
        # Create mixed metadata
        c_mixed = np.array([0, 1, -1] * 4)[:self.N]
        meta = {'censoring': c_mixed}

        # GPy vectorized call
        gpy_res = self.likelihood.logpdf_link(self.f, self.y, Y_metadata=meta)

        # Manual element-wise check
        scale = np.sqrt(self.variance)
        expected = np.zeros_like(gpy_res)

        for i in range(self.N):
            c = c_mixed[i]
            if c == 0:
                expected[i] = norm.logpdf(self.y[i], loc=self.f[i], scale=scale)
            elif c == 1:
                expected[i] = norm.logsf(self.y[i], loc=self.f[i], scale=scale)
            elif c == -1:
                expected[i] = norm.logcdf(self.y[i], loc=self.f[i], scale=scale)

        np.testing.assert_allclose(gpy_res, expected, atol=1e-6,
                                   err_msg="Vectorized processing failed")

    def test_gradient_dlogpdf_dlink(self):
        """
        Check 1st Derivative w.r.t latent function (f) using Finite Difference.
        """
        c_mixed = np.array([0, 1, -1] * 4)[:self.N]
        meta = {'censoring': c_mixed}

        # Analytical Gradient
        analytical = self.likelihood.dlogpdf_dlink(self.f, self.y, Y_metadata=meta)

        # Finite Difference
        epsilon = 1e-5
        f_plus = self.f + epsilon
        f_minus = self.f - epsilon

        ll_plus = self.likelihood.logpdf_link(f_plus, self.y, Y_metadata=meta)
        ll_minus = self.likelihood.logpdf_link(f_minus, self.y, Y_metadata=meta)
        numerical = (ll_plus - ll_minus) / (2 * epsilon)

        np.testing.assert_allclose(analytical, numerical, atol=1e-5,
                                   err_msg="Gradient dlogpdf_dlink incorrect")

    def test_hessian_d2logpdf_dlink2(self):
        """
        Check 2nd Derivative w.r.t latent function (f).
        Important for Laplace Approximation.
        """
        c_mixed = np.array([0, 1, -1] * 4)[:self.N]
        meta = {'censoring': c_mixed}

        # Analytical Hessian
        analytical = self.likelihood.d2logpdf_dlink2(self.f, self.y, Y_metadata=meta)

        # Numerical: Finite difference of the 1st derivative
        epsilon = 1e-5

        # Need dlogpdf/dlink at f+eps and f-eps
        grad_plus = self.likelihood.dlogpdf_dlink(self.f + epsilon, self.y, Y_metadata=meta)
        grad_minus = self.likelihood.dlogpdf_dlink(self.f - epsilon, self.y, Y_metadata=meta)

        numerical = (grad_plus - grad_minus) / (2 * epsilon)

        np.testing.assert_allclose(analytical, numerical, atol=1e-5,
                                   err_msg="Hessian d2logpdf_dlink2 incorrect")

    def test_third_derivative_d3logpdf_dlink3(self):
        """
        Check 3rd Derivative. Important for some EP approximations.
        """
        c_mixed = np.array([1, -1] * 5)[:self.N]  # Skip 0 as 3rd deriv is 0
        meta = {'censoring': c_mixed}

        analytical = self.likelihood.d3logpdf_dlink3(self.f, self.y, Y_metadata=meta)

        epsilon = 1e-5
        hess_plus = self.likelihood.d2logpdf_dlink2(self.f + epsilon, self.y, Y_metadata=meta)
        hess_minus = self.likelihood.d2logpdf_dlink2(self.f - epsilon, self.y, Y_metadata=meta)

        numerical = (hess_plus - hess_minus) / (2 * epsilon)

        np.testing.assert_allclose(analytical, numerical, atol=1e-4,
                                   err_msg="3rd Derivative incorrect")

    def test_gradient_wrt_variance(self):
        """
        Check derivative of log likelihood w.r.t variance parameter.
        dlogpdf / dvar
        """
        c_mixed = np.array([0, 1, -1] * 4)[:self.N]
        meta = {'censoring': c_mixed}

        analytical = self.likelihood.dlogpdf_link_dvar(self.f, self.y, Y_metadata=meta)

        # Finite Difference by changing the object's variance
        epsilon = 1e-5
        original_var = self.likelihood.variance.values[0]

        # V + eps
        self.likelihood.variance = original_var + epsilon
        ll_plus = self.likelihood.logpdf_link(self.f, self.y, Y_metadata=meta)

        # V - eps
        self.likelihood.variance = original_var - epsilon
        ll_minus = self.likelihood.logpdf_link(self.f, self.y, Y_metadata=meta)

        # Reset
        self.likelihood.variance = original_var

        numerical = (ll_plus - ll_minus) / (2 * epsilon)

        np.testing.assert_allclose(analytical, numerical, atol=1e-5,
                                   err_msg="Gradient w.r.t Variance incorrect")

    def test_predictive_density(self):
        """
        Test log_predictive_density correctness.
        """
        mu_star = np.array([0.5, 0.5])
        var_star = np.array([0.1, 0.1])
        y_test = np.array([0.0, 1.0])
        # Case 1: Uncensored, Case 2: Right Censored
        meta = {'censoring': np.array([0, 1])}

        res = self.likelihood.log_predictive_density(y_test, mu_star, var_star, Y_metadata=meta)

        # Manual Check
        sigma2_tot = var_star + self.variance

        # 1. Uncensored Normal
        expected_0 = norm.logpdf(y_test[0], loc=mu_star[0], scale=np.sqrt(sigma2_tot[0]))

        # 2. Right Censored (P(Y > 1.0))
        # log(1 - CDF((y - mu)/sigma))
        expected_1 = norm.logsf(y_test[1], loc=mu_star[1], scale=np.sqrt(sigma2_tot[1]))

        np.testing.assert_allclose(res[0], expected_0)
        np.testing.assert_allclose(res[1], expected_1)

    def test_ep_moments_edge_cases(self):
        """
        Smoke test for EP moments matching to ensure no crashes on boundaries.
        """
        tau = 1.0
        v = 0.5

        # Test 0, 1, -1
        for c in [0, 1, -1]:
            Z, mu, sigma2 = self.likelihood.moments_match_ep(0.0, tau, v, censoring_i=c)
            self.assertTrue(np.isfinite(Z))
            self.assertTrue(np.isfinite(mu))
            self.assertTrue(np.isfinite(sigma2))
            self.assertGreater(sigma2, 0)


if __name__ == '__main__':
    unittest.main()