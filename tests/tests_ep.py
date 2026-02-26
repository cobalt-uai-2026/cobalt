import unittest
import numpy as np
import GPy

# --- Robust Import Logic ---
try:
    from censored_regressors.latent_inference.ep_gpy import (
        EP,
        EPCensored,
        cavityParams,
        gaussianApproximation,
        posteriorParams,
        marginalMoments
    )
except ImportError:
    import sys
    import os

    # Attempt to add src to path if running locally
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.censored_regressors.latent_inference.ep_gpy import (
        EP,
        EPCensored,
        cavityParams,
        gaussianApproximation,
        posteriorParams,
        marginalMoments
    )


class MockLikelihood:
    """
    A mock likelihood to test EPCensored without needing
    complex censored likelihood logic.
    Behaves like a standard Gaussian for moments.
    """

    def moments_match_ep(self, Y, tau, v, censoring, Y_metadata_i=None):
        # Return dummy moments that ensure stability
        # Z_hat, mu_hat, sigma2_hat
        # Z=1.0 (logZ=0), mu=0.0, sigma=1.0
        return 1.0, 0.0, 1.0

    def ep_gradients(self, Y, tau, v, dL_dK, Y_metadata=None, quad_mode='gh'):
        # Return dummy gradient (1D array)
        return np.array([0.1])


class TestEPHelpers(unittest.TestCase):
    """Tests for the helper arithmetic classes."""

    def setUp(self):
        self.num_data = 5
        self.tau = np.ones(self.num_data)
        self.v = np.zeros(self.num_data)

    def test_cavity_params_update(self):
        """Test the computation of cavity parameters (tau_cav, v_cav)."""
        cp = cavityParams(self.num_data)

        # Mock posterior params with necessary attributes
        class MockPostParams:
            def __init__(self):
                self.Sigma_diag = np.array([0.5] * 5)
                self.mu = np.array([1.0] * 5)

        # Mock Gaussian Approximation
        ga = gaussianApproximation(self.v, self.tau)  # tau=1, v=0

        eta = 1.0
        i = 0

        # Expected Math:
        # tau_cav = 1/Sigma_diag - eta * tau_site
        # tau_cav = 1/0.5 - 1.0 * 1.0 = 2 - 1 = 1.0
        # v_cav = mu/Sigma_diag - eta * v_site
        # v_cav = 1.0/0.5 - 1.0 * 0 = 2.0

        cp._update_i(eta, ga, MockPostParams(), i)

        self.assertAlmostEqual(cp.tau[i], 1.0)
        self.assertAlmostEqual(cp.v[i], 2.0)

    def test_gaussian_approximation_update(self):
        """Test the site parameter updates with damping."""
        ga = gaussianApproximation(self.v.copy(), self.tau.copy())

        marg_moments = marginalMoments(self.num_data)
        marg_moments.sigma2_hat[:] = 0.8
        marg_moments.mu_hat[:] = 0.5

        class MockPostParams:
            Sigma_diag = np.ones(5)  # 1.0
            mu = np.zeros(5)  # 0.0

        eta = 1.0
        delta = 0.5  # 50% damping
        i = 0

        # Calculate expected target
        # target_tau = 1/sigma2_hat - 1/Sigma_diag = 1/0.8 - 1/1 = 1.25 - 1 = 0.25
        # delta_tau = (delta/eta) * target_tau = 0.5 * 0.25 = 0.125
        # new_tau = old_tau + delta_tau = 1.0 + 0.125 = 1.125

        delta_tau, delta_v = ga._update_i(eta, delta, MockPostParams(), marg_moments, i)

        self.assertAlmostEqual(ga.tau[i], 1.125)


class TestEPInference(unittest.TestCase):
    """Tests for the main Inference classes."""

    def setUp(self):
        np.random.seed(42)
        self.N = 20
        self.X = np.linspace(0, 10, self.N)[:, None]
        # Synthetic Linear Data
        self.Y = np.sin(self.X) + np.random.randn(self.N, 1) * 0.1
        self.kernel = GPy.kern.RBF(input_dim=1, lengthscale=1.5, variance=1.0)

    def test_ep_gaussian_equivalence(self):
        """
        Critical Test: EP with a Gaussian Likelihood should yield
        results almost identical to Exact Gaussian Inference.
        """
        #

        likelihood = GPy.likelihoods.Gaussian(variance=0.1)

        # 1. Run Standard Exact Inference
        exact_inf = GPy.inference.latent_function_inference.ExactGaussianInference()
        post_exact, _, _ = exact_inf.inference(self.kernel, self.X, likelihood, self.Y)

        # 2. Run EP Inference
        # Note: EP should converge to exact Gaussian when likelihood is Gaussian
        ep_inf = EP(max_iters=10, delta=0.5)

        post_ep, log_marginal, grad_dict = ep_inf.inference(self.kernel, self.X, likelihood, self.Y)

        # 3. Compare Posterior Mean and Covariance
        np.testing.assert_allclose(
            post_exact.mean, post_ep.mean,
            atol=1e-3, err_msg="EP Mean diverges from Exact Gaussian Inference"
        )

        # Note: Covariance comparison might need slightly looser tolerance depending on EP convergence
        np.testing.assert_allclose(
            post_exact.covariance, post_ep.covariance,
            atol=1e-3, err_msg="EP Covariance diverges from Exact Gaussian Inference"
        )

    def test_ep_censored_structure(self):
        """
        Test that EPCensored runs and handles the extra 'censoring' argument.
        """
        ep_censored = EPCensored(max_iters=5)
        mock_lik = MockLikelihood()

        censoring = np.zeros(self.N)

        try:
            # FIX: inference returns (posterior, log_marginal, grad_dict)
            post, log_marginal, grad_dict = ep_censored.inference(
                self.kernel, self.X, mock_lik, self.Y, censoring
            )

            # Check if internal approximation states are accessible
            if ep_censored._ep_approximation is not None:
                # Unpack the tuple to verify structure
                post_params, ga_approx, cav_params, log_Z = ep_censored._ep_approximation
                self.assertIsInstance(ga_approx, gaussianApproximation)
                self.assertIsInstance(post_params, posteriorParams)
                self.assertIsInstance(cav_params, cavityParams)

        except Exception as e:
            self.fail(f"EPCensored inference crashed with error: {e}")

        # Check output types
        self.assertIsInstance(post.mean, np.ndarray)
        self.assertEqual(post.mean.shape, (self.N, 1))

    def test_serialization(self):
        """Test to_dict and from_dict reconstruction."""
        ep = EP(epsilon=1e-4, delta=0.8)

        # Simulate a modified state
        ep.epsilon = 1e-3

        # Serialize
        data_dict = ep.to_dict()

        # Check the class key
        self.assertEqual(data_dict['class'], "GPy.inference.latent_function_inference.expectation_propagation.EP")

        # Reconstruct
        # FIX: Pass data_dict.copy() so the original dict isn't mutated by .pop()
        ep_new = EP._build_from_input_dict(EP, data_dict.copy())

        self.assertEqual(ep_new.epsilon, 1e-3)
        self.assertEqual(ep_new.delta, 0.8)
        self.assertIsInstance(ep_new, EP)


if __name__ == "__main__":
    unittest.main()