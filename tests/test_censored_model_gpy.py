import unittest
import numpy as np
import GPy
from unittest.mock import MagicMock, patch

# Adjust import based on your package structure
try:
    from censored_regressors.models.censored_model_gpy import GPCensoredRegression
    from censored_regressors.latent_inference.ep_gpy import EPCensored
    from censored_regressors.likelihoods.censored_likelihood_gpy import CensoredGaussian
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.censored_regressors.models.censored_model_gpy import GPCensoredRegression
    from src.censored_regressors.latent_inference.ep_gpy import EPCensored
    from src.censored_regressors.likelihoods.censored_likelihood_gpy import CensoredGaussian


class TestGPCensoredRegression(unittest.TestCase):

    def setUp(self):
        # Standard dummy data
        self.N = 10
        self.D = 1
        self.X = np.random.randn(self.N, self.D)
        self.Y = np.random.randn(self.N, 1)
        self.censoring = np.zeros(self.N)  # Simple censoring for init tests

        # USE REAL KERNEL: GPy needs '.size', '.link_parameters', etc.
        self.real_kern = GPy.kern.RBF(input_dim=self.D)

        # USE REAL LIKELIHOOD for basic init tests
        self.real_lik = CensoredGaussian(censoring=self.censoring)

    def test_initialization_defaults(self):
        """Test basic initialization and metadata packing."""
        model = GPCensoredRegression(self.X, self.Y, self.censoring, kernel=self.real_kern)

        # 1. Check Metadata Packing
        self.assertIn('censoring', model.Y_metadata)
        self.assertEqual(model.Y_metadata['censoring'].shape, (self.N, 1))

        # 2. Check internal 1D censoring storage
        self.assertEqual(model.censoring.ndim, 1)

        # 3. Check Default Likelihood Type
        self.assertIsInstance(model.likelihood, CensoredGaussian)

        # 4. Check Default Inference Type
        self.assertIsInstance(model.inference_method, EPCensored)

    def test_initialization_custom_likelihood(self):
        """Test initializing with a user-provided likelihood."""
        # Use a real GPy likelihood (standard Gaussian) to satisfy isinstance checks
        custom_lik = GPy.likelihoods.Gaussian()

        model = GPCensoredRegression(self.X, self.Y, self.censoring, likelihood=custom_lik, kernel=self.real_kern)

        # GPy often wraps likelihoods, so we check type or identity
        self.assertIsInstance(model.likelihood, GPy.likelihoods.Gaussian)

    def test_parameters_changed_ep_censored(self):
        """
        Verify EPCensored.inference is called with the explicit 'censoring' argument.
        """
        model = GPCensoredRegression(
            self.X, self.Y, self.censoring,
            kernel=self.real_kern
        )

        # Create a Mock for the inference method
        mock_inference = MagicMock(spec=EPCensored)
        mock_inference.inference.return_value = (MagicMock(), -10.0, {'dL_dthetaL': 0, 'dL_dK': 0, 'dL_dm': 0})

        # Inject the mock
        model.inference_method = mock_inference

        # Trigger update
        model.parameters_changed()

        # Check call arguments manually
        call_args = mock_inference.inference.call_args
        args, _ = call_args

        # EPCensored signature: (kern, X, lik, Y, CENSORING, mean, metadata)
        # Censoring is the 5th argument (index 4)
        passed_censoring = args[4]

        self.assertTrue(np.array_equal(passed_censoring, model.censoring),
                        "EPCensored was not passed the explicit 1D censoring array")

    def test_parameters_changed_standard_inference(self):
        """
        Verify standard inference is NOT passed the extra censoring argument.
        """
        model = GPCensoredRegression(
            self.X, self.Y, self.censoring,
            kernel=self.real_kern
        )

        # Swap inference method to a Generic Mock (NOT EPCensored)
        mock_inference = MagicMock()
        mock_inference.inference.return_value = (MagicMock(), -5.0, {'dL_dthetaL': 0, 'dL_dK': 0, 'dL_dm': 0})
        model.inference_method = mock_inference

        # Trigger update
        model.parameters_changed()

        # Check arguments
        call_args = mock_inference.inference.call_args
        args, _ = call_args

        # Standard inference takes 6 args.
        self.assertLessEqual(len(args), 6, "Standard inference received too many arguments")

    def test_gradient_updates(self):
        """Ensure gradients returned by inference are passed to kernel/likelihood."""
        model = GPCensoredRegression(
            self.X, self.Y, self.censoring,
            kernel=self.real_kern
        )

        # Mock the inference to return specific gradients
        mock_inference = MagicMock(spec=EPCensored)
        grads = {
            'dL_dthetaL': np.array([0.5]),
            'dL_dK': np.random.randn(self.N, self.N),
            'dL_dm': None
        }
        mock_inference.inference.return_value = (MagicMock(), -10.0, grads)
        model.inference_method = mock_inference

        # Use patch.object to intercept calls to real GPy components
        with patch.object(model.likelihood, 'update_gradients') as mock_lik_update, \
                patch.object(model.kern, 'update_gradients_full') as mock_kern_update:
            model.parameters_changed()

            # --- FIX: Verify Likelihood Update ---
            self.assertTrue(mock_lik_update.called)
            args_lik, _ = mock_lik_update.call_args
            # Use Numpy assertion instead of assert_called_with
            np.testing.assert_array_equal(args_lik[0], grads['dL_dthetaL'])

            # --- FIX: Verify Kernel Update ---
            self.assertTrue(mock_kern_update.called)
            args_kern, _ = mock_kern_update.call_args
            # Args are (gradients, X)
            np.testing.assert_array_equal(args_kern[0], grads['dL_dK'])
            np.testing.assert_array_equal(args_kern[1], self.X)


if __name__ == '__main__':
    unittest.main()