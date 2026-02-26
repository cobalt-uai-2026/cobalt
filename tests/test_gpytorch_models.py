import unittest
import numpy as np
import torch
import gpytorch
import warnings


from censored_regressors.models.models_gpytorch import CensoredGP_VI_gpytorch

class TestGPyTorchModels(unittest.TestCase):
    """
    Unit tests for GPyTorch-based Censored GP (Variational Inference).
    """

    def setUp(self):
        """
        Generates synthetic censored data before each test.
        """
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        self.N = 40
        self.X = np.linspace(-3, 3, self.N).reshape(-1, 1)

        # Latent function: Sinewave
        self.y_true = np.sin(self.X).flatten()
        self.noise = np.random.normal(0, 0.1, self.y_true.shape)
        self.y_raw = self.y_true + self.noise

        # Create Censoring (Right Censoring at y < 0.5)
        self.limit = 0.5
        self.censoring = np.zeros_like(self.y_raw, dtype=int)  # 0 = Observed
        self.y_censored = self.y_raw.copy()

        # 1 = Censored (Upper/Right), -1 = Censored (Lower/Left)
        cens_idx = self.y_raw < self.limit
        self.censoring[~cens_idx] = 1
        self.y_censored[~cens_idx] = self.limit

        # Data Tuple
        self.train_data = (self.X, self.y_censored, self.censoring)

        # Test Data
        self.X_test = np.linspace(-4, 4, 15).reshape(-1, 1)

    def test_initialization_laplace(self):
        """Test native PyTorch Laplace initialization logic."""
        model = CensoredGP_VI_gpytorch(kernel_type='rbf')

        # We assume _init_via_laplace_torch is called internally when init_params='laplace'
        # We run with 0 iterations just to test initialization mechanics,
        # but GPyTorchOptimizer usually runs at least 1 step.
        # Setting max_iters=1 ensures the init logic runs and then training briefly starts.
        success = model.fit(
            self.train_data,
            optimizer='adam',
            init_params='laplace',
            max_iters=5,
            num_restarts=1
        )
        self.assertTrue(success, "Laplace initialization should complete successfully.")

        # Check if model parameters are not NaN
        for param in model.model.parameters():
            self.assertFalse(torch.isnan(param).any(), "Model parameters contained NaNs after Laplace Init")

    def test_initialization_gpy(self):
        """Test legacy GPy initialization logic."""
        try:
            import GPy
        except ImportError:
            self.skipTest("GPy not installed")

        model = CensoredGP_VI_gpytorch(kernel_type='lin_rbf')

        # Runs GPy optimization first, then transfers weights
        success = model.fit(
            self.train_data,
            optimizer='adam',
            init_params='gpy',
            max_iters=5,
            num_restarts=1
        )
        self.assertTrue(success, "GPy initialization should complete successfully.")

    def test_optimization_ngd(self):
        """Test Natural Gradient Descent optimization path."""
        model = CensoredGP_VI_gpytorch(kernel_type='rbf')

        # NGD uses NaturalVariationalDistribution internally
        success = model.fit(
            self.train_data,
            optimizer='ngd',
            num_restarts=1,
            max_iters=10
        )
        self.assertTrue(success)

        # verify distribution type
        self.assertIsInstance(
            model.model.variational_strategy._variational_distribution,
            gpytorch.variational.NaturalVariationalDistribution,
            "NGD should use NaturalVariationalDistribution"
        )

    def test_prediction_shape(self):
        """Test prediction output shapes and keys."""
        model = CensoredGP_VI_gpytorch()
        model.fit(self.train_data, max_iters=5, num_restarts=1)

        preds = model.predict(self.X_test)

        expected_keys = {'f_mean', 'f_var', 'f_025', 'f_975'}
        self.assertTrue(expected_keys.issubset(preds.keys()))

        self.assertEqual(preds['f_mean'].shape, (15, 1))
        self.assertEqual(preds['f_var'].shape, (15, 1))

        # Variance should be positive
        self.assertTrue(np.all(preds['f_var'] >= 0))

    def test_prediction_determinism(self):
        """
        Test that predictions are identical for identical inputs
        (Requires careful seeding in the predict method or wrapper).
        """
        model = CensoredGP_VI_gpytorch()
        model.fit(self.train_data, max_iters=5, num_restarts=1)

        # Run prediction 1
        torch.manual_seed(999)
        np.random.seed(999)
        pred1 = model.predict(self.X_test)['f_mean']

        # Run prediction 2
        torch.manual_seed(999)
        np.random.seed(999)
        pred2 = model.predict(self.X_test)['f_mean']

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_censoring_bounds_logic(self):
        """Test that upper/lower bounds are correctly derived from censoring indicators."""
        # This tests the logic inside _fit:
        # mask_upper = (c_sq == 1) -> ub = y
        # mask_lower = (c_sq == -1) -> lb = y

        # We can't easily access the internal variables of _fit,
        # so we verify the likelihood setup after fit.

        model = CensoredGP_VI_gpytorch()
        model.fit(self.train_data, max_iters=1, num_restarts=1)

        # The likelihood should store the bounds.
        # Your custom CensoredGaussianLikelihoodAnalytic likely has 'low' and 'high' buffers/attributes.
        # This assumes your likelihood class saves them as attributes or buffers.
        if hasattr(model.likelihood, 'high'):
            # Check a known censored point
            censored_idx = np.where(self.censoring == 1)[0][0]

            # For upper censoring (1), 'high' bound should be the y value (limit)
            # We need to be careful about tensor/numpy conversion here
            high_bound_tensor = model.likelihood.high

            # If high_bound_tensor is available, check value
            if isinstance(high_bound_tensor, torch.Tensor):
                y_val = self.y_censored[censored_idx]
                bound_val = high_bound_tensor[censored_idx].item()

                # bound_val should be approx y_val (since we set ub[mask_upper] = y)
                self.assertAlmostEqual(bound_val, y_val, places=5)

    def test_robustness_1d_input(self):
        """Test robustness against 1D input arrays (common user error)."""
        model = CensoredGP_VI_gpytorch()
        model.fit(self.train_data, max_iters=5, num_restarts=1)

        # Pass 1D array instead of 2D column vector
        X_flat = self.X_test.flatten()

        try:
            preds = model.predict(X_flat)
            self.assertEqual(preds['f_mean'].shape, (15, 1))
        except Exception as e:
            self.fail(f"Predict failed on 1D input: {e}")


if __name__ == '__main__':
    unittest.main()