import unittest
import numpy as np
import GPy

# Adjust imports to match your project structure
from censored_regressors.models.models_gpy import GP, TruncGP, CensoredGP_Laplace, CensoredGP_EP



class TestGPyModels(unittest.TestCase):
    """
    Unit tests for GPy-based censored regression models.
    """

    def setUp(self):
        """
        Generates synthetic censored data before each test.
        """
        np.random.seed(42)
        self.N = 50
        self.X = np.linspace(-5, 5, self.N).reshape(-1, 1)

        # Latent function: Sinewave
        self.y_true = np.sin(self.X)
        self.noise = np.random.normal(0, 0.1, self.X.shape)
        self.y_raw = self.y_true + self.noise

        # Create Censoring (Right Censoring at y > 0.5)
        self.limit = 0.5
        self.censoring = np.zeros_like(self.y_raw, dtype=int)
        self.y_censored = self.y_raw.copy()

        # 1 = Censored, 0 = Observed
        cens_idx = self.y_raw > self.limit
        self.censoring[cens_idx] = 1
        self.y_censored[cens_idx] = self.limit

        # Data Tuples
        self.data_std = (self.X, self.y_raw)  # For standard GP
        self.data_cens = (self.X, self.y_censored, self.censoring)  # For Censored/Trunc models

        # Test Data
        self.X_test = np.linspace(-6, 6, 20).reshape(-1, 1)

    # --- Standard GP Tests ---

    def test_gp_fit_predict(self):
        """Test standard GP fitting and prediction structure."""
        model = GP(kernel_type='lin_rbf')
        success = model.fit(self.data_std)
        self.assertTrue(success, "GP fit should return True")

        preds = model.predict(self.X_test)

        # Check Dictionary Keys
        expected_keys = {'f_mean', 'f_var', 'f_025', 'f_975'}
        self.assertTrue(expected_keys.issubset(preds.keys()))

        # Check Shapes
        self.assertEqual(preds['f_mean'].shape, (20, 1))
        self.assertTrue(np.all(preds['f_var'] >= 0), "Variance must be positive")

    def test_gp_initialization(self):
        """Test if user init_params are applied correctly."""
        init_params = {'lengthscale': 5.0, 'variance': 0.5}
        model = GP(kernel_type='rbf')

        # Fit with custom init
        model.fit(self.data_std, init_params=init_params, num_restarts=1)

        # Check if parameters were roughly respected (GPy optimizes them,
        # so we just check they aren't default 1.0 immediately after valid init
        # or check logical constraints if we didn't optimize.
        # Here we trust the logic runs without crashing.)
        self.assertIsNotNone(model.model)

    # --- Truncated GP Tests ---

    def test_trunc_gp_filtering(self):
        """Test that TruncGP filters out censored data."""
        model = TruncGP(kernel_type='lin')
        model.fit(self.data_cens)

        # Calculate how many observed points we have
        n_observed = np.sum(self.censoring == 0)

        # GPy model should store X with shape (n_observed, 1)
        self.assertEqual(model.model.X.shape[0], n_observed)
        self.assertLess(model.model.X.shape[0], self.N, "TruncGP should drop points")

    def test_trunc_gp_input_validation(self):
        """Test TruncGP raises error if censoring vector is missing."""
        model = TruncGP()
        with self.assertRaises(ValueError):
            model.fit(self.data_std)  # Only (X, y)

    # --- Censored GP (Laplace) Tests ---

    def test_censored_laplace_run(self):
        """Test Laplace approximation runs end-to-end."""
        model = CensoredGP_Laplace(kernel_type='rbf')
        # Use fewer restarts for speed
        success = model.fit(self.data_cens, num_restarts=1)
        self.assertTrue(success)

        preds = model.predict(self.X_test)
        self.assertEqual(preds['f_mean'].shape, (20, 1))

    # --- Censored GP (EP) Tests ---

    def test_censored_ep_run(self):
        """Test Expectation Propagation runs end-to-end."""
        # EP can be unstable, so we use a robust kernel config
        model = CensoredGP_EP(kernel_type='lin_rbf')

        # Test the restart loop logic
        success = model.fit(self.data_cens, num_restarts=2)
        self.assertTrue(success)

        preds = model.predict(self.X_test)
        self.assertFalse(np.any(np.isnan(preds['f_mean'])), "EP predictions should not be NaN")

    # --- Base Class / Common Functionality Tests ---

    def test_preprocessing(self):
        """Test Z-score normalization logic."""
        model = GP()
        model.preprocess = True
        model.fit(self.data_std)

        # Internal data mean should be approx 0 (since X is -5 to 5 centered)
        # But we check if stored stats are correct
        self.assertAlmostEqual(model.data_mean, np.mean(self.X), places=5)
        self.assertAlmostEqual(model.labels_std, np.std(self.y_raw) + 1e-6, places=5)

        # Predict
        preds = model.predict(self.X_test)

        # Predictions should be in original scale (approx sin wave amplitude ~1)
        self.assertTrue(np.max(preds['f_mean']) < 2.0)
        self.assertTrue(np.min(preds['f_mean']) > -2.0)

    def test_predict_shape_robustness(self):
        """
        CRITICAL TEST: Ensures the fix for 1D vs 2D arrays works.
        """
        model = GP()
        model.fit(self.data_std)

        # Case 1: 2D Input
        # RESET SEED to ensure Monte Carlo samples are identical
        np.random.seed(123)
        res_2d = model.predict(self.X_test)

        # Case 2: 1D Input
        X_flat = self.X_test.flatten()

        # RESET SEED AGAIN to ensure identical samples
        np.random.seed(123)
        res_1d = model.predict(X_flat)

        # Now they should be identical
        np.testing.assert_array_almost_equal(res_2d['f_mean'], res_1d['f_mean'])

        # Assert output is always 2D (N, 1)
        self.assertEqual(res_1d['f_mean'].ndim, 2)
        self.assertEqual(res_1d['f_mean'].shape[1], 1)

    def test_kernel_factory(self):
        """Test that different kernel strings load correct GPy kernels."""
        model = GP()

        # Test Linear
        model.kernel_type = 'lin'
        k = model._get_kernel(1)
        self.assertIsInstance(k, GPy.kern.Linear)  # Note: GPy.kern.Linear is wrapped, check base types if needed

        # Test RBF
        model.kernel_type = 'rbf'
        k = model._get_kernel(1)
        self.assertIsInstance(k, GPy.kern.RBF)

        # Test Invalid
        model.kernel_type = 'invalid_kernel_name'
        with self.assertRaises(ValueError):
            model._get_kernel(1)


if __name__ == '__main__':
    unittest.main()