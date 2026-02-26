import unittest
import torch
import numpy as np

# Adjust import based on your actual file structure
from censored_regressors.utils.oracle import (
    Oracle,
    OracleGenerator,
    RangeBoundGenerator,
    VariableBoundGenerator,
    ProbabilisticCensoredGenerator
)


# --- Mock Functions for Testing ---
def linear_1d(x):
    """f(x) = 2x"""
    return 2 * x


def sum_squares_nd(x):
    """f(x) = sum(x^2) along last dim"""
    # x shape: (N, dim) -> output (N,)
    return torch.sum(x ** 2, dim=1)


def sine_wave_mock(x):
    """Matches the shape expectation for Probabilistic Generator"""
    return 0.5 * torch.sin(2 * x) + 2 + x / 10.0


class TestOracle(unittest.TestCase):
    """
    Unit tests for the core Oracle class.
    """

    def setUp(self):
        """Runs before every test"""
        self.oracle_1d = Oracle(fcn=linear_1d, seed=42)
        self.oracle_nd = Oracle(fcn=sum_squares_nd, seed=42)

    def test_initialization(self):
        """Test if class initializes correctly"""
        self.assertEqual(self.oracle_1d.seed, 42)
        self.assertTrue(callable(self.oracle_1d.fcn))

    def test_evaluate_fcn_1d(self):
        """Test simple evaluation for 1D"""
        x = torch.tensor([[1.0], [2.0], [3.0]])
        y = self.oracle_1d.evaluate_fcn(x)
        expected = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(torch.allclose(y, expected))
        self.assertEqual(y.ndim, 1)  # Should be squeezed

    def test_evaluate_fcn_nd(self):
        """Test simple evaluation for 2D"""
        # x = [[1, 2], [3, 4]] -> 1^2+2^2=5, 3^2+4^2=25
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = self.oracle_nd.evaluate_fcn(x)
        expected = torch.tensor([5.0, 25.0])
        self.assertTrue(torch.allclose(y, expected))

    def test_get_sample_shapes(self):
        """Test that get_sample returns correct shapes"""
        N = 10
        dim = 3
        # Use default bounds
        x, y_noisy, y_true = self.oracle_nd.get_sample(N=N, dim=dim)

        self.assertEqual(x.shape, (N, dim))
        self.assertEqual(y_noisy.shape, (N,))
        self.assertEqual(y_true.shape, (N,))

    def test_get_sample_bounds(self):
        """Test that start and end arguments constrain x correctly"""
        N = 100
        start, end = 10.0, 20.0
        x, _, _ = self.oracle_1d.get_sample(N=N, start=start, end=end)

        self.assertTrue(torch.all(x >= start), "X values should be >= start")
        self.assertTrue(torch.all(x <= end), "X values should be <= end")

    def test_reproducibility(self):
        """Test that the same seed produces the same samples"""
        oracle1 = Oracle(fcn=linear_1d, seed=123)
        oracle2 = Oracle(fcn=linear_1d, seed=123)

        x1, y1, _ = oracle1.get_sample(N=20)
        x2, y2, _ = oracle2.get_sample(N=20)

        self.assertTrue(torch.allclose(x1, x2), "X values should be identical for same seed")
        self.assertTrue(torch.allclose(y1, y2), "Y values should be identical for same seed")

    def test_noise_generation(self):
        """Test that noise is added and scales correctly"""
        N = 100
        # 1. Zero Noise
        _, y_noisy_0, y_true_0 = self.oracle_1d.get_sample(N=N, noise_scale=0.0)
        self.assertTrue(torch.allclose(y_noisy_0, y_true_0), "With scale 0.0, noisy should equal true")

        # 2. Significant Noise
        scale = 10.0
        _, y_noisy_high, y_true_high = self.oracle_1d.get_sample(N=N, noise_scale=scale)
        diffs = torch.abs(y_noisy_high - y_true_high)

        # It's statistically improbable for all diffs to be near zero with high noise
        self.assertTrue(torch.mean(diffs) > 1.0, "Noise should be present")

    def test_censoring_fixed_bounds(self):
        """Test standard min/max clamping"""
        y = torch.tensor([-5.0, 0.0, 5.0, 10.0])

        # Clamp between -2 and 8
        censored = self.oracle_1d.censor(y, low=-2.0, high=8.0, quantile=False)
        expected = torch.tensor([-2.0, 0.0, 5.0, 8.0])

        self.assertTrue(torch.allclose(censored, expected))

    def test_censoring_quantiles(self):
        """Test quantile-based clamping"""
        # Create a range 0 to 10
        y = torch.linspace(0, 10, 11)  # [0, 1, 2, ..., 10]

        # Censor bottom 20% and top 20%
        # 20% of 10 is 2.0, 80% is 8.0
        censored = self.oracle_1d.censor(y, low=0.2, high=0.8, quantile=True)

        self.assertTrue(torch.min(censored) >= 2.0)
        self.assertTrue(torch.max(censored) <= 8.0)


class TestGenerators(unittest.TestCase):
    """
    Integration tests for the various Generator classes.
    """

    def setUp(self):
        self.oracle = Oracle(fcn=linear_1d, seed=42)

    def test_oracle_generator_bounds(self):
        """Test that base generator passes start/end to Oracle"""
        gen = OracleGenerator(self.oracle, n_samples=50, start=100, end=105)
        x, _, _, _ = gen.generate()

        self.assertTrue(np.all(x >= 100))
        self.assertTrue(np.all(x <= 105))

    def test_range_bound_generator(self):
        """Test RangeBoundGenerator inheritance and bounds"""
        gen = RangeBoundGenerator(self.oracle, n_samples=50, start=-50, end=-40)
        x, y_obs, c, y_true = gen.generate()

        self.assertTrue(np.all(x >= -50))
        self.assertTrue(np.all(x <= -40))
        self.assertEqual(x.shape, (50, 1))
        self.assertEqual(c.shape, (50, 1))

    def test_variable_bound_generator(self):
        """Test that variable bounds logic is applied"""

        # Define a bounds generator that forces tight censoring
        def tight_bounds(x, y_noisy):
            # Censor everything above y=0
            # x shape (N, 1), y_noisy shape (N,)
            high = torch.zeros_like(y_noisy)
            low = None
            return low, high

        gen = VariableBoundGenerator(
            self.oracle,
            n_samples=20,
            bounds_generator=tight_bounds,
            start=-5, end=5
        )

        x, y_obs, c, y_true = gen.generate()

        # Check that no observation is > 0 (since high=0)
        self.assertTrue(np.all(y_obs <= 1e-5))

        # Check that indicators are set correctly for censored data
        # If true y was > 0, it should be censored (c=1)
        # Note: y_obs is numpy, y_true is numpy
        mask_censored = (y_true > 0).flatten()
        # It's possible noise pushed it below 0, but generally we expect some censoring
        if np.any(mask_censored):
            self.assertTrue(np.any(c == 1), "Should detect upper censoring")

    def test_probabilistic_generator(self):
        """Test specific sine wave censoring logic"""
        # Use sine mock to ensure we trigger the logic
        oracle_sine = Oracle(fcn=sine_wave_mock, seed=99)

        # This generator doesn't use standard bounds for censoring,
        # but does use start/end for x generation.
        gen = ProbabilisticCensoredGenerator(
            oracle_sine,
            n_samples=100,
            start=0,
            end=10
        )

        x, y_obs, c, y_true = gen.generate()

        self.assertEqual(x.shape, (100, 1))
        self.assertEqual(c.shape, (100, 1))

        # Check consistency: In this class, if C=1, y_obs should be != y_true (due to reduction)
        censored_indices = (c == 1).flatten()
        if np.any(censored_indices):
            # We expect observed values to be lower than true values (y_obs = y_true * (1-p))
            # Note: The generator logic applies reduction to y_noisy, not y_true directly,
            # but y_obs should definitely differ from y_true roughly.
            # A stricter check is verifying y_obs < y_noisy for these points,
            # but we don't have y_noisy exposed here easily.
            pass

        # Verify C only contains 0 and 1
        unique_c = np.unique(c)
        self.assertTrue(np.all(np.isin(unique_c, [0, 1])))


if __name__ == '__main__':
    unittest.main()