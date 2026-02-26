import unittest
import torch
import numpy as np
import math


# Assuming the file above is named 'censored_normal.py'

from censored_regressors.distributions.censored_normal import CensoredNormal

class TestCensoredNormal(unittest.TestCase):
    def setUp(self):
        self.loc = torch.tensor([0.0, 5.0])
        self.scale = torch.tensor([1.0, 2.0])
        self.low = torch.tensor([-0.5, 4.0])
        self.high = torch.tensor([0.5, 6.0])
        self.dist = CensoredNormal(self.loc, self.scale, self.low, self.high)

    def test_shapes(self):
        """Test batch shapes and sample shapes."""
        self.assertEqual(self.dist.batch_shape, torch.Size([2]))

        # Sample shape (3, 2)
        samples = self.dist.sample((3,))
        self.assertEqual(samples.shape, torch.Size([3, 2]))

        # Check clamping
        self.assertTrue((samples[:, 0] >= -0.5).all())
        self.assertTrue((samples[:, 0] <= 0.5).all())
        self.assertTrue((samples[:, 1] >= 4.0).all())
        self.assertTrue((samples[:, 1] <= 6.0).all())

    def test_mean_accuracy_mc(self):
        """Check if analytical mean matches Monte Carlo estimation."""
        n_samples = 100000
        samples = self.dist.sample((n_samples,))
        mc_mean = samples.mean(dim=0)
        analytical_mean = self.dist.mean

        # Allow small tolerance for MC error
        self.assertTrue(torch.allclose(mc_mean, analytical_mean, atol=1e-2))

    def test_variance_accuracy_mc(self):
        """Check if analytical variance matches Monte Carlo estimation."""
        n_samples = 100000
        samples = self.dist.sample((n_samples,))
        mc_var = samples.var(dim=0)
        analytical_var = self.dist.variance

        self.assertTrue(torch.allclose(mc_var, analytical_var, atol=1e-2))

    def test_entropy_consistency(self):
        """
        Check if closed-form entropy matches log_prob expectation.
        H(x) = -E[log p(x)]
        """
        n_samples = 50000
        # For censored, p(x) has point masses.
        # Integration via MC on log_prob should roughly match entropy()
        samples = self.dist.sample((n_samples,))
        log_probs = self.dist.log_prob(samples)
        mc_entropy = -log_probs.mean(dim=0)

        # Entropy calculation on mixed continuous/discrete distributions
        # can be tricky with simple MC because hitting the exact boundary
        # in floating point is rare unless rsample specifically outputs it.
        # Our rsample clamps, so it hits boundaries.

        analytical_entropy = self.dist.entropy()
        self.assertTrue(torch.allclose(mc_entropy, analytical_entropy, atol=5e-2))

    def test_gradients(self):
        """Ensure gradients can flow through rsample and parameters."""
        loc = torch.tensor([0.0], requires_grad=True)
        scale = torch.tensor([1.0], requires_grad=True)
        dist = CensoredNormal(loc, scale, -0.5, 0.5)

        # Test 1: Reparameterization (rsample)
        sample = dist.rsample()
        loss = sample.sum()
        loss.backward()
        self.assertIsNotNone(loc.grad)
        self.assertIsNotNone(scale.grad)

        # Reset
        loc.grad = None
        scale.grad = None

        # Test 2: Analytical Mean
        mean_val = dist.mean
        mean_val.backward()
        self.assertIsNotNone(loc.grad)
        self.assertIsNotNone(scale.grad)


    def test_infinite_boundaries(self):
        """Test that setting boundaries to inf recovers standard Normal behavior."""
        loc = torch.tensor([0.0])
        scale = torch.tensor([1.0])
        # Effectively standard normal
        dist = CensoredNormal(loc, scale, float('-inf'), float('inf'))

        ref_normal = torch.distributions.Normal(loc, scale)

        self.assertTrue(torch.allclose(dist.mean, ref_normal.mean))
        self.assertTrue(torch.allclose(dist.variance, ref_normal.variance))
        self.assertTrue(torch.allclose(dist.entropy(), ref_normal.entropy()))


if __name__ == '__main__':
    unittest.main()