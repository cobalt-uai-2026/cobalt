import torch
import torch.nn as nn
import math
import unittest
import time
from numbers import Number
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

from censored_regressors.distributions.censored_normal import CensoredNormal

# ==========================================
# 1. ROBUST TOBIT LOSS (Fixed __init__)
# ==========================================
class RobustTobitLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RobustTobitLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mu_pred, sigma_pred, target, censorship):
        z = (target - mu_pred) / sigma_pred
        z = torch.clamp(z, min=-50., max=50.)

        log_pdf = -0.5 * math.log(2 * math.pi) - torch.log(sigma_pred) - 0.5 * (z ** 2)
        log_survival = torch.special.log_ndtr(-z)
        log_cdf = torch.special.log_ndtr(z)

        loss = torch.zeros_like(target)
        loss = torch.where(censorship == 0, log_pdf, loss)
        loss = torch.where(censorship == 1, log_survival, loss)
        loss = torch.where(censorship == -1, log_cdf, loss)

        nll = -loss
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

class TestTobitLikelihood(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.N = 1000
        self.mu = torch.randn(self.N)
        self.sigma = torch.rand(self.N) + 0.5
        self.target = torch.randn(self.N)

    def test_01_exact_equivalence(self):
        """
        Verify that the Updated CensoredNormal now matches RobustTobitLoss
        EXACTLY, even in extreme tails.
        """
        print("\n--- Test 1: Exact Equivalence (All Ranges) ---")

        # We purposely include extreme values here to verify the fix
        # Create a batch with normal values AND extreme values (Z=20)
        mu = torch.cat([self.mu, torch.tensor([0.0])])
        sigma = torch.cat([self.sigma, torch.tensor([1.0])])
        target = torch.cat([self.target, torch.tensor([20.0])])  # Z=20
        censorship = torch.cat([torch.randint(-1, 2, (self.N,)), torch.tensor([1])])

        # 1. Tobit Loss
        tobit_fn = RobustTobitLoss(reduction='none')
        loss_tobit = tobit_fn(mu, sigma, target, censorship)

        # 2. CensoredNormal (Vectorized bounds)
        # Construct low/high based on censorship for the CN class
        low = torch.where(censorship == -1, target, torch.tensor(-float('inf')))
        high = torch.where(censorship == 1, target, torch.tensor(float('inf')))

        cn = CensoredNormal(mu, sigma, low, high)
        loss_cn = -cn.log_prob(target)

        # Check difference
        diff = (loss_tobit - loss_cn).abs().max()
        print(f"Max Difference: {diff:.2e}")

        # With the fix, this should be effectively zero
        self.assertTrue(diff < 1e-5, "Implementations should be numerically identical")

    def test_02_gradient_clamping(self):
        """
        Verify that CensoredNormal now clamps gradients correctly.
        """
        print("\n--- Test 2: Gradient Clamping Check ---")
        mu = torch.tensor([0.0], requires_grad=True)
        sigma = torch.tensor([1.0], requires_grad=True)
        target = torch.tensor([100.0])  # Z=100

        # Right censored case
        cn = CensoredNormal(mu, sigma, low=torch.tensor(-float('inf')), high=target)
        loss = -cn.log_prob(target)
        loss.backward()

        print(f"Loss at Z=100: {loss.item():.4f}")
        print(f"Grad Mu: {mu.grad.item()}")

        # Gradient should be 0.0 due to clamp(-50, 50)
        self.assertEqual(mu.grad.item(), 0.0, "Gradient should be clamped to 0 for extreme outliers")


if __name__ == "__main__":
    unittest.main()