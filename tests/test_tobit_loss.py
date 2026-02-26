import torch
import torch.nn as nn
import torch.autograd as autograd
import math
import unittest

from censored_regressors.losses.tobit_loss import GaussianNLLLoss, RobustTobitLoss

class TestLossGradients(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        # Use double precision for sensitive gradient checks
        torch.set_default_dtype(torch.float64)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def compute_second_derivative(self, loss_fn, mu, sigma, target, censorship):
        """Helper to compute d^2(Loss)/d(mu)^2"""
        mu = mu.clone().detach().requires_grad_(True)
        sigma = sigma.clone().detach().requires_grad_(True)

        # UPDATED CALL: Signature is now (mu, target, censorship, sigma)
        loss = loss_fn(mu_pred=mu, target=target, censorship=censorship, sigma_pred=sigma)

        # 1. First Derivative
        grad_mu = autograd.grad(loss, mu, create_graph=True)[0]

        # 2. Second Derivative
        grad2_mu = autograd.grad(grad_mu, mu, retain_graph=True)[0]

        return loss, grad_mu, grad2_mu

    # ==========================================
    # TEST 1: Extreme Value Stability (Forward)
    # ==========================================
    def test_extreme_forward_stability(self):
        print("\n--- Test 1: Extreme Forward Stability ---")

        # Case: Z = 100 (Way past clamp limit of 50)
        mu = torch.tensor([0.0])
        sigma = torch.tensor([1.0])
        target = torch.tensor([100.0])
        censorship = torch.tensor([1.0])  # Right censored

        # Tobit
        tobit = RobustTobitLoss()
        # UPDATED CALL
        loss_t = tobit(mu_pred=mu, target=target, censorship=censorship, sigma_pred=sigma)

        # Gaussian
        gauss = GaussianNLLLoss()
        # UPDATED CALL
        loss_g = gauss(mu_pred=mu, target=target, censorship=censorship, sigma_pred=sigma)

        print(f"Tobit Loss (Z=100): {loss_t.item()}")
        print(f"Gauss Loss (Z=100): {loss_g.item()}")

        self.assertFalse(torch.isnan(loss_t), "Tobit loss returned NaN for extreme input")
        self.assertFalse(torch.isinf(loss_t), "Tobit loss returned Inf for extreme input")

        # Verify Clamping Logic:
        # Z=50 -> log_ndtr(-50) is approx -1253.
        # NLL = -log_survival approx 1253.
        self.assertTrue(1200 < loss_t.item() < 1300, f"Tobit loss {loss_t.item()} is not consistent with clamped Z=50")

    # ==========================================
    # TEST 2: First Gradient Correctness
    # ==========================================
    def test_first_gradient_behavior(self):
        print("\n--- Test 2: First Gradient Check ---")

        # Case A: Normal Range (Z=0)
        mu = torch.tensor([0.0], requires_grad=True)
        sigma = torch.tensor([1.0], requires_grad=True)
        target = torch.tensor([0.0])
        censorship = torch.tensor([0.0])  # Uncensored

        # UPDATED CALL
        loss = RobustTobitLoss()(mu_pred=mu, target=target, censorship=censorship, sigma_pred=sigma)
        loss.backward()

        self.assertFalse(torch.isnan(mu.grad), "Gradient is NaN in normal range")
        print(f"Normal Range Grad: {mu.grad.item()}")

        # Case B: The Clamp Boundary (Z > 50)
        # Gradient should be EXACTLY zero because the value is clamped constant inside the Loss
        mu_ext = torch.tensor([0.0], requires_grad=True)
        sigma_ext = torch.tensor([1.0], requires_grad=True)
        target_ext = torch.tensor([100.0])  # Z = 100
        censorship_ext = torch.tensor([1.0])

        # UPDATED CALL
        loss_ext = RobustTobitLoss()(mu_ext, target_ext, censorship_ext, sigma_pred=sigma_ext)
        loss_ext.backward()

        print(f"Extreme Range Grad: {mu_ext.grad.item()}")
        self.assertEqual(mu_ext.grad.item(), 0.0, "Gradient should be 0.0 when clamped")

    # ==========================================
    # TEST 3: Second Gradient (Hessian) Check
    # ==========================================
    def test_second_gradient_stability(self):
        print("\n--- Test 3: Second Derivative (Hessian) Stability ---")

        mu = torch.tensor([0.0])
        sigma = torch.tensor([1.0])
        target = torch.tensor([2.0])  # Z=2
        censorship = torch.tensor([1.0])  # Right censored

        loss, grad1, grad2 = self.compute_second_derivative(RobustTobitLoss(), mu, sigma, target, censorship)

        print(f"Loss:  {loss.item():.4f}")
        print(f"Grad1: {grad1.item():.4f}")
        print(f"Grad2: {grad2.item():.4f}")

        self.assertFalse(torch.isnan(grad2), "Second derivative is NaN")
        self.assertFalse(torch.isinf(grad2), "Second derivative is Inf")

        # Check Hessian at Clamp Boundary
        target_ext = torch.tensor([100.0])
        _, g1_ext, g2_ext = self.compute_second_derivative(RobustTobitLoss(), mu, sigma, target_ext, censorship)

        print(f"Clamp Boundary Grad2: {g2_ext.item()}")
        self.assertEqual(g2_ext.item(), 0.0, "Second derivative should be 0 in clamped region")

    # ==========================================
    # TEST 4: Tobit vs Gaussian Equivalence
    # ==========================================
    def test_uncensored_equivalence(self):
        print("\n--- Test 4: Tobit vs Gaussian Consistency ---")
        # When censorship=0, Tobit should equal GaussianNLL
        mu = torch.randn(10)
        sigma = torch.rand(10) + 0.1
        target = torch.randn(10)
        censorship = torch.zeros(10)

        tobit_fn = RobustTobitLoss(reduction='mean')
        gauss_fn = GaussianNLLLoss()

        # UPDATED CALLS
        loss_t = tobit_fn(mu_pred=mu, target=target, censorship=censorship, sigma_pred=sigma)
        loss_g = gauss_fn(mu_pred=mu, target=target, censorship=censorship, sigma_pred=sigma)

        diff = abs(loss_t.item() - loss_g.item())
        print(f"Difference: {diff:.2e}")
        self.assertLess(diff, 1e-6, "Tobit and Gaussian should match for uncensored data")


if __name__ == "__main__":
    unittest.main()