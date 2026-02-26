import math
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
from torch import Tensor
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from censored_regressors.distributions.censored_normal import CensoredNormal

__all__ = ["CensoredGaussianLikelihood", "CensoredGaussianLikelihoodAnalytic"]


class CensoredGaussianLikelihood(Likelihood):
    """
    Base class for Censored Gaussian Likelihoods.

    forward(): Returns the predictive distribution p(y|f) as a CensoredNormal.
    """

    def __init__(self, variance=None, low=None, high=None) -> None:
        super().__init__()

        # 1. Initialize Noise Model (sigma^2)
        # Ensure variance is a tensor
        if variance is None:
            variance = torch.tensor(1.0)
        elif not torch.is_tensor(variance):
            variance = torch.tensor(float(variance))

        self.noise = HomoskedasticNoise()
        self.noise.initialize(noise=variance)

        # 2. Register Bounds as Buffers
        # We use +/- infinity for missing bounds to simplify logic.
        # Note: register_buffer automatically creates self.low and self.high attributes
        # and ensures they move to the GPU/CPU along with the model.
        if low is None:
            low_t = torch.tensor(-float('inf'))
        else:
            low_t = torch.as_tensor(low)

        if high is None:
            high_t = torch.tensor(float('inf'))
        else:
            high_t = torch.as_tensor(high)

        self.register_buffer("low", low_t.float())
        self.register_buffer("high", high_t.float())

    @property
    def variance(self):
        return self.noise.noise

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any):
        """
        Computes p(y|f).

        Args:
            function_samples: The latent function values (f)
        Returns:
            CensoredNormal distribution parameterized by f, sigma, low, high.
        """
        return CensoredNormal(
            loc=function_samples,
            scale=self.variance.sqrt(),
            low=self.low,
            high=self.high
        )


class CensoredGaussianLikelihoodAnalytic(CensoredGaussianLikelihood):
    """
    Analytic implementation supporting multiple integration backends:
    1. 'gauss_hermite': Fast, differentiable, standard for GPs.
    2. 'trapez': Robust grid-based baseline.
    3. 'simpson': Higher-order grid-based baseline.
    """

    def __init__(self, variance=None, low=None, high=None,
                 alpha=1., gamma=1., dzeta=1.,
                 integration_type='gauss_hermite',
                 n_points=30,
                 grid_width=12.0) -> None:
        """
        Args:
            n_points: Number of quadrature nodes or grid points.
            grid_width: For grid methods, integrates over [-width, +width] sigma.
        """
        # Pass variance/low/high to the parent to ensure self.low/self.high are registered
        super().__init__(variance=variance, low=low, high=high)

        # Loss scaling factors
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.register_buffer("dzeta", torch.tensor(float(dzeta)))

        self.integration_type = integration_type
        self.n_points = n_points
        self.grid_width = grid_width

        # Pre-compute buffers based on type
        if integration_type == 'gauss_hermite':
            gh_x, gh_w = np.polynomial.hermite.hermgauss(n_points)
            self.register_buffer("gh_x", torch.from_numpy(gh_x).double())
            self.register_buffer("gh_w", torch.from_numpy(gh_w).double())

        elif integration_type in ['trapez', 'simpson']:
            # For grid methods, we pre-calculate the standard normal grid
            if integration_type == 'simpson' and n_points % 2 == 0:
                # Simpson's rule requires odd number of points
                n_points += 1

            grid = torch.linspace(-grid_width, grid_width, n_points, dtype=torch.float64)
            self.register_buffer("grid_x", grid)

            # Pre-compute standard normal pdf weights e^(-0.5 x^2)
            self.register_buffer("grid_log_w", -0.5 * grid.pow(2))
            self.dx = (grid[1] - grid[0]).item()

    def expected_log_prob(self, target: Tensor, input: dist.MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        terms = self._expected_log_prob_terms(target, input, *params, **kwargs)
        return (self.alpha * terms['normal_part'] +
                self.gamma * terms['upper_censored_part'] +
                self.dzeta * terms['lower_censored_part'])

    def _expected_log_prob_terms(self, target: Tensor, input: dist.MultivariateNormal, *params: Any,
                                 **kwargs: Any) -> dict:
        mean, variance = input.mean, input.variance
        noise = self.variance
        sigma = noise.sqrt()
        std = variance.sqrt()

        # Ensure bounds are available (safety check)
        if not hasattr(self, 'low') or not hasattr(self, 'high'):
            raise RuntimeError("Likelihood bounds (low/high) not initialized. Ensure super().__init__ was called.")

        # --- FIX: ROBUST MASKING ---
        # 1. Cast bounds to match target dtype (e.g., if target is Double, bounds become Double)
        low_cast = self.low.to(dtype=target.dtype, device=target.device)
        high_cast = self.high.to(dtype=target.dtype, device=target.device)

        # 2. Use Epsilon for float comparison robustness
        # If the value is within 1e-5 of the bound, consider it censored.
        eps = 1e-5

        # Mask creation
        is_left = (target <= low_cast + eps)
        is_right = (target >= high_cast - eps)
        is_observed = ~(is_left | is_right)

        # 1. Continuous Part
        normal_term = -0.5 * (math.log(2 * math.pi) + noise.log() + ((target - mean).square() + variance) / noise)

        # 2. Censored Parts
        # Replace Infs with 0.0 for calculation safety (masked out later anyway)
        safe_high = torch.where(torch.isinf(high_cast), torch.zeros_like(high_cast), high_cast)
        safe_low = torch.where(torch.isinf(low_cast), torch.zeros_like(low_cast), low_cast)

        # A: (bound - mu) / sigma
        # B: std_dev / sigma
        b = std / sigma

        # Upper Censored Term: log Phi( (mu - high) / sigma_combined? )
        # Actually, integrating log Phi( (high - f)/sigma ) over q(f)
        # Note: definition of a_upper depends on integral direction.
        # Using standard form: a = (mean - bound) / sigma
        a_upper = (mean - safe_high) / sigma
        a_lower = (safe_low - mean) / sigma

        censored_val_upper = self._integrate_log_phi(a_upper, b, dtype=target.dtype)
        censored_val_lower = self._integrate_log_phi(a_lower, b, dtype=target.dtype)

        # 3. Combine using Masks
        res = torch.zeros_like(target)
        normal_part = torch.where(is_observed, normal_term, res)
        upper_censored_part = torch.where(is_right, censored_val_upper, res)
        lower_censored_part = torch.where(is_left, censored_val_lower, res)

        return {
            'normal_part': normal_part,
            'upper_censored_part': upper_censored_part,
            'lower_censored_part': lower_censored_part
        }

    def _integrate_log_phi(self, a: Tensor, b: Tensor, return_error: bool = False, dtype=torch.float32) -> Union[
        Tensor, Tuple[Tensor, float]]:
        """
        Computes E_{z~N(0,1)} [ log Phi(a + b*z) ].
        """
        # Ensure computation happens in Double if input is Double for precision
        orig_dtype = a.dtype if dtype is None else dtype
        a_d, b_d = a.double(), b.double()

        # --- A. Gauss-Hermite Quadrature ---
        if self.integration_type == 'gauss_hermite':
            # Ensure GH weights match device/dtype
            gh_x = self.gh_x.to(device=a.device, dtype=torch.float64)
            gh_w = self.gh_w.to(device=a.device, dtype=torch.float64)

            nodes = gh_x.unsqueeze(0)  # [1, Q]
            z = nodes * math.sqrt(2.0)
            arg = a_d.unsqueeze(-1) + b_d.unsqueeze(-1) * z  # [Batch, Q]

            log_phi_vals = self._log_phi_robust(arg)

            # Weighted sum: sum( w_i * f(z_i) ) / sqrt(pi)
            weighted_sum = (log_phi_vals * gh_w).sum(dim=-1)
            result = weighted_sum / math.sqrt(math.pi)

            if return_error:
                return result.to(dtype=orig_dtype), 0.0
            return result.to(dtype=orig_dtype)

        # --- B. Grid-Based Methods (Trapez / Simpson) ---
        else:
            grid_x = self.grid_x.to(device=a.device, dtype=torch.float64)
            grid_log_w = self.grid_log_w.to(device=a.device, dtype=torch.float64)

            z = grid_x.unsqueeze(0)  # [1, N]

            # 1. Evaluate Function: log Phi(a + bz)
            arg = a_d.unsqueeze(-1) + b_d.unsqueeze(-1) * z
            f_vals = self._log_phi_robust(arg)

            # 2. Multiply by PDF weight
            integrand = f_vals * torch.exp(grid_log_w)

            # 3. Integrate
            if self.integration_type == 'trapez':
                inner_sum = integrand[..., 1:-1].sum(dim=-1)
                ends = 0.5 * (integrand[..., 0] + integrand[..., -1])
                area = (inner_sum + ends) * self.dx
                result = area / math.sqrt(2 * math.pi)
                error = 0.0

            elif self.integration_type == 'simpson':
                y = integrand
                sum_odds = 4.0 * y[..., 1:-1:2].sum(dim=-1)
                sum_evens = 2.0 * y[..., 2:-2:2].sum(dim=-1)
                total = y[..., 0] + y[..., -1] + sum_odds + sum_evens
                area = (self.dx / 3.0) * total
                result = area / math.sqrt(2 * math.pi)
                error = 0.0

            if return_error:
                return result.to(dtype=orig_dtype), error
            return result.to(dtype=orig_dtype)

    def _log_phi_robust(self, z: Tensor) -> Tensor:
        """
        Robust log(Phi(z)) for double precision inputs.
        """
        mask_neg = z < -5.5
        mask_pos = z > 10.0
        mask_mid = ~(mask_neg | mask_pos)

        res = torch.zeros_like(z)

        if mask_mid.any():
            val = z[mask_mid]
            res[mask_mid] = torch.log(0.5 * torch.special.erfc(-val / math.sqrt(2.0)).clamp(min=1e-100))

        if mask_neg.any():
            val = z[mask_neg]
            log_2pi = math.log(2 * math.pi)
            # Asymptotic expansion for log(Phi(z)) when z << -1
            res[mask_neg] = -0.5 * val.pow(2) - torch.log(-val) - 0.5 * log_2pi

        # For mask_pos (z > 10), log(Phi(z)) -> log(1) = 0, so res remains 0.0
        return res