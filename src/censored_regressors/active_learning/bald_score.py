import math
import torch
import numpy as np
from typing import Union, Tuple, Optional
from torch.distributions import Normal

# Ensure this import matches your package structure
try:
    from censored_regressors.distributions.censored_normal import CensoredNormal
except ImportError:
    # Fallback for local testing if package not installed
    from ..distributions.censored_normal import CensoredNormal

__all__=["CensoredBALD"]

class CensoredBALD:
    """
    Computes Bayesian Active Learning by Disagreement (BALD) scores for
    Censored Regression models.

    BALD = H[y | x] - E_f[ H[y | f, x] ]
    (Total Predictive Entropy - Expected Aleatoric Entropy)
    """

    def __init__(self, model, noise_std: float, low: float = None, high: float = None):
        """
        Args:
            model: A model object with a `predict(X)` method returning (mean, var) numpy arrays.
            noise_std: The standard deviation of the observation noise (sigma_y).
            low: Lower censoring bound. Use -inf or None for open.
            high: Upper censoring bound. Use +inf or None for open.
        """
        self.model = model
        self.noise_std = float(noise_std)
        self.noise_var = torch.tensor(self.noise_std ** 2)

        # Handle infinite bounds
        self.low = float(low) if low is not None else -float('inf')
        self.high = float(high) if high is not None else float('inf')

    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to get standardized Torch mean/var from the model."""
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X

        # Ensure 2D input for typical GP/sklearn models
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        mu_np, var_np = self.model.predict(X_np)

        # Convert to torch, flatten to 1D [N]
        mu = torch.from_numpy(mu_np).flatten().float()
        var = torch.from_numpy(var_np).flatten().float()

        return mu, var

    def _censored_entropy(self, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Wrapper to compute entropy of a Censored Normal distribution."""
        l_t = torch.tensor(self.low, device=loc.device, dtype=loc.dtype)
        u_t = torch.tensor(self.high, device=loc.device, dtype=loc.dtype)
        return CensoredNormal(loc, scale, l_t, u_t).entropy()

    def get_score(self, X: Union[torch.Tensor, np.ndarray], method: str = 'gauss_hermite', **kwargs) -> torch.Tensor:
        mean, var = self.predict(X)

        if method == 'gauss_hermite':
            return self._score_gauss_hermite(mean, var, **kwargs)
        elif method == 'monte_carlo':
            return self._score_monte_carlo(mean, var, **kwargs)
        elif method == 'houlsby':
            return self._score_houlsby(mean, var)
        elif method == 'gaussian':
            return self._score_gaussian_baseline(mean, var)
        else:
            raise ValueError(f"Unknown method: {method}")

    # --- Implementation Strategies ---

    def _score_gauss_hermite(self, mean_f, var_f, deg=30):
        # 1. Total Predictive Entropy H[y|x]
        total_std = (var_f + self.noise_var).sqrt()
        h_predictive = self._censored_entropy(mean_f, total_std)

        # 2. Expected Aleatoric Entropy E_f[ H[y|f] ]
        x_nodes, weights = np.polynomial.hermite.hermgauss(deg)
        x_nodes = torch.tensor(x_nodes, dtype=mean_f.dtype, device=mean_f.device)
        weights = torch.tensor(weights, dtype=mean_f.dtype, device=mean_f.device)
        scale_factor = 1.0 / math.sqrt(math.pi)

        f_std = var_f.sqrt()

        # f_samples: [N_nodes, N_data]
        f_samples = mean_f.unsqueeze(0) + f_std.unsqueeze(0) * math.sqrt(2) * x_nodes.unsqueeze(1)

        sigma_y = self.noise_var.sqrt()
        h_conditional = self._censored_entropy(f_samples, sigma_y)

        expected_h_conditional = scale_factor * torch.sum(weights.unsqueeze(1) * h_conditional, dim=0)

        # Clamp to 0 to prevent numerical noise causing negative mutual information
        return (h_predictive - expected_h_conditional).clamp(min=0.0)

    def _score_monte_carlo(self, mean_f, var_f, n_samples=5000):
        # 1. Total Entropy
        total_std = (var_f + self.noise_var).sqrt()
        h_predictive = self._censored_entropy(mean_f, total_std)

        # 2. Expected Conditional Entropy
        # FIX: Add jitter to variance to prevent crash when var_f=0 (zero uncertainty)
        # torch.distributions.Normal requires strictly positive scale.
        safe_std = (var_f + 1e-6).sqrt()

        f_dist = Normal(mean_f, safe_std)
        f_samples = f_dist.sample((n_samples,))

        sigma_y = self.noise_var.sqrt()
        h_conditional_samples = self._censored_entropy(f_samples, sigma_y)

        # Clamp to 0
        return (h_predictive - h_conditional_samples.mean(dim=0)).clamp(min=0.0)

    def _score_houlsby(self, mean_f, var_f):
        total_var = var_f + self.noise_var
        total_std = total_var.sqrt()

        l_t = torch.tensor(self.low, device=mean_f.device, dtype=mean_f.dtype)
        u_t = torch.tensor(self.high, device=mean_f.device, dtype=mean_f.dtype)
        norm = Normal(torch.tensor(0.0, device=mean_f.device), torch.tensor(1.0, device=mean_f.device))

        # --- Term 1 ---
        alpha = (l_t - mean_f) / total_std
        beta = (u_t - mean_f) / total_std

        prob_uncensored = torch.ones_like(mean_f)
        if self.low > -float('inf'): prob_uncensored -= norm.cdf(alpha)
        if self.high < float('inf'): prob_uncensored -= (1.0 - norm.cdf(beta))

        prob_uncensored = torch.clamp(prob_uncensored, min=1e-7)
        term1 = 0.5 * torch.log(1 + var_f / self.noise_var) * prob_uncensored

        # --- Term 2 ---
        asym = torch.zeros_like(mean_f)
        if self.high < float('inf'): asym += beta * norm.log_prob(beta).exp()
        if self.low > -float('inf'): asym -= alpha * norm.log_prob(alpha).exp()

        term2 = -0.5 * (var_f / total_var) * asym

        # --- Term 3 ---
        A, MU, SIG = 0.3676, -0.3339, 0.9794
        sigma_y = self.noise_var.sqrt()
        std_z = var_f.sqrt() / sigma_y

        def approx_integral(boundary_val, sign_flip=1.0):
            mean_z = (boundary_val - mean_f) / sigma_y * sign_flip
            var_conv = std_z.pow(2) + SIG ** 2
            scale = A * (SIG / var_conv.sqrt())
            exponent = -0.5 * (mean_z - MU).pow(2) / var_conv
            return -scale * torch.exp(exponent)

        term3 = torch.zeros_like(mean_f)
        if self.low > -float('inf'): term3 += approx_integral(l_t, sign_flip=1.0)
        if self.high < float('inf'): term3 += approx_integral(u_t, sign_flip=-1.0)

        # FIX: Clamp result to 0.0. Approximations can drift slightly negative.
        return (term1 + term2 + term3).clamp(min=0.0)

    def _score_gaussian_baseline(self, mean_f, var_f):
        return 0.5 * torch.log(1 + var_f / self.noise_var)