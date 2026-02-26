import torch
import numpy as np
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.nn.module import PyroParam

# Ensure this import matches your package structure
from censored_regressors.distributions.censored_normal_pyro import PyroCensoredNormal


class VariationalCensoredNormal(dist.TorchDistribution):
    # FIX 3: Define arg_constraints to satisfy Pyro/PyTorch validation requirements
    arg_constraints = {}

    def __init__(self, f_loc, f_var, obs_scale, low, high, num_quad_points=20):
        self.f_loc = f_loc
        self.f_var = f_var
        self.obs_scale = obs_scale
        self.low = low
        self.high = high

        gh_x, gh_w = np.polynomial.hermite.hermgauss(num_quad_points)
        self.gh_x = torch.tensor(gh_x, dtype=f_loc.dtype, device=f_loc.device)
        self.gh_w = torch.tensor(gh_w, dtype=f_loc.dtype, device=f_loc.device)
        self.inv_sqrt_pi = 1.0 / np.sqrt(np.pi)

        batch_shape = self.f_loc.shape
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size())

    def log_prob(self, value):
        f_std = self.f_var.sqrt()

        nodes = self.gh_x.view(-1, *([1] * len(self.f_loc.shape)))
        weights = self.gh_w.view(-1, *([1] * len(self.f_loc.shape)))

        f_samples = self.f_loc + f_std * nodes * np.sqrt(2.0)

        d = PyroCensoredNormal(
            loc=f_samples,
            scale=self.obs_scale,
            low=self.low,
            high=self.high
        )

        value_expanded = value.unsqueeze(0)
        log_probs = d.log_prob(value_expanded)

        expected_log_prob = self.inv_sqrt_pi * torch.sum(weights * log_probs, dim=0)
        return expected_log_prob

    def sample(self, sample_shape=torch.Size()):
        total_scale = (self.f_var + self.obs_scale.pow(2)).sqrt()
        d = PyroCensoredNormal(self.f_loc, total_scale, self.low, self.high)
        return d.sample(sample_shape)


class CensoredHomoscedGaussian(Likelihood):
    def __init__(self, variance=None, low=None, high=None, num_quad_points=20):
        super().__init__()

        # FIX 1: Ensure variance is a Tensor before passing to PyroParam
        if variance is None:
            variance = torch.tensor(1.)
        else:
            variance = torch.as_tensor(variance, dtype=torch.float32)

        self.variance = PyroParam(variance, constraints.positive)

        self.register_buffer("low", torch.as_tensor(low) if low is not None else torch.tensor(-float('inf')))
        self.register_buffer("high", torch.as_tensor(high) if high is not None else torch.tensor(float('inf')))

        self.num_quad_points = num_quad_points

    def forward(self, f_loc, f_var, y=None):
        y_dist = VariationalCensoredNormal(
            f_loc=f_loc,
            f_var=f_var,
            obs_scale=self.variance.sqrt(),
            low=self.low,
            high=self.high,
            num_quad_points=self.num_quad_points
        )

        if y is not None:
            if y.shape != f_loc.shape:
                y_dist = y_dist.expand(y.shape)

        return pyro.sample("y", y_dist, obs=y)