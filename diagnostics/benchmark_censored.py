import time
import math
import torch
import torch.distributions as dist
import numpy as np
from scipy.integrate import quad
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise

# ==========================================
# 1. OLD IMPLEMENTATION (User's Code)
# ==========================================
SQRT_2PI = math.sqrt(2 * math.pi)


class ScipyExpectation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma, utility_func):
        ctx.save_for_backward(mu, sigma)
        ctx.utility_func = utility_func
        mu_val, sigma_val = mu.item(), sigma.item()
        lower, upper = mu_val - 12 * abs(sigma_val), mu_val + 12 * abs(sigma_val)

        def f_pdf(x):
            pdf = (1.0 / (abs(sigma_val) * SQRT_2PI)) * np.exp(-0.5 * ((x - mu_val) / sigma_val) ** 2)
            return utility_func(x) * pdf

        result, _ = quad(f_pdf, lower, upper)
        return torch.tensor(result, dtype=torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        # Simplified backward to avoid crashing during benchmark
        return grad_output, grad_output, None


def normal_log_cdf_scalar(x):
    from scipy.stats import norm
    return norm.logcdf(x if isinstance(x, (float, int)) else x.item())


class OldLikelihood(Likelihood):
    def __init__(self, variance=1.0, low=-1.0, high=1.0):
        super().__init__()
        self.noise = HomoskedasticNoise()
        self.noise.initialize(noise=torch.tensor(variance))
        self.low = low
        self.high = high

    # --- FIX: Added required forward method ---
    def forward(self, function_samples, *params, **kwargs):
        return dist.Normal(function_samples, self.noise.noise.sqrt())

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise.noise
        sigma = noise.sqrt()
        std = variance.sqrt()

        a_upper = (mean - self.high) / sigma
        b_upper = std / sigma

        # CPU-bound loop simulation
        a_flat = a_upper.view(-1)
        b_flat = b_upper.view(-1)

        results = [ScipyExpectation.apply(a_flat[i], b_flat[i], normal_log_cdf_scalar)
                   for i in range(len(a_flat))]

        res = torch.stack(results).view_as(target).float()
        return res


# ==========================================
# 2. NEW IMPLEMENTATION (Optimized)
# ==========================================
class NewLikelihood(Likelihood):
    def __init__(self, variance=1.0, low=-1.0, high=1.0, quad_points=20):
        super().__init__()
        self.noise = HomoskedasticNoise()
        self.noise.initialize(noise=torch.tensor(variance))
        self.register_buffer("low", torch.tensor(low).float())
        self.register_buffer("high", torch.tensor(high).float())

        gh_x, gh_w = np.polynomial.hermite.hermgauss(quad_points)
        self.register_buffer("gh_x", torch.from_numpy(gh_x).double())
        self.register_buffer("gh_w", torch.from_numpy(gh_w).double())

    # --- FIX: Added required forward method ---
    def forward(self, function_samples, *params, **kwargs):
        return dist.Normal(function_samples, self.noise.noise.sqrt())

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise.noise
        sigma = noise.sqrt()
        std = variance.sqrt()

        a_upper = (mean - self.high) / sigma
        b = std / sigma

        return self._integrate_log_phi(a_upper, b)

    def _integrate_log_phi(self, a, b):
        a_d, b_d = a.double(), b.double()
        nodes = self.gh_x.unsqueeze(0)
        z = nodes * math.sqrt(2.0)
        arg = a_d.unsqueeze(-1) + b_d.unsqueeze(-1) * z

        # Simplified robust log_phi for benchmark
        val = arg
        # log(0.5 * erfc(-z/sqrt(2)))
        # We use a clamp to prevent -inf
        log_phi = torch.log(0.5 * torch.special.erfc(-val / math.sqrt(2.0)).clamp(min=1e-100))

        return (log_phi * self.gh_w).sum(-1).float() / math.sqrt(math.pi)


# ==========================================
# 3. BENCHMARK SUITE
# ==========================================
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Benchmarking batch sizes
    BATCH_SIZES = [10, 50, 100]  # Keep small for CPU/Old model sanity
    if device.type == 'cuda':
        BATCH_SIZES.append(1000)

    print(f"{'Batch':<10} | {'Method':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Speedup':<10}")
    print("-" * 65)

    for B in BATCH_SIZES:
        # Data
        target = torch.randn(B, device=device)
        mean = torch.randn(B, requires_grad=True, device=device)
        cov = torch.eye(B, device=device) * 0.5
        distrib = dist.MultivariateNormal(mean, cov)

        # --- OLD MODEL (Force CPU for SciPy compatibility) ---
        old_model = OldLikelihood().cpu()
        target_cpu = target.cpu()
        distrib_cpu = dist.MultivariateNormal(mean.cpu(), cov.cpu())

        # Timing Old
        start = time.time()
        loss_old = old_model.expected_log_prob(target_cpu, distrib_cpu).sum()
        fwd_old = (time.time() - start) * 1000

        # Only backward if batch is small (SciPy backward is insanely slow)
        if B <= 100:
            start = time.time()
            loss_old.backward()
            bwd_old = (time.time() - start) * 1000
        else:
            bwd_old = 99999.0  # Skip backward for large batch

        # --- NEW MODEL ---
        new_model = NewLikelihood().to(device)

        # Warmup
        if B > 10:
            _ = new_model.expected_log_prob(target, distrib).sum()

        # Reset Grads
        mean.grad = None
        distrib = dist.MultivariateNormal(mean, cov)

        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.time()
        loss_new = new_model.expected_log_prob(target, distrib).sum()
        if device.type == 'cuda': torch.cuda.synchronize()
        fwd_new = (time.time() - start) * 1000

        start = time.time()
        loss_new.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        bwd_new = (time.time() - start) * 1000

        total_old = fwd_old + bwd_old
        total_new = fwd_new + bwd_new
        speedup = total_old / total_new if B <= 100 else fwd_old / fwd_new

        print(f"{B:<10} | {'OLD':<10} | {fwd_old:<10.2f} | {bwd_old:<10.2f} | {'1.0x':<10}")
        print(f"{B:<10} | {'NEW':<10} | {fwd_new:<10.2f} | {bwd_new:<10.2f} | {speedup:<10.1f}x")
        print("-" * 65)


if __name__ == "__main__":
    run_benchmark()