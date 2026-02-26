import torch
import torch.distributions as dist
import numpy as np
import math
from scipy.integrate import quad
from scipy.stats import norm
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise

# ==========================================
# 1. REFERENCE IMPLEMENTATION (SciPy QUAD)
#    - "Ground Truth" (very slow, high precision)
# ==========================================
SQRT_2 = math.sqrt(2.0)
SQRT_PI = math.sqrt(math.pi)


class ReferenceLikelihood(Likelihood):
    def __init__(self, variance=1.0, low=-1e10, high=1e10):
        super().__init__()
        self.noise = HomoskedasticNoise()
        self.noise.initialize(noise=torch.tensor(variance))
        self.low = low
        self.high = high

    def forward(self, function_samples, *params, **kwargs):
        return dist.Normal(function_samples, self.noise.noise.sqrt())

    def expected_log_prob(self, target, input, *params, **kwargs):
        # We focus on the UPPER censored part for this test
        # E[log Phi( (mean - high)/sigma + (std/sigma)*z )]
        mean, variance = input.mean, input.variance
        noise = self.noise.noise
        sigma = noise.sqrt()
        std = variance.sqrt()

        a = (mean - self.high).item() / sigma.item()
        b = std.item() / sigma.item()

        # Direct integration of e^{-z^2} * log_phi(a + bz)
        def integrand(z):
            # z is standard normal variable
            val = a + b * z
            # scipy.stats.norm.logcdf is robust
            log_phi = norm.logcdf(val)
            weight = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            return log_phi * weight

        # Integrate from -15 to +15 sigma (covering 99.999...% of mass)
        val, err = quad(integrand, -20, 20)
        return torch.tensor(val)


# ==========================================
# 2. ROBUST IMPLEMENTATION (Gauss-Hermite)
#    - The optimized code we wrote
# ==========================================
class RobustLikelihood(Likelihood):
    def __init__(self, variance=1.0, low=-1e10, high=1e10, quad_points=30):
        super().__init__()
        self.noise = HomoskedasticNoise()
        self.noise.initialize(noise=torch.tensor(variance))
        self.register_buffer("high", torch.tensor(high).float())

        gh_x, gh_w = np.polynomial.hermite.hermgauss(quad_points)
        self.register_buffer("gh_x", torch.from_numpy(gh_x).double())
        self.register_buffer("gh_w", torch.from_numpy(gh_w).double())

    def forward(self, function_samples, *params, **kwargs):
        return dist.Normal(function_samples, self.noise.noise.sqrt())

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        sigma = self.noise.noise.sqrt()
        std = variance.sqrt()

        a = (mean - self.high) / sigma
        b = std / sigma

        return self._integrate_log_phi(a, b)

    def _integrate_log_phi(self, a, b):
        # Upcast to double for precision
        a_d, b_d = a.double(), b.double()
        nodes = self.gh_x
        z = nodes * SQRT_2

        arg = a_d + b_d * z
        log_phi_vals = self._log_phi_robust(arg)

        # Gauss-Hermite sum
        return (log_phi_vals * self.gh_w).sum() / SQRT_PI

    def _log_phi_robust(self, z):
        # 1. Safe region
        mask_safe = (z > -6) & (z < 10)
        mask_neg = z <= -6

        res = torch.zeros_like(z)

        if mask_safe.any():
            # log(0.5 * erfc(-z/sqrt(2)))
            res[mask_safe] = torch.log(0.5 * torch.special.erfc(-z[mask_safe] / SQRT_2).clamp(min=1e-100))

        if mask_neg.any():
            # Asymptotic expansion: -z^2/2 - log|z| - 0.5log(2pi)
            v = z[mask_neg]
            log_2pi = math.log(2 * math.pi)
            res[mask_neg] = -0.5 * v.pow(2) - torch.log(-v) - 0.5 * log_2pi

        return res


# ==========================================
# 3. DIAGNOSTIC RUNNER
# ==========================================
def run_diagnostics():
    print(f"{'CASE DESCRIPTION':<35} | {'REF (SciPy)':<15} | {'NEW (Robust)':<15} | {'DIFF':<10} | {'STATUS'}")
    print("-" * 95)

    cases = [
        # 1. The "Easy" Case
        {"name": "Normal Range (Mean=0, High=1)",
         "mean": 0.0, "var": 1.0, "high": 1.0, "noise": 1.0},

        # 2. The "Borderline" Case
        {"name": "Boundary (Mean=High)",
         "mean": 1.0, "var": 1.0, "high": 1.0, "noise": 1.0},

        # 3. The "Censored" Case (Data is likely censored)
        {"name": "Deep Censored (Mean >> High)",
         "mean": 5.0, "var": 1.0, "high": 1.0, "noise": 1.0},

        # 4. The "Impossible" Case (Extreme Tail)
        # This usually breaks non-robust implementations (returns -inf or NaN)
        {"name": "Extreme Tail (Mean=20, High=0)",
         "mean": 20.0, "var": 1.0, "high": 0.0, "noise": 1.0},

        # 5. The "Tiny Uncertainty" Case (Approaching Dirac Delta)
        # Integration grid must be very fine or adaptive
        {"name": "Tiny Variance (Var=1e-6)",
         "mean": 2.0, "var": 1e-6, "high": 1.0, "noise": 0.1},

        # 6. The "High Uncertainty" Case
        # Integration must cover a wide range
        {"name": "Huge Variance (Var=100)",
         "mean": 0.0, "var": 100.0, "high": 1.0, "noise": 1.0},
    ]

    for case in cases:
        # Setup Inputs
        mean = torch.tensor(float(case['mean']))
        var = torch.tensor(float(case['var']))
        high = float(case['high'])
        noise = float(case['noise'])

        # Instantiate Models
        ref_model = ReferenceLikelihood(variance=noise, high=high)
        new_model = RobustLikelihood(variance=noise, high=high)

        distrib = dist.MultivariateNormal(mean.unsqueeze(0), torch.diag(var.unsqueeze(0)))
        target = torch.tensor([0.0])  # Dummy target

        # Run Forward
        with torch.no_grad():
            try:
                val_ref = ref_model.expected_log_prob(target, distrib).item()
            except Exception as e:
                val_ref = float('nan')

            val_new = new_model.expected_log_prob(target, distrib).item()

        # Check Diff
        diff = abs(val_ref - val_new)

        # Status Logic
        if math.isnan(val_new) or math.isinf(val_new):
            status = "CRASH ❌"
        elif math.isnan(val_ref):
            status = "REF FAIL ⚠️"  # Reference implementation failed
        elif diff < 1e-3:
            status = "PASS ✅"
        elif diff < 1e-1:
            status = "OK 🔸"
        else:
            status = "FAIL ❌"

        print(f"{case['name']:<35} | {val_ref:<15.4f} | {val_new:<15.4f} | {diff:<10.4f} | {status}")


if __name__ == "__main__":
    run_diagnostics()