import math
import torch
import numpy as np
from numbers import Number
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

__all__ = ["CensoredNormal"]


class CensoredNormal(ExponentialFamily):
    """
    Creates a censored normal distribution parameterized by loc, scale, low, and high.

    Updated to match 'Robust Tobit Loss' numerical stability:
    1. Uses torch.special.log_ndtr for precise tail log-probabilities.
    2. Clamps Z-scores to +/- 50 to prevent gradient explosion.
    """
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0)
    }
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, loc, scale, low, high, validate_args=None):
        self.loc, self.scale, self.low, self.high = broadcast_all(loc, scale, low, high)

        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(low, Number) and isinstance(high,
                                                                                                            Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()

        # Jitter is no longer needed for log_prob with log_ndtr,
        # but kept small for entropy/division safety if needed elsewhere.
        self.jitter_ = 1e-16
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CensoredNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(CensoredNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def log_prob(self, value):
        """
        Robust Tobit-style Log Probability.
        """
        if self._validate_args:
            self._validate_sample(value)

        # 1. Standardize and Clamp (Gradient Safety)
        # We clamp Z to +/- 50.0 to prevent floating point explosion, matching Tobit Loss.
        val_z = (value - self.loc) / self.scale
        val_z = torch.clamp(val_z, min=-50., max=50.)

        # 2. Standardize and Clamp Boundaries
        # We MUST clamp these too. If low = -inf, z_low becomes -inf.
        # PyTorch evaluates gradients for all branches of torch.where.
        # Gradients of log_ndtr(-inf) can produce NaNs that pollute the output.
        z_low = (self.low - self.loc) / self.scale
        z_low = torch.clamp(z_low, min=-50., max=50.)

        z_high = (self.high - self.loc) / self.scale
        z_high = torch.clamp(z_high, min=-50., max=50.)

        # 2. Continuous Case: Standard Gaussian Log PDF
        # log( 1/sigma * exp(-0.5*z^2) / sqrt(2pi) )
        log_prob_cont = -0.5 * (math.log(2 * math.pi) + val_z ** 2) - self.scale.log()

        # 3. Left Censored Case: log( P(X <= low) )
        # log( CDF(z_low) ) -> uses stable log_ndtr
        log_prob_low = torch.special.log_ndtr(z_low)

        # 4. Right Censored Case: log( P(X >= high) )
        # log( 1 - CDF(z_high) ) = log( CDF(-z_high) ) -> uses stable log_ndtr
        log_prob_high = torch.special.log_ndtr(-z_high)

        # 5. Select based on value position
        # If value matches boundary, take the boundary mass. Otherwise take PDF.
        result = log_prob_cont
        result = torch.where(value <= self.low, log_prob_low, result)
        result = torch.where(value >= self.high, log_prob_high, result)

        return result

    @property
    def mean(self):
        """Analytical Mean of Censored Normal"""
        alpha = (self.low - self.loc) / self.scale
        beta = (self.high - self.loc) / self.scale

        # Use stable CDF (ndtr) instead of erf
        Phi_l = torch.special.ndtr(alpha)
        Phi_u = torch.special.ndtr(beta)

        phi_l = torch.exp(-0.5 * alpha ** 2) / math.sqrt(2 * math.pi)
        phi_u = torch.exp(-0.5 * beta ** 2) / math.sqrt(2 * math.pi)

        # Handle Infs safely
        phi_l = torch.where(torch.isinf(self.low), torch.tensor(0., device=self.loc.device), phi_l)
        phi_u = torch.where(torch.isinf(self.high), torch.tensor(0., device=self.loc.device), phi_u)

        term_center = self.loc * (Phi_u - Phi_l) - self.scale * (phi_u - phi_l)
        term_l = torch.where(torch.isinf(self.low), torch.tensor(0., device=self.loc.device), self.low * Phi_l)
        term_u = torch.where(torch.isinf(self.high), torch.tensor(0., device=self.loc.device), self.high * (1 - Phi_u))

        return term_center + term_l + term_u

    @property
    def variance(self):
        """Analytical Variance of Censored Normal"""
        alpha = (self.low - self.loc) / self.scale
        beta = (self.high - self.loc) / self.scale

        Phi_l = torch.special.ndtr(alpha)
        Phi_u = torch.special.ndtr(beta)

        phi_l = torch.exp(-0.5 * alpha ** 2) / math.sqrt(2 * math.pi)
        phi_u = torch.exp(-0.5 * beta ** 2) / math.sqrt(2 * math.pi)

        phi_l = torch.where(torch.isinf(self.low), torch.tensor(0., device=self.loc.device), phi_l)
        phi_u = torch.where(torch.isinf(self.high), torch.tensor(0., device=self.loc.device), phi_u)

        term1 = (self.loc ** 2 + self.scale ** 2) * (Phi_u - Phi_l)
        term2 = -2 * self.loc * self.scale * (phi_u - phi_l)

        z_phi_l = torch.where(torch.isinf(self.low), torch.tensor(0., device=self.loc.device), alpha * phi_l)
        z_phi_u = torch.where(torch.isinf(self.high), torch.tensor(0., device=self.loc.device), beta * phi_u)
        term3 = -(self.scale ** 2) * (z_phi_u - z_phi_l)

        e_x2_uncensored = term1 + term2 + term3

        term4_l = torch.where(torch.isinf(self.low), torch.tensor(0., device=self.loc.device), self.low ** 2 * Phi_l)
        term4_u = torch.where(torch.isinf(self.high), torch.tensor(0., device=self.loc.device),
                              self.high ** 2 * (1 - Phi_u))

        e_x2 = e_x2_uncensored + term4_l + term4_u
        return e_x2 - self.mean.pow(2)

    @property
    def stddev(self):
        return self.variance.sqrt()

    def entropy(self):
        """Closed form Entropy"""
        alpha = (self.low - self.loc) / self.scale
        beta = (self.high - self.loc) / self.scale

        Phi_l = torch.special.ndtr(alpha)
        Phi_u = torch.special.ndtr(beta)

        phi_l = torch.exp(-0.5 * alpha ** 2) / math.sqrt(2 * math.pi)
        phi_u = torch.exp(-0.5 * beta ** 2) / math.sqrt(2 * math.pi)

        phi_l = torch.where(torch.isinf(self.low), torch.tensor(0., device=self.loc.device), phi_l)
        phi_u = torch.where(torch.isinf(self.high), torch.tensor(0., device=self.loc.device), phi_u)

        h_normal = 0.5 * math.log(2 * math.pi * math.e) + self.scale.log()

        term1 = h_normal * (Phi_u - Phi_l)
        term2 = 0.5 * (beta * phi_u - alpha * phi_l)
        term2 = torch.where(torch.isinf(self.high) | torch.isinf(self.low), torch.nan_to_num(term2), term2)

        def safe_log(x):
            return torch.where(x < self.jitter_, torch.log(x + self.jitter_), x.log())

        term3_l = torch.where(torch.isinf(self.low), torch.tensor(0., device=self.loc.device), Phi_l * safe_log(Phi_l))
        term3_u = torch.where(torch.isinf(self.high), torch.tensor(0., device=self.loc.device),
                              (1 - Phi_u) * safe_log(1 - Phi_u))

        return term1 - term2 - (term3_l + term3_u)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            samples = torch.normal(self.loc.expand(shape), self.scale.expand(shape))
            return samples.clamp(min=self.low, max=self.high)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        rsamples = self.loc + eps * self.scale
        return rsamples.clamp(min=self.low, max=self.high)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # Use stable ndtr
        cdf_ = torch.special.ndtr((value - self.loc) / self.scale)
        cdf_ = torch.where(value < self.low, torch.tensor(0., device=value.device), cdf_)
        cdf_ = torch.where(value > self.high, torch.tensor(1., device=value.device), cdf_)
        return cdf_

    def icdf(self, value):
        result = self._normal_icdf_standardized(value) * self.scale + self.loc
        return result.clamp(min=self.low, max=self.high)

    # --- Helpers ---
    def _normal_log_prob_standardized(self, value):
        return -0.5 * (value ** 2 + math.log(2 * math.pi))

    def _normal_cdf_standardized(self, value):
        # Updated to use stable ndtr
        return torch.special.ndtr(value)

    def _normal_icdf_standardized(self, value):
        return torch.erfinv(2 * value - 1) * math.sqrt(2)