import math
from numbers import Number, Real

import torch

from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from censored_regressors.distributions.censored_normal import CensoredNormal
from torch.distributions.utils import _standard_normal, broadcast_all

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape

__all__ = ["PyroCensoredNormal"]

class PyroCensoredNormal(CensoredNormal, TorchDistribution):
    def __init__(self, loc, scale, low, high, jitter=1e-16, validate_args=None, **kwargs):
        self._unbroadcasted_loc = loc
        self._unbroadcasted_scale = scale
        self._unbroadcasted_low = low
        self._unbroadcasted_high = high
        self._unbroadcasted_jitter = jitter
        self.jitter = jitter
        super().__init__(loc, scale, low, high, validate_args=validate_args)
        self.jitter_ = self.jitter

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        new = super().expand(batch_shape, _instance=new)
        new._unbroadcasted_loc = self._unbroadcasted_loc
        new._unbroadcasted_scale = self._unbroadcasted_scale
        new._unbroadcasted_low = self._unbroadcasted_low
        new._unbroadcasted_high = self._unbroadcasted_high
        new._unbroadcasted_jitter = self._unbroadcasted_jitter
        new.jitter = new._unbroadcasted_jitter
        new.jitter_ = new.jitter

        return new


    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        # FIX 2: Removed self.jitter_ from arguments.
        # constraints.interval only takes (lower, upper).
        return constraints.interval(self.low, self.high)
