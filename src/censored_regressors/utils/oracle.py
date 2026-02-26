import torch
import numpy as np


# --- 1. Test Function Registry ---
class TestFunctions:
    """
    Container for standard 1D/ND test functions and their configurations.
    """

    @staticmethod
    def fcn1(x):
        return 0.5 * torch.sin(3 * x)

    @staticmethod
    def fcn2(x):
        # f(x) = (6x - 2)^2 * sin(12x - 4)
        return torch.pow((6 * x - 2), 2) * torch.sin(2 * (6 * x - 2))

    @staticmethod
    def fcn3(x):
        # f(x) = x * sin(x)
        return x * torch.sin(x)

    @staticmethod
    def fcn4(x):
        # f(x) = 0.5*sin(x) - 0.02*(10-x)^2 + 2
        return 0.5 * torch.sin(x) - 0.02 * torch.pow((10 - x), 2) + 2

    @staticmethod
    def fcn_sine_wave(x):
        # The specific function: 0.5*sin(2x) + 2 + x/10
        return 0.5 * torch.sin(2 * x) + 2 + x / 10.0

    @staticmethod
    def fcn_linear(x, slope=1.0, intercept=1.0):
        # Simple Linear Model: y = mx + c
        return slope * x + intercept

    @staticmethod
    def fcn_linear2D(x, slope1=1.0, slope2=1.0, intercept=1.0):
        # Simple Linear Model: y = mx + c
        return slope1 * x[:, 0] + slope2 * x[:, 1] + intercept

    # Metadata dictionary mapping names to configs
    REGISTRY = {
        'fcn1': {
            'label': r"$f(x) = 0.5 \sin(3x)$",
            'fcn': fcn1,
            'start': 0.0, 'end': 5.0
        },
        'fcn2': {
            'label': r"$f(x) = (6x-2)^2 \sin(12x-4)$",
            'fcn': fcn2,
            'start': 0.0, 'end': 1.0
        },
        'fcn3': {
            'label': r"$f(x) = x \sin(x)$",
            'fcn': fcn3,
            'start': 0.0, 'end': 10.0
        },
        'fcn4': {
            'label': r"$f(x) = 0.5 \sin(x) - \dots$",
            'fcn': fcn4,
            'start': 0.0, 'end': 20.0
        },
        'sine_wave': {
            'label': r"$f(x) = 0.5 \sin(2x) + 2 + x/10$",
            'fcn': fcn_sine_wave,
            'start': 0.0, 'end': 10.0
        },
        'linear': {
            'label': r"$y = x + 1$",
            'fcn': fcn_linear,
            'start': -10.0, 'end': 10.0
        },
        'linear2D': {
            'label': r"$y = x_1 + x_2 + 1$",
            'fcn': fcn_linear2D,
            'start': -10.0, 'end': 10.0
        }
    }


# --- 2. Oracle Class ---
class Oracle:
    def __init__(self, fcn, seed=42):
        """
        Args:
            fcn (callable): Function mapping (N, dim) -> (N,)
            seed (int): Random seed
        """
        self.fcn = fcn
        self.seed = seed
        torch.manual_seed(self.seed)

    @classmethod
    def from_name(cls, name, seed=42):
        """
        Factory method to create an Oracle from a known test function name.
        """
        if name not in TestFunctions.REGISTRY:
            raise ValueError(f"Unknown function '{name}'. Available: {list(TestFunctions.REGISTRY.keys())}")

        config = TestFunctions.REGISTRY[name]
        print(f"Loading Oracle for: {config['label']}")
        return cls(fcn=config['fcn'], seed=seed)

    def evaluate_fcn(self, x):
        """Evaluates function and ensures output is 1D tensor (N,)"""
        y = self.fcn(x)
        return y.squeeze()

    def get_sample(self, N, dim=1, start=None, end=None, noise_scale=0.2, noisy=True):
        """
        Generates N samples.
        If start/end are None, defaults are -3.0/3.0.
        """
        torch.manual_seed(self.seed)

        # Simple default fallback
        s = start if start is not None else -3.0
        e = end if end is not None else 3.0

        # 1. Generate X
        dist = torch.distributions.Uniform(low=s, high=e)
        x = dist.sample(sample_shape=(N, dim))

        if dim == 1:
            x, _ = torch.sort(x, dim=0)

        # 2. Generate Y
        y_true = self.evaluate_fcn(x)

        if noisy:
            y_out = self.add_noise(y_true, noise_scale)
            return x, y_out, y_true
        else:
            return x, y_true, y_true

    def add_noise(self, y, noise_scale):
        if noise_scale <= 1e-9:
            return y
        torch.manual_seed(self.seed)
        noise = noise_scale * torch.randn(y.shape)
        return y + noise

    def censor(self, y, low=None, high=None, quantile=False):
        l_val, h_val = low, high

        if quantile:
            if low is not None: l_val = torch.quantile(y, low)
            if high is not None: h_val = torch.quantile(y, high)

        y_censored = y.clone()
        if l_val is not None:
            if not torch.is_tensor(l_val): l_val = torch.tensor(l_val)
            y_censored = torch.max(y_censored, l_val)

        if h_val is not None:
            if not torch.is_tensor(h_val): h_val = torch.tensor(h_val)
            y_censored = torch.min(y_censored, h_val)

        return y_censored


# --- 3. Generators (Refactored) ---

class OracleGenerator:
    """
    Base Generator.
    Now handles 'start' and 'end' in __init__ so that get_sample
    always receives the correct bounds and noise parameters.
    """

    def __init__(self, oracle, n_samples=100, dim=1, noise_scale=0.1,
                 censoring_low=None, censoring_high=None, quantile_censoring=False,
                 start=None, end=None):
        self.oracle = oracle
        self.n = n_samples
        self.dim = dim
        self.noise = noise_scale
        self.low = censoring_low
        self.high = censoring_high
        self.quantile = quantile_censoring
        self.start = start
        self.end = end

    def generate(self):
        # UNIFIED LOGIC: Pass start/end/noise explicitly to Oracle
        x, y_noisy, y_true = self.oracle.get_sample(
            N=self.n,
            dim=self.dim,
            noise_scale=self.noise,
            start=self.start,
            end=self.end
        )

        # Apply censoring
        y_obs = self.oracle.censor(y_noisy, self.low, self.high, self.quantile)

        # Convert to Numpy/Indicators
        y_noisy_np = y_noisy.detach().numpy()
        y_obs_np = y_obs.detach().numpy()
        c_np = np.zeros_like(y_obs_np, dtype=int)

        if self.high is not None: c_np[y_obs_np < y_noisy_np] = 1
        if self.low is not None: c_np[y_obs_np > y_noisy_np] = -1

        return (x.detach().numpy(), y_obs_np.reshape(-1, 1),
                c_np.reshape(-1, 1), y_true.detach().numpy().reshape(-1, 1))


class RangeBoundGenerator(OracleGenerator):
    """
    Since OracleGenerator now handles start/end logic,
    this class is mostly a convenience wrapper to preserve API compatibility.
    """

    def __init__(self, oracle, n_samples=200, start=-10, end=10, **kwargs):
        # We pass start and end up to the base class
        super().__init__(oracle, n_samples=n_samples, start=start, end=end, **kwargs)

    # No need to override generate() anymore; base class handles it correctly.


class VariableBoundGenerator(OracleGenerator):
    """
    Generator that supports User-Defined Boundaries (Vectors).
    """

    def __init__(self, oracle, n_samples=100, dim=1, noise_scale=0.1,
                 bounds_generator=None, start=None, end=None):

        # Pass start/end to base so x is generated in the correct range
        super().__init__(oracle, n_samples, dim, noise_scale, start=start, end=end)
        self.bounds_generator = bounds_generator

    def generate(self):
        # 1. Get Base Data (Uses self.start / self.end from base class)
        x_torch, y_noisy_torch, y_true_torch = self.oracle.get_sample(
            N=self.n,
            dim=self.dim,
            noise_scale=self.noise,
            start=self.start,
            end=self.end
        )

        # 2. Generate Variable Bounds
        if self.bounds_generator:
            def to_tensor(arr):
                if arr is None: return None
                return torch.as_tensor(arr, dtype=torch.float32).reshape(y_noisy_torch.shape)

            low_arr, high_arr = self.bounds_generator(x_torch, y_noisy_torch)
            low_t = to_tensor(low_arr)
            high_t = to_tensor(high_arr)
        else:
            low_t, high_t = None, None

        # 3. Apply Censoring manually
        y_obs_torch = y_noisy_torch.clone()

        if low_t is not None:
            y_obs_torch = torch.max(y_obs_torch, low_t)
        if high_t is not None:
            y_obs_torch = torch.min(y_obs_torch, high_t)

        # 4. Create Indicators
        y_noisy_np = y_noisy_torch.detach().numpy()
        y_obs_np = y_obs_torch.detach().numpy()
        c_np = np.zeros_like(y_obs_np, dtype=int)

        if high_t is not None:
            high_np = high_t.detach().numpy()
            c_np[y_noisy_np > high_np] = 1

        if low_t is not None:
            low_np = low_t.detach().numpy()
            c_np[y_noisy_np < low_np] = -1

        return (
            x_torch.detach().numpy(),
            y_obs_np.reshape(-1, 1),
            c_np.reshape(-1, 1),
            y_true_torch.detach().numpy().reshape(-1, 1)
        )


class ProbabilisticCensoredGenerator(OracleGenerator):
    """
    Generator with specific 'SineWave' censoring logic.
    Now respects noise_scale, start, and end args.
    """

    def __init__(self, oracle, n_samples=100, seed=42, noise_scale=0.1, start=None, end=None):
        # Pass start/end to base
        super().__init__(oracle, n_samples=n_samples, noise_scale=noise_scale,
                         censoring_low=None, censoring_high=None, start=start, end=end)
        self.seed = seed

    def generate(self):
        # 1. Get Base Data (Uses self.start / self.end / self.noise)
        x_torch, y_noisy_torch, y_true_torch = self.oracle.get_sample(
            N=self.n,
            dim=self.dim,
            noise_scale=self.noise,
            start=self.start,
            end=self.end
        )

        # Convert to Numpy
        x = x_torch.detach().numpy()
        y_obs = y_noisy_torch.detach().numpy().flatten()
        y_true = y_true_torch.detach().numpy().flatten()

        # 2. Apply Censoring Logic
        sine_component = 0.5 * np.sin(2 * x.flatten()) + 2
        censoring_mask = (sine_component >= 2)

        # 3. Apply Reduction
        n_censored = np.sum(censoring_mask)
        if n_censored > 0:
            np.random.seed(self.seed)
            p_c = np.random.uniform(low=0.2, high=0.3, size=n_censored)
            y_obs[censoring_mask] = y_obs[censoring_mask] * (1 - p_c)

        # 4. Return formatted
        C = np.zeros_like(y_obs, dtype=int)
        C[censoring_mask] = 1

        return (
            x.reshape(-1, 1),
            y_obs.reshape(-1, 1),
            C.reshape(-1, 1),
            y_true.reshape(-1, 1)
        )