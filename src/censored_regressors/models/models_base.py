import abc
import copy
import warnings
import numpy as np
import tqdm
from typing import Dict, Any, Tuple, Union, Optional

# --- Third Party Imports ---
import torch
import gpytorch
import GPy

# --- GPyTorch Components ---
from gpytorch.mlls import VariationalELBO
from gpytorch.optim import NGD
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, MaternKernel
from gpytorch.constraints import GreaterThan


class RegressionMethod(abc.ABC):
    """
    Abstract base class for regression methods.

    Responsibilities:
    1. Data Preprocessing (Z-score normalization).
    2. Prediction Scaling (Transforming predictions back to original scale).
    3. Interface definition (fit/predict).
    """

    def __init__(self):
        self.preprocess = False
        self.data_mean = 0.0
        self.data_std = 1.0
        self.labels_mean = 0.0
        self.labels_std = 1.0

    def _preprocess(self, data, train: bool = True):
        """
        Zero-mean, unit-variance normalization.
        Handles both (X, Y) and (X, Y, Censoring) tuples.
        """
        if train:
            # Unpack data
            if len(data) == 3:
                inputs, labels, censoring = data
            else:
                inputs, labels = data
                censoring = None

            # Calculate stats
            self.data_mean = inputs.mean(axis=0)
            self.data_std = inputs.std(axis=0) + 1e-6  # Epsilon for stability
            self.labels_mean = labels.mean(axis=0)
            self.labels_std = labels.std(axis=0) + 1e-6

            # Normalize
            X_norm = (inputs - self.data_mean) / self.data_std
            Y_norm = (labels - self.labels_mean) / self.labels_std

            if censoring is not None:
                return (X_norm, Y_norm, censoring)
            return (X_norm, Y_norm)
        else:
            # Test data only has X
            return (data - self.data_mean) / self.data_std

    def _reverse_trans_labels(self, results):
        """
        Rescales predictions back to original space.
        """
        # Dictionary format (Preferred)
        if isinstance(results, dict):
            scaled_results = {}
            for key, val in results.items():
                if key in ['f_mean', 'f_025', 'f_975', 'y_mean']:
                    # Linear scaling: x_scaled * std + mean
                    scaled_results[key] = val * self.labels_std + self.labels_mean
                elif key == 'f_var':
                    # Variance scaling: var * std^2
                    scaled_results[key] = val * (self.labels_std ** 2)
                else:
                    scaled_results[key] = val
            return scaled_results

        # Legacy Tuple format (mean, var)
        mean, var = results
        mean_orig = mean * self.labels_std + self.labels_mean
        var_orig = var * (self.labels_std ** 2)
        return mean_orig, var_orig

    def _compute_sample_stats(self, samples: np.ndarray, axis: int) -> Dict[str, np.ndarray]:
        """
        Shared logic to compute mean, variance and CIs from posterior samples.
        """
        f_mean = np.mean(samples, axis=axis)
        f_var = np.var(samples, axis=axis)
        f_025 = np.quantile(samples, 0.025, axis=axis)
        f_975 = np.quantile(samples, 0.975, axis=axis)

        # Reshape to ensure (N, 1) output format
        return {
            'f_mean': f_mean.reshape(-1, 1),
            'f_var': f_var.reshape(-1, 1),
            'f_025': f_025.reshape(-1, 1),
            'f_975': f_975.reshape(-1, 1)
        }

    def fit(self, train_data, **kwargs):
        """Public fit method with preprocessing hook."""
        if self.preprocess:
            train_data = self._preprocess(train_data, True)
        return self._fit(train_data, **kwargs)

    def predict(self, test_data):
        """Public predict method with preprocessing hook."""
        if self.preprocess:
            test_data_proc = self._preprocess(test_data, False)
        else:
            test_data_proc = test_data

        results = self._predict(test_data_proc)

        if self.preprocess:
            results = self._reverse_trans_labels(results)

        return results

    @abc.abstractmethod
    def _fit(self, train_data, **kwargs):
        pass

    @abc.abstractmethod
    def _predict(self, test_data):
        pass


class BaseGPyModel(RegressionMethod):
    """
    Base class for GPy-backed models.
    """

    def __init__(self, kernel_type='lin_rbf', name='BaseGP'):
        super().__init__()
        self.kernel_type = kernel_type
        self.model = None
        self.name = name

    def _get_kernel(self, input_dim):
        """Factory for GPy kernels."""
        kern_dict = {
            'lin': lambda: GPy.kern.Linear(input_dim, ARD=True),
            'rbf': lambda: GPy.kern.RBF(input_dim, ARD=True),
            'lin_rbf': lambda: GPy.kern.RBF(input_dim, ARD=True) + GPy.kern.Linear(input_dim, ARD=True),
            'matern32': lambda: GPy.kern.Matern32(input_dim, ARD=True),
            'lin_matern32': lambda: GPy.kern.Matern32(input_dim, ARD=True) + GPy.kern.Linear(input_dim, ARD=True),
            'matern52': lambda: GPy.kern.Matern52(input_dim, ARD=True),
            'lin_matern52': lambda: GPy.kern.Matern52(input_dim, ARD=True) + GPy.kern.Linear(input_dim, ARD=True)
        }

        if self.kernel_type not in kern_dict:
            raise ValueError(f"Kernel type '{self.kernel_type}' not found.")

        return kern_dict[self.kernel_type]()

    def _apply_init_params(self, params: Dict[str, Any]):
        """Applies user initialization to GPy models."""
        if self.model is None or not params:
            return

        print(f"  Applying user initialization: {params}")

        # 1. Initialize Likelihood Variance
        lik_var = params.get('noise') or params.get('variance') or params.get('likelihood_variance')
        if lik_var is not None and hasattr(self.model.likelihood, 'variance'):
            self.model.likelihood.variance[:] = lik_var

        # 2. Initialize Kernel Parameters
        def set_kern_params(kern):
            if hasattr(kern, 'parts'):
                for part in kern.parts:
                    set_kern_params(part)
            else:
                if 'lengthscale' in params and hasattr(kern, 'lengthscale'):
                    kern.lengthscale[:] = params['lengthscale']

                sig_var = params.get('outputscale') or params.get('signal_variance')
                if sig_var is not None and hasattr(kern, 'variance'):
                    kern.variance[:] = sig_var

        if hasattr(self.model, 'kern'):
            set_kern_params(self.model.kern)

    def _apply_constraints(self):
        """Applies stability constraints to avoid numerical issues."""
        if self.model is None: return

        # Constrain Lengthscales
        if hasattr(self.model.kern, 'lengthscale'):
            self.model.kern.lengthscale.constrain_bounded(1e-2, 2e2)
        elif hasattr(self.model.kern, 'parts'):
            for p in self.model.kern.parts:
                if hasattr(p, 'lengthscale'):
                    p.lengthscale.constrain_bounded(1e-2, 2e2)

        # Constrain Likelihood Variance
        if hasattr(self.model.likelihood, 'variance'):
            self.model.likelihood.variance.constrain_bounded(1e-5, 1e2)

    def _predict(self, test_data):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)

        # GPy Shape: (N_points, Output_Dim, N_samples) -> e.g. (200, 1, 1000)
        # OR (N_points, N_samples) depending on model type
        f_samples = self.model.posterior_samples_f(test_data, size=1000)

        # Determine axis
        if f_samples.ndim == 3:
            sample_axis = 2
        elif f_samples.ndim == 2:
            sample_axis = 1
        else:
            sample_axis = 0

        return self._compute_sample_stats(f_samples, axis=sample_axis)


import abc
import copy
import warnings
import numpy as np
import tqdm
from typing import Dict, Any, Tuple, Union, Optional

# --- Third Party Imports ---
import torch
import gpytorch
import GPy

# --- GPyTorch Components ---
from gpytorch.mlls import VariationalELBO
from gpytorch.optim import NGD
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, MaternKernel
from gpytorch.constraints import GreaterThan


# ... [RegressionMethod and BaseGPyModel classes remain unchanged] ...

class BaseGPyTorchModel(RegressionMethod):
    """
    Base class for GPyTorch-backed models.
    """

    def __init__(self, kernel_type='lin_rbf', name='BaseGPyTorch'):
        super().__init__()
        self.kernel_type = kernel_type
        self.name = name
        self.model = None
        self.likelihood = None

    def _get_kernel(self, input_dim):
        """Factory for GPyTorch kernels."""

        # [Same as before]
        def make_rbf():
            k = RBFKernel(ard_num_dims=input_dim)
            k.lengthscale_constraint = GreaterThan(0.05)
            return ScaleKernel(k)

        def make_linear():
            return ScaleKernel(LinearKernel())

        def make_matern(nu):
            k = MaternKernel(nu=nu, ard_num_dims=input_dim)
            k.lengthscale_constraint = GreaterThan(0.05)
            return ScaleKernel(k)

        kern_dict = {
            'lin': make_linear,
            'rbf': make_rbf,
            'lin_rbf': lambda: make_rbf() + make_linear(),
            'matern32': lambda: make_matern(nu=1.5),
            'lin_matern32': lambda: make_matern(nu=1.5) + make_linear(),
            'matern52': lambda: make_matern(nu=2.5),
            'lin_matern52': lambda: make_matern(nu=2.5) + make_linear(),
        }

        if self.kernel_type not in kern_dict:
            raise ValueError(f"Kernel type '{self.kernel_type}' not found.")
        return kern_dict[self.kernel_type]()

    def _apply_init(self, init_params: Dict[str, Any]):
        """Applies specific values to GPyTorch hyperparameters."""
        # [Same as before]
        print(f"  Applying user initialization: {init_params}")
        if 'noise' in init_params and hasattr(self.likelihood, 'noise'):
            self.likelihood.noise_covar.initialize(noise=init_params['noise'])

        for name, module in self.model.named_modules():
            if 'outputscale' in init_params and hasattr(module, 'outputscale'):
                module.outputscale = init_params['outputscale']
            if 'lengthscale' in init_params and hasattr(module, 'lengthscale'):
                try:
                    if module.lengthscale.dim() > 0 and module.lengthscale.shape[-1] > 1:
                        module.lengthscale = torch.full_like(module.lengthscale, init_params['lengthscale'])
                    else:
                        module.lengthscale = init_params['lengthscale']
                except Exception as e:
                    warnings.warn(f"Failed to init lengthscale: {e}")

    def _fit(self, train_data, max_iters=1000, lr=0.1, **kwargs):
        """
        Fits the GPyTorch model.
        """
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model or Likelihood not initialized.")

        # Unpack and Convert to Torch
        if len(train_data) == 3:
            inputs, labels, _ = train_data
        else:
            inputs, labels = train_data

        # Ensure Float/Double precision and correct device
        # (Assuming CPU for now, add .cuda() here if needed)
        x_torch = torch.from_numpy(inputs).float()
        y_torch = torch.from_numpy(labels).float().flatten()

        # Initialize Optimizer Wrapper
        optimizer = GPyTorchOptimizer(self.model, self.likelihood, x_torch, y_torch)

        # Run Optimization (Defaults to NGD for variational, Adam for hyperparams)
        final_loss = optimizer.train(max_iters=max_iters, lr=lr)
        return final_loss

    def _predict(self, test_data):
        """Prediction using pure posterior sampling."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)

        x_torch = torch.from_numpy(test_data).float()

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            dist = self.model(x_torch)
            f_samples_torch = dist.sample(torch.Size([1000]))
            f_samples = f_samples_torch.t().numpy()

        return self._compute_sample_stats(f_samples, axis=1)


class GPyTorchOptimizer:
    """
    Optimizer wrapper that handles Variational Parameters and Hyperparameters separately.
    """

    def __init__(self, model, likelihood, x, y):
        self.model = model
        self.likelihood = likelihood
        self.x = x
        self.y = y
        # VariationalELBO is standard for ApproximateGP
        self.mll = VariationalELBO(likelihood, model, num_data=y.size(0))

    def _split_parameters(self):
        """
        Identify variational parameters vs model/likelihood hyperparameters.
        Returns:
            var_params (list): Parameters for NGD (mean/chol_cov).
            hyper_params (list): Kernel lengthscales, outputscales, likelihood noise.
        """
        # 1. Identify Variational Parameters
        # forcing list() ensures we can iterate it multiple times if needed
        variational_params = list(self.model.variational_parameters())

        # Create a set of IDs for fast O(1) lookup
        variational_ids = set(id(p) for p in variational_params)

        # 2. Identify All Other Parameters (Hyperparameters)
        hyperparameters = []

        # Combine model and likelihood parameters into one iterable to check
        # We use a set of IDs for hyperparameters to prevent duplicates if
        # the likelihood is somehow attached to the model (rare but possible)
        hyper_ids = set()

        # Chain iterators to handle both model and likelihood
        import itertools
        all_params = itertools.chain(self.model.parameters(), self.likelihood.parameters())

        for p in all_params:
            p_id = id(p)
            # If it is NOT variational AND we haven't seen it yet
            if p_id not in variational_ids and p_id not in hyper_ids:
                hyperparameters.append(p)
                hyper_ids.add(p_id)

        return variational_params, hyperparameters

    def train(self, optimizer_name='ngd', max_iters=1000, lr=0.1):
        self.model.train()
        self.likelihood.train()

        # NGD is the preferred method for this split
        if optimizer_name.lower() == 'ngd':
            return self._train_ngd_adam(max_iters, var_lr=lr, hyper_lr=0.01)
        elif optimizer_name.lower() == 'adam':
            # Fallback to standard Adam for everything if requested
            return self._train_adam(max_iters, lr)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported.")

    def _train_ngd_adam(self, max_iters, var_lr=0.1, hyper_lr=0.01):
        """
        Optimization Strategy:
        1. Variational Params -> Natural Gradient Descent (NGD)
        2. Hyperparameters    -> Adam
        """
        var_params, hyper_params = self._split_parameters()

        # 1. Setup Optimizers
        # NGD for variational distribution (natural geometry)
        ngd_optimizer = NGD(
            var_params,
            num_data=self.y.size(0),
            lr=var_lr
        )

        # Adam for kernel hyperparameters and likelihood noise
        # Note: Hyperparameters often need a smaller LR than variational params
        adam_optimizer = torch.optim.Adam(
            hyper_params,
            lr=hyper_lr
        )

        # 2. Setup Schedulers (Optional but recommended)
        # Only decay Adam LR, NGD usually handles steps naturally
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            adam_optimizer, mode='min', factor=0.5, patience=20
        )

        # 3. Training Loop
        loop = tqdm.tqdm(range(max_iters), desc="NGD+Adam", leave=False)
        best_loss = float('inf')
        last_best_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for i in loop:
            # Zero gradients for both
            ngd_optimizer.zero_grad()
            adam_optimizer.zero_grad()

            # Forward Pass
            output = self.model(self.x)
            loss = -self.mll(output, self.y)

            if torch.isnan(loss):
                print("  [!] Loss became NaN.")
                break

            # Backward Pass
            loss.backward()

            # Step both optimizers
            ngd_optimizer.step()
            adam_optimizer.step()

            # Logging & Scheduling
            current_loss = loss.item()
            scheduler.step(current_loss)

            # Early Stopping Check
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                epochs_no_improve = 0
                last_best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 30:
                # Restore best model if we haven't improved in a while
                break

            if i % 50 == 0:
                loop.set_description(f"Loss: {current_loss:.3f}")

        # Restore best state
        self.model.load_state_dict(last_best_state)
        return best_loss

    def _train_adam(self, max_iters, lr):
        """Fallback: Use Adam for everything (Standard)."""
        optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}, {'params': self.likelihood.parameters()}],
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15
        )

        loop = tqdm.tqdm(range(max_iters), desc="Adam (All)", leave=False)
        best_loss = float('inf')

        for i in loop:
            optimizer.zero_grad()
            output = self.model(self.x)
            loss = -self.mll(output, self.y)
            loss.backward()
            optimizer.step()

            scheduler.step(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()

            if i % 50 == 0:
                loop.set_description(f"Loss: {loss.item():.3f}")

        return best_loss
