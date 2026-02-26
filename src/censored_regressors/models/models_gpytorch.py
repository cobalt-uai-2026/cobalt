import copy
import warnings
import traceback
import numpy as np
import torch
import gpytorch

# --- PyTorch Utilities ---
from torch.autograd.functional import hessian
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    NaturalVariationalDistribution,
    VariationalStrategy,
    UnwhitenedVariationalStrategy
)
from gpytorch.utils.errors import NotPSDError

# --- GPy (Legacy Support) ---
try:
    import GPy
except ImportError:
    GPy = None

# --- Local Modules ---
from censored_regressors.models.models_base import BaseGPyTorchModel, GPyTorchOptimizer
# Ensure this import works
from censored_regressors.likelihoods.censored_likelihood_gpytorch import CensoredGaussianLikelihoodAnalytic

# --- Helper for GPy Fallback ---
try:
    from censored_regressors.models.censored_model_gpy import GPCensoredRegression
except ImportError:
    GPCensoredRegression = None


class GPModel(ApproximateGP):
    """
    Standard ApproximateGP wrapper for GPyTorch.
    """

    def __init__(self, x, y, kernel_module, variational_dist_type='cholesky', variational_strategy_type='unwhitened'):
        num_inducing = x.size(0)

        if variational_dist_type == 'cholesky':
            variational_distribution = CholeskyVariationalDistribution(num_inducing)
        elif variational_dist_type == 'natural':
            variational_distribution = NaturalVariationalDistribution(num_inducing)
        else:
            raise ValueError(f"Unknown variational_dist_type: {variational_dist_type}")

        if variational_strategy_type == 'variationalstrategy':
            strategy = VariationalStrategy(
                self, x, variational_distribution, learn_inducing_locations=False, jitter_val=1e-3
            )
        elif variational_strategy_type == 'unwhitened':
            strategy = UnwhitenedVariationalStrategy(
                self, x, variational_distribution, learn_inducing_locations=False, jitter_val=1e-3
            )
        else:
            raise ValueError(f"Unknown variational_strategy_type: {variational_strategy_type}")

        super().__init__(strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel_module
        self.x = x
        self.y = y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CensoredGP_VI_gpytorch(BaseGPyTorchModel):
    name = 'CensoredGP_VI'

    def __init__(self, kernel_type='lin_rbf',  name='CensoredGP_VI', **kwargs):
        super().__init__(kernel_type=kernel_type, name=name)
        self.likelihood = None
        self.model = None

    def _fit(self, *args, **kwargs):
        # 1. Flatten the args
        # If the loop passes model.fit((X, Y, C)), args might be ((X, Y, C),)
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            data_to_process = args[0]
        else:
            data_to_process = args

        # 2. Extract exactly the first 3 elements (X, Y, Censoring)
        try:
            inputs = data_to_process[0]
            labels = data_to_process[1]
            censoring = data_to_process[2]
        except (IndexError, TypeError):
            raise ValueError(
                f"VI Model expected 3 data elements, but got: {type(data_to_process)} with length {len(data_to_process) if hasattr(data_to_process, '__len__') else 'unknown'}")

        # Extract parameters from kwargs (or use defaults)
        optimizer = kwargs.get('optimizer', 'ngd')
        num_restarts = kwargs.get('num_restarts', 3)
        max_iters = kwargs.get('max_iters', 2000)
        init_params = kwargs.get('init_params', 'laplace')

        # --- Proceed with tensor conversion ---
        x = torch.tensor(inputs).double()
        y = torch.tensor(labels).squeeze().double()
        c_sq = torch.tensor(censoring).squeeze().double()

        # The COBALT paper defines c_i in {-1, 0, 1}:
        #  0 = Observed (Uncensored)
        #  1 = Right/Upper Censored (We know true f_i >= y_i)
        # -1 = Left/Lower Censored (We know true f_i <= y_i)

        # Initialize observation bounds with "infinite" values to default to Uncensored.
        ub = torch.full_like(y, float('inf'))
        lb = torch.full_like(y, -float('inf'))

        # --- Right/Upper Censoring (c = 1) ---
        # Math: The true latent value is greater than or equal to the observation (f_i >= y_i).
        # Likelihood Logic: To trigger the `is_right` mask (target >= high),
        # we must set the upper bound of the observation window to y_i.
        mask_right = (c_sq == 1)
        ub[mask_right] = y[mask_right]

        # --- Left/Lower Censoring (c = -1) ---
        # Math: The true latent value is less than or equal to the observation (f_i <= y_i).
        # Likelihood Logic: To trigger the `is_left` mask (target <= low),
        # we must set the lower bound of the observation window to y_i.
        mask_left = (c_sq == -1)
        lb[mask_left] = y[mask_left]

        # Optimizer Config
        if optimizer == 'ngd':
            dist_type = 'natural'
            strat_type = 'unwhitened'
        else:
            dist_type = 'cholesky'
            strat_type = 'unwhitened'

        best_loss = float('inf')
        best_state_dict = None

        print(f"Fitting {self.name} with {optimizer.upper()} (Restarts: {num_restarts}, Init: {init_params})...")

        for i in range(num_restarts):
            # 1. Instantiate Likelihood with Bounds
            self.likelihood = CensoredGaussianLikelihoodAnalytic(low=lb, high=ub).double()
            # FIX: Register constraint on .noise because that is the name used in your Likelihood module
            if hasattr(self.likelihood, 'noise') and hasattr(self.likelihood.noise, 'register_constraint'):
                self.likelihood.noise.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-4))

            # 2. Instantiate Model
            kernel_module = self._get_kernel(input_dim=x.shape[-1])
            self.model = GPModel(
                x, y,
                kernel_module=kernel_module,
                variational_dist_type=dist_type,
                variational_strategy_type=strat_type
            ).double()

            # Align Inducing Points
            if hasattr(self.model.variational_strategy, 'inducing_points'):
                self.model.variational_strategy.inducing_points.data.copy_(x)

            # 3. Initialization Logic
            init_success = False
            if i == 0:
                if (init_params == 'empirical'):
                    print(f"  [Restart {i + 1}] Attempting Empirical Initialization...")
                    init_success = self._init_empirical(x, y)
                elif (init_params == 'gpy'):
                    print(f"  [Restart {i + 1}] Attempting GPy Laplace Initialization...")
                    init_success = self._init_from_gpy(inputs, labels, censoring)
                elif init_params == 'laplace':
                    print(f"  [Restart {i + 1}] Attempting Native PyTorch Laplace Initialization...")
                    init_success = self._init_via_laplace_torch(x, y)
                elif isinstance(init_params, dict):
                    self._apply_init(init_params)

            if not init_success and i == 0 and init_params is not None and not isinstance(init_params, dict):
                print("  [!] Initialization failed. Falling back to random initialization.")
                self._randomize_hyperparameters()
            elif i > 0 or init_params is None:
                self._randomize_hyperparameters()

            # 4. Optimize
            try:
                self._clear_cache()
                with gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.cholesky_max_tries(10):
                    trainer = GPyTorchOptimizer(self.model, self.likelihood, x, y)
                    current_max_iters = 200 if init_success else max_iters
                    loss = trainer.train(optimizer_name=optimizer, max_iters=current_max_iters)

                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = copy.deepcopy(self.model.state_dict())

            except (RuntimeError, NotPSDError) as e:
                print(f"  [!] Restart {i + 1} failed due to instability: {e}")
                if i == num_restarts - 1 and best_state_dict is None: raise e
                continue

        if best_state_dict:
            self.model.load_state_dict(best_state_dict)
            return True
        return False

    def _init_empirical(self, x, y):
        """
        Initializes hyperparameters based on empirical data statistics.
        Designed specifically to handle the nesting in BaseGPyTorchModel._get_kernel.
        """
        try:
            with torch.no_grad():
                # 1. Calculate Empirical Stats
                # Note: Using labels_std from your RegressionMethod if preprocess=True
                y_tensor = y if torch.is_tensor(y) else torch.tensor(y).double()
                y_var = torch.var(y_tensor)

                # 2. Set Noise Floor
                initial_noise = y_var * 0.1
                if hasattr(self.likelihood, 'noise'):
                    self.likelihood.noise.initialize(noise=initial_noise)

                # 3. Median Heuristic for Lengthscale
                n_samples = min(x.size(0), 1000)
                dist_matrix = torch.cdist(x[:n_samples], x[:n_samples])
                v = dist_matrix[torch.triu(torch.ones_like(dist_matrix), diagonal=1) == 1]
                median_dist = torch.median(v) if v.numel() > 0 else torch.tensor(1.0).to(x)
                initial_lengthscale = median_dist if median_dist > 1e-6 else torch.tensor(1.0).to(x)

                # 4. Recursive Parameter Injection
                # This handles ScaleKernel -> AdditiveKernel -> [RBF, Linear]
                def apply_recursive(module):
                    found_len = False
                    # Set outputscale if it's a ScaleKernel
                    if hasattr(module, 'outputscale'):
                        module.outputscale = y_var

                    # Set lengthscale if it's a Kernel that supports it (RBF, Matern)
                    if hasattr(module, 'lengthscale'):
                        module.lengthscale = initial_lengthscale
                        found_len = True

                    # Set variance if it's a Linear kernel
                    if isinstance(module, gpytorch.kernels.LinearKernel):
                        # Linear kernels often perform better starting with unit variance
                        # if the ScaleKernel above it is already set to y_var
                        module.variance = 1.0

                        # Dive into children (this handles SumKernels and ScaleKernel.base_kernel)
                    for child in module.children():
                        if apply_recursive(child):
                            found_len = True
                    return found_len

                has_len = apply_recursive(self.model.covar_module)

                status = "with lengthscale" if has_len else "no lengthscale (Linear)"
                print(f"  [+] Empirical Init successful ({status}).")
                return True

        except Exception as e:
            print(f"  [!] Empirical Init failed: {e}")
            return False

    def _init_via_laplace_torch(self, train_x, train_y):
        """Native PyTorch Laplace Approximation."""
        try:
            self.model.train()
            self.likelihood.train()

            # Optimize f_map and Hyperparams
            f_map = torch.nn.Parameter(torch.zeros_like(train_y).unsqueeze(-1))  # (N, 1)

            map_optimizer = torch.optim.Adam([
                {'params': f_map},
                {'params': self.model.covar_module.parameters()},
                {'params': self.likelihood.parameters()}
            ], lr=0.05)

            for k in range(150):
                map_optimizer.zero_grad()

                # 1. Log Prior: P(f | GP)
                prior_dist = self.model.forward(train_x)
                log_prior = prior_dist.log_prob(f_map.squeeze())

                # 2. Log Likelihood: P(y | f)
                # Use standard GPyTorch call which invokes expected_log_prob or log_prob
                # Since we don't have access to .log_prob_density from here easily if using generic Likelihood
                # we call likelihood(f_map) which returns distribution, then calculate log_prob(y)
                output_dist = self.likelihood(f_map.squeeze())
                log_lik = output_dist.log_prob(train_y).sum()

                loss = -(log_lik + log_prior)
                loss.backward()
                map_optimizer.step()

            # --- Phase 2: Compute Hessian ---
            self.model.eval()
            self.likelihood.eval()

            def neg_log_posterior(f_val):
                prior = self.model.forward(train_x)
                lp_prior = prior.log_prob(f_val)
                out = self.likelihood(f_val)
                lp_data = out.log_prob(train_y).sum()
                return -(lp_data + lp_prior)

            f_star = f_map.detach().squeeze().requires_grad_(True)
            H = hessian(neg_log_posterior, f_star)

            # --- Phase 3: Invert for Covariance ---
            jitter = torch.eye(H.size(0), dtype=H.dtype, device=H.device) * 1e-4
            S_laplace = torch.linalg.inv(H + jitter)

            # --- Phase 4: Inject into Variational Distribution ---
            with torch.no_grad():
                dist_module = self.model.variational_strategy._variational_distribution

                if isinstance(dist_module, CholeskyVariationalDistribution):
                    L_laplace = torch.linalg.cholesky(S_laplace)
                    dist_module.variational_mean.data.copy_(f_star)
                    dist_module.chol_variational_covar.data.copy_(L_laplace)

                elif isinstance(dist_module, NaturalVariationalDistribution):
                    precision_matrix = H
                    nat_vec = torch.matmul(precision_matrix, f_star)
                    nat_mat = -0.5 * precision_matrix
                    dist_module.natural_vec.data.copy_(nat_vec)
                    dist_module.natural_mat.data.copy_(nat_mat)

            self._clear_cache()
            print("  [+] Native Laplace initialization successful.")
            return True

        except Exception as e:
            print(f"  [!] Native Laplace Init failed: {e}")
            return False

    def _init_from_gpy(self, X, Y, censoring):
        """Legacy GPy initialization with robust kernel parameter extraction."""
        if GPy is None or GPCensoredRegression is None:
            print("  [!] GPy or GPCensoredRegression not imported.")
            return False

        try:
            X_gpy = X.reshape(-1, 1) if X.ndim == 1 else X
            Y_gpy = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            C_gpy = censoring.reshape(-1, 1) if censoring.ndim == 1 else censoring

            input_dim = X.shape[1]
            if self.kernel_type == 'lin_rbf':
                kernel_gpy = GPy.kern.RBF(input_dim, ARD=True) + GPy.kern.Linear(input_dim, ARD=True)
            else:
                kernel_gpy = GPy.kern.RBF(input_dim, ARD=True)

            m_gpy = GPCensoredRegression(
                X_gpy, Y_gpy, censoring=C_gpy, kernel=kernel_gpy,
                inference_method=GPy.inference.latent_function_inference.Laplace()
            )
            m_gpy.optimize(messages=False, max_iters=1000)

            # --- Robust Parameter Extraction ---
            gpy_len = None
            # Check the main kernel and its parts (for Sum/Add kernels)
            parts = m_gpy.kern.parts if hasattr(m_gpy.kern, 'parts') else [m_gpy.kern]

            for p in parts:
                if hasattr(p, 'lengthscale'):
                    gpy_len = torch.tensor(p.lengthscale.values).double()
                    if gpy_len.ndim == 1:
                        gpy_len = gpy_len.view(1, 1, -1)
                    break

            gpy_var = torch.tensor(m_gpy.kern.variance.values if hasattr(m_gpy.kern, 'variance') else 1.0).double()
            gpy_noise = torch.tensor(m_gpy.likelihood.variance.values).double()

            # Transfer to GPyTorch
            if gpy_len is not None:
                if hasattr(self.model.covar_module, 'base_kernel'):
                    self.model.covar_module.base_kernel.lengthscale = gpy_len
                elif hasattr(self.model.covar_module, 'lengthscale'):
                    self.model.covar_module.lengthscale = gpy_len

            if hasattr(self.likelihood, 'noise_covar'):
                self.likelihood.noise_covar.data = gpy_noise
            elif hasattr(self.likelihood, 'noise'):
                self.likelihood.noise.data = gpy_noise

            # Posterior Transfer
            post_mean = torch.tensor(m_gpy.posterior.mean).double().squeeze()
            post_cov = torch.tensor(m_gpy.posterior.covariance).double() + torch.eye(X.shape[0]).double() * 1e-5

            dist_module = self.model.variational_strategy._variational_distribution
            with torch.no_grad():
                if isinstance(dist_module, CholeskyVariationalDistribution):
                    chol_cov = torch.linalg.cholesky(post_cov)
                    dist_module.variational_mean.data.copy_(post_mean)
                    dist_module.chol_variational_covar.data.copy_(chol_cov)
                elif isinstance(dist_module, NaturalVariationalDistribution):
                    sigma_inv = torch.linalg.inv(post_cov)
                    dist_module.natural_vec.data.copy_(torch.matmul(sigma_inv, post_mean))
                    dist_module.natural_mat.data.copy_(-0.5 * sigma_inv)

            print("  [+] GPy Laplace parameters transplanted.")
            return True
        except Exception as e:
            print(f"  [!] GPy Init Error: {e}")
            return False

    def _randomize_hyperparameters(self):
        if hasattr(self.model.covar_module, 'outputscale'):
            self.model.covar_module.outputscale = 1.0 + torch.rand(1).double()

        # Robust check for noise parameter name
        noise_val = torch.tensor([0.1]).double() + torch.rand(1).double() * 0.1
        if hasattr(self.likelihood, 'noise_covar'):
            self.likelihood.noise_covar.data = noise_val
        elif hasattr(self.likelihood, 'noise'):
            self.likelihood.noise.data = noise_val

    def _clear_cache(self):
        if hasattr(self.model, 'clear_gpytorch_cache'):
            self.model.clear_gpytorch_cache()
        if hasattr(self.model.variational_strategy, '_memoize_cache'):
            self.model.variational_strategy._memoize_cache.clear()
