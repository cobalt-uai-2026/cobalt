import numpy as np
import warnings

# Try importing torch/gpytorch, handle if missing (though your code implies they exist)
try:
    import torch
    import gpytorch
except ImportError:
    torch = None
    gpytorch = None


# =============================================================================
# NLPD Computations (Negative Log Predictive Density)
# =============================================================================

def calc_nlpd(model, X, y_true, likelihood=None, censoring=None):
    """
    Calculates the Negative Log Predictive Density (NLPD) for observed data.
    """
    # 1. Unwrap model if it's a wrapper
    inner_model = model.model if hasattr(model, 'model') else model

    # 2. Check for PyTorch (GPyTorch)
    is_torch = False
    if torch is not None:
        if isinstance(model, torch.nn.Module) or isinstance(inner_model, torch.nn.Module):
            is_torch = True

    if is_torch:
        return _calc_nlpd_gpytorch(model, X, y_true, likelihood, censoring)

    # 3. Check for GPy (Legacy)
    # GPy models usually have 'log_predictive_density' or 'predict' methods
    is_gpy = hasattr(inner_model, 'log_predictive_density') or hasattr(inner_model, 'predict')

    if is_gpy:
        return _calc_nlpd_gpy(model, X, y_true, censoring)

    warnings.warn(f"Model type {type(model)} not recognized. Returning NaN.")
    return np.nan


def calc_latent_nlpd(model, X, y_latent_true, likelihood=None):
    """
    Calculates NLPD w.r.t the true latent function values.
    """
    # 1. Unwrap model
    inner_model = model.model if hasattr(model, 'model') else model

    # 2. Check for PyTorch (GPyTorch)
    is_torch = False
    if torch is not None:
        if isinstance(model, torch.nn.Module) or isinstance(inner_model, torch.nn.Module):
            is_torch = True

    if is_torch:
        return _calc_latent_nlpd_gpytorch(model, X, y_latent_true, likelihood)

    # 3. Check for GPy
    is_gpy = hasattr(inner_model, 'log_predictive_density') or hasattr(inner_model, 'predict')
    if is_gpy:
        return _calc_latent_nlpd_gpy(model, X, y_latent_true)

    warnings.warn(f"Model type {type(model)} not recognized. Returning NaN.")
    return np.nan


# =============================================================================
# Internal Helpers (GPy vs GPyTorch)
# =============================================================================

def _calc_nlpd_gpy(model, X, y_true, censoring):
    """Helper: Calculates NLPD for GPy models."""
    gpy_model = model.model if hasattr(model, 'model') else model

    # Handle Tensor inputs if user passed them
    if torch is not None and torch.is_tensor(X):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if torch is not None and torch.is_tensor(y_true):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)

    y_reshaped = y_true_np.reshape(-1, 1)

    metadata = None
    if censoring is not None:
        if torch is not None and torch.is_tensor(censoring):
            c_np = censoring.detach().cpu().numpy()
        else:
            c_np = np.asarray(censoring)
        metadata = {'censoring': c_np.reshape(-1, 1)}

    # GPy's log_predictive_density returns the log prob for each point
    lpd = gpy_model.log_predictive_density(X_np, y_reshaped, Y_metadata=metadata)
    return -np.mean(lpd)


def _calc_latent_nlpd_gpy(model, X, y_latent_true):
    """Helper: Calculates Latent NLPD for GPy models (Gaussian NLL on f)."""
    gpy_model = model.model if hasattr(model, 'model') else model

    if torch is not None and torch.is_tensor(X):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if torch is not None and torch.is_tensor(y_latent_true):
        y_true_np = y_latent_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_latent_true)

    # Predict latent function f (not y)
    # predict() in GPy typically returns (mean, var) of f
    mu, var = gpy_model.predict(X_np)

    # Calculate Negative Log Likelihood for Gaussian: N(y_true | mu, var)
    # NLL = 0.5 * log(2*pi*var) + (y - mu)^2 / (2*var)

    # Flatten everything
    mu = mu.reshape(-1)
    var = var.reshape(-1)
    y_true = y_true_np.reshape(-1)

    # Avoid div by zero
    var = np.maximum(var, 1e-9)

    nll = 0.5 * (np.log(2 * np.pi * var) + np.square(y_true - mu) / var)
    return np.mean(nll)


def _calc_nlpd_gpytorch(model, X, y_true, likelihood, censoring=None):
    torch_model = model.model if hasattr(model, 'model') else model

    # Attempt to find likelihood if not passed
    if likelihood is None:
        likelihood = getattr(model, 'likelihood', getattr(torch_model, 'likelihood', None))

    torch_model.eval()
    if likelihood: likelihood.eval()

    # Robust device detection
    try:
        param = next(torch_model.parameters())
        device, dtype = param.device, param.dtype
    except StopIteration:
        device, dtype = 'cpu', torch.float32

    # Prepare Tensors
    X_torch = torch.as_tensor(X, device=device, dtype=dtype)
    y_torch = torch.as_tensor(y_true, device=device, dtype=dtype).reshape(-1)

    if censoring is not None:
        censoring = torch.as_tensor(censoring, device=device, dtype=dtype).reshape(-1)

    try:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            q_f = torch_model(X_torch)

            # --- STRATEGY 1: Analytical Marginal (e.g. ExactGP) ---
            if hasattr(likelihood, 'log_marginal'):
                # Note: log_marginal signatures vary, check your specific likelihood implementation
                log_prob = likelihood.log_marginal(y_torch, q_f, censoring)
                return -log_prob.mean().item()

            # --- STRATEGY 2: Robust Monte Carlo (e.g. Censored Likelihoods) ---
            elif hasattr(likelihood, 'log_prob_density') or hasattr(likelihood, 'expected_log_prob'):
                means = q_f.mean
                vars = q_f.variance
                vars = torch.clamp(vars, min=1e-6)
                stddevs = vars.sqrt()

                num_samples = 1000
                # Shape: (S, N)
                epsilon = torch.randn(num_samples, len(means), device=device, dtype=dtype)
                f_samples = means.unsqueeze(0) + stddevs.unsqueeze(0) * epsilon

                # Calculate Log Likelihoods per sample
                # Try passing censoring, fall back if not supported
                try:
                    log_lik_samples = likelihood.log_prob_density(f_samples, y_torch, censoring)
                except TypeError:
                    # Fallback for likelihoods that don't take censoring (e.g. standard Gaussian)
                    log_lik_samples = likelihood.log_prob_density(f_samples, y_torch)

                # Numerical Stability for LogSumExp
                # Replace NaNs/Infs with small number so they don't propagate
                log_lik_samples = torch.nan_to_num(log_lik_samples, nan=-1e10, neginf=-1e10)

                # LogSumExp trick: log(1/S * sum(exp(log_p))) = logsumexp(log_p) - log(S)
                S = torch.tensor(num_samples, device=device, dtype=dtype)
                lpd = torch.logsumexp(log_lik_samples, dim=0) - torch.log(S)

                # Clip very low probabilities
                lpd = torch.clamp(lpd, min=-1e10)

                return -lpd.mean().item()

            # --- STRATEGY 3: Standard Gaussian Fallback ---
            else:
                pred_dist = likelihood(q_f)

                if pred_dist.event_shape[0] != y_torch.shape[0]:
                    warnings.warn(f"Shape mismatch: {pred_dist.event_shape[0]} vs {y_torch.shape[0]}.")
                    return np.nan

                means = pred_dist.mean
                vars = pred_dist.variance
                vars = torch.clamp(vars, min=1e-6)

                marginal_dist = torch.distributions.Normal(means, vars.sqrt())
                log_prob = marginal_dist.log_prob(y_torch)
                return -log_prob.mean().item()

    except Exception as e:
        warnings.warn(f"GPyTorch NLPD failed: {e}")
        return np.nan


def _calc_latent_nlpd_gpytorch(model, X, y_latent_true, likelihood=None):
    torch_model = model.model if hasattr(model, 'model') else model

    try:
        param = next(torch_model.parameters())
        device, dtype = param.device, param.dtype
    except StopIteration:
        device, dtype = 'cpu', torch.float32

    X_torch = torch.as_tensor(X, device=device, dtype=dtype)
    y_torch = torch.as_tensor(y_latent_true, device=device, dtype=dtype).reshape(-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        q_f = torch_model(X_torch)
        means = q_f.mean
        vars = q_f.variance

        # Clamp variance to avoid numerical issues
        vars = torch.clamp(vars, min=1e-6)

        # Latent NLPD is effectively how well the GP posterior explains the true latent function
        # We treat this as a Gaussian NLL
        marginal_dist = torch.distributions.Normal(means, vars.sqrt())
        log_prob = marginal_dist.log_prob(y_torch)

        return -log_prob.mean().item()


# =============================================================================
# Error & Coverage Metrics (Unchanged)
# =============================================================================

def hinge_mae(y_true, y_pred, censoring):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    censoring = np.asarray(censoring).flatten()

    errors = np.abs(y_pred - y_true)

    mask_right = (censoring == 1)
    if np.any(mask_right):
        errors[mask_right] = np.maximum(0, y_true[mask_right] - y_pred[mask_right])

    mask_left = (censoring == -1)
    if np.any(mask_left):
        errors[mask_left] = np.maximum(0, y_pred[mask_left] - y_true[mask_left])

    return np.mean(errors)


def interval_coverage(y_true, gp_lower, gp_upper, censoring):
    y_true = np.asarray(y_true).flatten()
    gp_lower = np.asarray(gp_lower).flatten()
    gp_upper = np.asarray(gp_upper).flatten()
    censoring = np.asarray(censoring).flatten()

    hits = np.zeros(len(y_true), dtype=bool)

    mask_unc = (censoring == 0)
    hits[mask_unc] = (y_true[mask_unc] >= gp_lower[mask_unc]) & (y_true[mask_unc] <= gp_upper[mask_unc])

    mask_right = (censoring == 1)
    hits[mask_right] = (gp_upper[mask_right] >= y_true[mask_right])

    mask_left = (censoring == -1)
    hits[mask_left] = (gp_lower[mask_left] <= y_true[mask_left])

    return np.mean(hits)


def latent_interval_coverage(y_true_latent, gp_lower, gp_upper):
    y_true_latent = np.asarray(y_true_latent).flatten()
    gp_lower = np.asarray(gp_lower).flatten()
    gp_upper = np.asarray(gp_upper).flatten()

    hits = (y_true_latent >= gp_lower) & (y_true_latent <= gp_upper)
    return np.mean(hits)