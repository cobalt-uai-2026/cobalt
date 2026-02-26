import abc
import warnings
import numpy as np
import torch
import gpytorch
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error,
    accuracy_score, precision_score, recall_score, jaccard_score
)

# Import the core mathematical calculations from your metrics file
# Ensure 'calc_nlpd' and 'calc_latent_nlpd' are the ROBUST versions we fixed previously
from censored_regressors.metrics.metrics import (
    hinge_mae, interval_coverage, calc_nlpd, calc_latent_nlpd, latent_interval_coverage
)


# ==========================================
# 0. Internal Helpers (Robust Subsetting)
# ==========================================
def _get_classification_labels(ground_truth_limits, pred, censoring):
    """
    Internal helper to map continuous predictions to binary regions (0=Observed, 1=Censored).
    """
    limits = np.asarray(ground_truth_limits).flatten()
    pred = np.asarray(pred).flatten()
    censoring = np.asarray(censoring).flatten()

    y_true_class = (censoring != 0).astype(int)
    y_pred_class = np.zeros_like(y_true_class)

    mask_right = (censoring == 1)
    if np.any(mask_right):
        y_pred_class[mask_right] = (pred[mask_right] >= limits[mask_right]).astype(int)

    mask_left = (censoring == -1)
    if np.any(mask_left):
        y_pred_class[mask_left] = (pred[mask_left] <= limits[mask_left]).astype(int)

    return y_true_class, y_pred_class


def _subset_data(X, y, censoring, mask):
    """
    Safely subsets X, y, and censoring arrays/tensors using a boolean mask.
    Handles both numpy arrays and torch tensors seamlessly.
    """
    # Ensure mask is a flat numpy array for logic checks
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy().flatten()
    else:
        mask_np = np.asarray(mask).flatten()

    if not np.any(mask_np):
        return None, None, None

    def _slice_item(item):
        if item is None:
            return None
        if torch.is_tensor(item):
            # Convert mask to tensor on the same device
            mask_t = torch.tensor(mask_np, device=item.device, dtype=torch.bool)
            # Handle 2D tensors (like X features) vs 1D tensors (targets)
            return item[mask_t, :] if item.dim() > 1 else item[mask_t]
        else:
            item_np = np.asarray(item)
            return item_np[mask_np, :] if item_np.ndim > 1 else item_np[mask_np]

    return _slice_item(X), _slice_item(y), _slice_item(censoring)


# ==========================================
# 1. Metaclass & Base Class
# ==========================================
class MetricFunction(abc.ABCMeta):
    """Metaclass that allows classes to be called like functions."""

    def __call__(cls, *args, **kwargs):
        instance = super(MetricFunction, cls).__call__()
        return instance.evaluate(*args, **kwargs)


class Evaluation(metaclass=MetricFunction):
    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


# ==========================================
# 2. Metric Implementations
# ==========================================
class NLPD(Evaluation):
    """Global Negative Log Predictive Density (All Data)."""

    def evaluate(self, model, X, ground_truth, censoring=None, likelihood=None):
        return calc_nlpd(model, X, ground_truth, censoring=censoring, likelihood=likelihood)


class NLPD_c(Evaluation):
    """NLPD computed ONLY on originally censored points."""

    def evaluate(self, model, X, ground_truth, censoring, likelihood=None, **kwargs):
        if censoring is None:
            return np.nan

        c_np = censoring.detach().cpu().numpy() if torch.is_tensor(censoring) else np.asarray(censoring)
        mask = (c_np.flatten() != 0)

        # FIX: Subset X, y, AND censoring together
        X_sub, y_sub, c_sub = _subset_data(X, ground_truth, censoring, mask)

        if X_sub is None or len(X_sub) == 0:
            return np.nan

        # Pass the subsetted censoring explicitly to calc_nlpd
        return calc_nlpd(model, X_sub, y_sub, censoring=c_sub, likelihood=likelihood)


class NLPD_nc(Evaluation):
    """NLPD computed ONLY on originally uncensored (observed) points."""

    def evaluate(self, model, X, ground_truth, censoring, likelihood=None, **kwargs):
        if censoring is None:
            return np.nan

        c_np = censoring.detach().cpu().numpy() if torch.is_tensor(censoring) else np.asarray(censoring)
        mask = (c_np.flatten() == 0)

        # FIX: Subset X, y, AND censoring together
        X_sub, y_sub, c_sub = _subset_data(X, ground_truth, censoring, mask)

        if X_sub is None or len(X_sub) == 0:
            return np.nan

        return calc_nlpd(model, X_sub, y_sub, censoring=c_sub, likelihood=likelihood)


class Latent_NLPD(Evaluation):
    """Exact NLPD for Latent Uncensored Targets (Global)."""

    def evaluate(self, model, X, ground_truth_latent, likelihood=None, **kwargs):
        return calc_latent_nlpd(model, X, ground_truth_latent, likelihood=likelihood)


class Latent_NLPD_c(Evaluation):
    """
    Exact NLPD evaluated ONLY on points that were originally censored.
    We verify how well the model recovered the true hidden value.
    """

    def evaluate(self, model, X, ground_truth_latent, censoring, likelihood=None, **kwargs):
        if censoring is None:
            return np.nan

        c_np = censoring.detach().cpu().numpy() if torch.is_tensor(censoring) else np.asarray(censoring)
        mask = (c_np.flatten() != 0)

        # Subset X and Latent Truth (censoring array not needed for latent calc, but needed for mask)
        X_sub, y_sub, _ = _subset_data(X, ground_truth_latent, censoring, mask)

        if X_sub is None or len(X_sub) == 0:
            return np.nan

        return calc_latent_nlpd(model, X_sub, y_sub, likelihood=likelihood)


class Latent_NLPD_nc(Evaluation):
    """
    Exact NLPD evaluated ONLY on points that were originally observed.
    """

    def evaluate(self, model, X, ground_truth_latent, censoring, likelihood=None, **kwargs):
        if censoring is None:
            return np.nan

        c_np = censoring.detach().cpu().numpy() if torch.is_tensor(censoring) else np.asarray(censoring)
        mask = (c_np.flatten() == 0)

        X_sub, y_sub, _ = _subset_data(X, ground_truth_latent, censoring, mask)

        if X_sub is None or len(X_sub) == 0:
            return np.nan

        return calc_latent_nlpd(model, X_sub, y_sub, likelihood=likelihood)


# ... (Standard Error Metrics Wrapper Classes) ...

class MAE(Evaluation):
    def evaluate(self, ground_truth, pred, **kwargs):
        return mean_absolute_error(ground_truth, pred)


class MAE_c(Evaluation):
    def evaluate(self, ground_truth, pred, censoring, **kwargs):
        ground_truth = np.asarray(ground_truth).flatten()
        pred = np.asarray(pred).flatten()
        censoring = np.asarray(censoring).flatten()
        mask = (censoring != 0)
        return mean_absolute_error(ground_truth[mask], pred[mask]) if np.any(mask) else np.nan


class MAE_nc(Evaluation):
    def evaluate(self, ground_truth, pred, censoring, **kwargs):
        ground_truth = np.asarray(ground_truth).flatten()
        pred = np.asarray(pred).flatten()
        censoring = np.asarray(censoring).flatten()
        mask = (censoring == 0)
        return mean_absolute_error(ground_truth[mask], pred[mask]) if np.any(mask) else np.nan


class RMSE(Evaluation):
    def evaluate(self, ground_truth, pred, **kwargs):
        return root_mean_squared_error(ground_truth, pred)


class Hinge_MAE(Evaluation):
    def evaluate(self, ground_truth, pred, censoring, **kwargs):
        return hinge_mae(ground_truth, pred, censoring)


class Interval_Coverage(Evaluation):
    def evaluate(self, ground_truth, gp_lower, gp_upper, censoring, **kwargs):
        return interval_coverage(ground_truth, gp_lower, gp_upper, censoring)


class Latent_Interval_Coverage(Evaluation):
    def evaluate(self, ground_truth_latent, gp_lower, gp_upper, **kwargs):
        return latent_interval_coverage(ground_truth_latent, gp_lower, gp_upper)


class Accuracy(Evaluation):
    def evaluate(self, ground_truth_limits, pred, censoring, **kwargs):
        y_true, y_pred = _get_classification_labels(ground_truth_limits, pred, censoring)
        return accuracy_score(y_true, y_pred)


class Precision(Evaluation):
    def evaluate(self, ground_truth_limits, pred, censoring, **kwargs):
        y_true, y_pred = _get_classification_labels(ground_truth_limits, pred, censoring)
        return precision_score(y_true, y_pred, zero_division=0)


class Recall(Evaluation):
    def evaluate(self, ground_truth_limits, pred, censoring, **kwargs):
        y_true, y_pred = _get_classification_labels(ground_truth_limits, pred, censoring)
        return recall_score(y_true, y_pred, zero_division=0)


class Jaccard(Evaluation):
    def evaluate(self, ground_truth_limits, pred, censoring, **kwargs):
        y_true, y_pred = _get_classification_labels(ground_truth_limits, pred, censoring)
        return jaccard_score(y_true, y_pred)


# ==========================================
# 3. Summary Evaluation Functions
# ==========================================

def evaluate_observed(X, ground_truth_cens, pred, gp_lower, gp_upper, censoring, model, likelihood=None):
    """Evaluates the model using ONLY the observed/censored data."""
    return {
        'NLPD': NLPD(model, X, ground_truth_cens, censoring=censoring, likelihood=likelihood),
        'NLPD_c': NLPD_c(model, X, ground_truth_cens, censoring=censoring, likelihood=likelihood),
        'NLPD_nc': NLPD_nc(model, X, ground_truth_cens, censoring=censoring, likelihood=likelihood),

        'RMSE': RMSE(ground_truth_cens, pred),
        'MAE': MAE(ground_truth_cens, pred),
        'MAE_nc': MAE_nc(ground_truth_cens, pred, censoring),

        'Hinge_MAE': Hinge_MAE(ground_truth_cens, pred, censoring),
        'Coverage': Interval_Coverage(ground_truth_cens, gp_lower, gp_upper, censoring),

        'Accuracy': Accuracy(ground_truth_cens, pred, censoring),
        'Precision': Precision(ground_truth_cens, pred, censoring),
        'Recall': Recall(ground_truth_cens, pred, censoring),
        'Jaccard': Jaccard(ground_truth_cens, pred, censoring)
    }


def evaluate_latent(X, ground_truth_cens, ground_truth_latent, pred, gp_lower, gp_upper, censoring, model,
                    likelihood=None):
    """Evaluates the model using the Oracle LATENT values."""
    return {
        'NLPD_latent': Latent_NLPD(model, X, ground_truth_latent, likelihood=likelihood),
        'NLPD_c_latent': Latent_NLPD_c(model, X, ground_truth_latent, censoring=censoring, likelihood=likelihood),
        'NLPD_nc_latent': Latent_NLPD_nc(model, X, ground_truth_latent, censoring=censoring, likelihood=likelihood),

        'RMSE': RMSE(ground_truth_latent, pred),
        'MAE': MAE(ground_truth_latent, pred),
        'MAE_c': MAE_c(ground_truth_latent, pred, censoring),
        'MAE_nc': MAE_nc(ground_truth_latent, pred, censoring),

        'Coverage': Latent_Interval_Coverage(ground_truth_latent, gp_lower, gp_upper),

        'Accuracy': Accuracy(ground_truth_cens, pred, censoring),
        'Precision': Precision(ground_truth_cens, pred, censoring),
        'Recall': Recall(ground_truth_cens, pred, censoring),
        'Jaccard': Jaccard(ground_truth_cens, pred, censoring)
    }