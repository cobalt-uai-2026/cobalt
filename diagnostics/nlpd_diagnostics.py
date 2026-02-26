import torch
import gpytorch
import numpy as np
import warnings
import math


# =============================================================================
# 1. The "Original" Problematic Implementation (Joint NLPD)
# =============================================================================
def _calc_nlpd_joint_original(model, X, y_true, likelihood):
    """
    Computes Joint NLPD: -1/N * log p(y_1, y_2, ..., y_N | X)

    WHY IT FAILS:
    1. Correlation: It assumes we care about the correlation between test points.
       If test points are close, the covariance matrix becomes singular.
    2. Complexity: It requires inverting the full N x N covariance matrix (O(N^3)).
    3. Scaling: The joint probability of N points is a tiny number. Dividing by N
       doesn't make it comparable to the marginal probability if correlations exist.
    """
    torch_model = model.model if hasattr(model, 'model') else model
    if likelihood is None:
        likelihood = getattr(model, 'likelihood', getattr(torch_model, 'likelihood', None))

    torch_model.eval()
    if likelihood: likelihood.eval()

    # Force CPU for stability in tests
    device = torch.device('cpu')
    dtype = torch.float32

    X_torch = torch.tensor(X, device=device, dtype=dtype) if not torch.is_tensor(X) else X
    y_torch = torch.tensor(y_true, device=device, dtype=dtype).squeeze() if not torch.is_tensor(
        y_true) else y_true.squeeze()

    try:
        # We add jitter, but for large N or collinear inputs, it's often not enough
        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3):
            q_f = torch_model(X_torch)
            pred_dist = likelihood(q_f)

            if pred_dist.event_shape[0] != y_torch.shape[0]:
                return np.nan

            # --- THE BOTTLENECK ---
            # This function attempts to Cholesky decompose the full N x N matrix.
            # If N > 1000 or X has duplicates, this crashes or returns NaN.
            nlpd_tensor = gpytorch.metrics.negative_log_predictive_density(pred_dist, y_torch)
            return nlpd_tensor.mean().item()

    except Exception as e:
        # Only catch math errors to show them in the test output
        return np.nan


# =============================================================================
# 2. The "Corrected" Robust Implementation (Marginal NLPD)
# =============================================================================
def _calc_nlpd_marginal_fixed(model, X, y_true, likelihood):
    """
    Computes Marginal NLPD: -1/N * sum( log p(y_i | x_i) )

    WHY IT WORKS:
    1. Independence: It treats every test point as an independent prediction task.
       This matches standard regression metrics (like RMSE) and GPy's behavior.
    2. Stability: It only computes the DIAGONAL variances. No matrix inversion.
    3. Complexity: O(N) instead of O(N^3). Fast and never crashes.
    """
    torch_model = model.model if hasattr(model, 'model') else model
    if likelihood is None:
        likelihood = getattr(model, 'likelihood', getattr(torch_model, 'likelihood', None))

    torch_model.eval()
    if likelihood: likelihood.eval()

    X_torch = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
    y_torch = torch.tensor(y_true, dtype=torch.float32).squeeze() if not torch.is_tensor(y_true) else y_true

    # fast_pred_var tells GPyTorch "We don't need the full covariance matrix"
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        output = torch_model(X_torch)
        pred_dist = likelihood(output)

        # --- CRITICAL FIX ---
        # 1. pred_dist.mean     -> Shape (N,)
        # 2. pred_dist.variance -> Shape (N,)  <-- This is the magic.
        # Accessing .variance on a MultivariateNormal computes ONLY the diagonal.
        means = pred_dist.mean
        vars = pred_dist.variance

        # 3. Create N independent Gaussian distributions
        normal = torch.distributions.Normal(means, vars.sqrt())

        # 4. Compute log prob for each point individually
        log_prob = normal.log_prob(y_torch)

        # 5. Return the negative average
        nlpd = -log_prob.mean().item()
        return nlpd


# =============================================================================
# 3. Test Suite
# =============================================================================

class SimpleGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def run_tests():
    print("=== Running NLPD Diagnostics ===")

    # ---------------------------------------------------------
    # Setup Data
    # ---------------------------------------------------------
    train_x = torch.linspace(0, 1, 50)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(50) * 0.1

    # Validation Set
    test_x = torch.linspace(0, 1, 50)
    test_y = torch.sin(test_x * (2 * math.pi))  # Ground truth

    # Initialize Model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SimpleGP(train_x, train_y, likelihood)
    model.eval()
    likelihood.eval()

    # ---------------------------------------------------------
    # Test 1: Standard Accuracy Comparison
    # ---------------------------------------------------------
    print(f"\n[Test 1] Standard Data (N={len(test_x)})")

    nlpd_joint = _calc_nlpd_joint_original(model, test_x, test_y, likelihood)
    nlpd_marginal = _calc_nlpd_marginal_fixed(model, test_x, test_y, likelihood)

    print(f"  Joint NLPD (Original):  {nlpd_joint:.4f}")
    print(f"  Marginal NLPD (Fixed):  {nlpd_marginal:.4f}")
    print("  -> Note: These differ because Joint accounts for correlations between test points.")

    # ---------------------------------------------------------
    # Test 2: Stress Test (Collinear / Singular Matrix)
    # ---------------------------------------------------------
    print(f"\n[Test 2] Stress Test (200 Identical Inputs)")
    print("  -> Creating 200 identical points. Covariance matrix will be singular (rank 1).")

    bad_x = torch.zeros(200)
    bad_y = torch.zeros(200)

    # Attempt Original
    nlpd_joint_stress = _calc_nlpd_joint_original(model, bad_x, bad_y, likelihood)
    print(f"  Original (Joint): {nlpd_joint_stress} (Expected: NaN or Crash)")

    # Attempt Fixed
    nlpd_marginal_stress = _calc_nlpd_marginal_fixed(model, bad_x, bad_y, likelihood)
    print(f"  Fixed (Marginal): {nlpd_marginal_stress:.4f} (Expected: Valid Number)")

    # ---------------------------------------------------------
    # Conclusion
    # ---------------------------------------------------------
    if np.isnan(nlpd_joint_stress) and not np.isnan(nlpd_marginal_stress):
        print("\n[SUCCESS] The fixed Marginal implementation is robust to singular matrices.")
    elif not np.isnan(nlpd_marginal_stress):
        print("\n[SUCCESS] The fixed Marginal implementation works (Joint method unexpectedly survived).")
    else:
        print("\n[FAIL] The fixed implementation returned NaN.")


if __name__ == "__main__":
    run_tests()