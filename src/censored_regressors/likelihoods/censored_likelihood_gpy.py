import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.core.parameterization import Param
from paramz.transformations import Logexp

__all__ = ["CensoredGaussian"]
# --- ROBUST HELPERS (SCIPY) ---
from scipy.special import log_ndtr


def logCdfNormal(z):
    """Robust Log CDF using Scipy."""
    return log_ndtr(z)


def derivLogCdfNormal(z):
    """Robust Derivative of Log CDF (Inverse Mills Ratio)."""
    # exp(log_pdf - log_cdf) is stable
    log_pdf = -0.5 * np.log(2 * np.pi) - 0.5 * z ** 2
    log_cdf = log_ndtr(z)
    return np.exp(log_pdf - log_cdf)


# ------------------------------

class CensoredGaussian(Likelihood):
    """
    Censored Gaussian likelihood
    Supports Uncensored (0), Right-Censored (1), and Left-Censored (-1).
    """

    def __init__(self, gp_link=None, variance=1., censoring=None, name='CensoredGaussian'):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(CensoredGaussian, self).__init__(gp_link, name=name)

        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)
        self.censoring = censoring

        if isinstance(gp_link, link_functions.Identity):
            self.log_concave = True

    def to_dict(self):
        input_dict = super(CensoredGaussian, self)._save_to_input_dict()
        input_dict["class"] = "GPy.likelihoods.CensoredGaussian"
        input_dict["variance"] = self.variance.values.tolist()
        return input_dict

    def _preprocess_values(self, Y, censoring):
        return Y.copy()

    def _get_censoring(self, Y_metadata):
        if Y_metadata is not None and 'censoring' in Y_metadata:
            c = Y_metadata['censoring']
            if np.ndim(c) > 0 and c.size == 1:
                return c.item()
            return c
        return self.censoring

    def update_gradients(self, gradients):
        self.variance.gradient = np.sum(gradients)

    def ep_gradients(self, Y, cav_tau, cav_v, dL_dKdiag, Y_metadata=None, quad_mode='gh'):
            c = self._get_censoring(Y_metadata)
            c = np.asarray(c).flatten()
            Y = np.asarray(Y).flatten()

            mu_cav = cav_v / cav_tau
            sigma2_cav = 1.0 / cav_tau
            total_var = self.variance.values[0] + sigma2_cav
            sigma_tot = np.sqrt(total_var)

            grad_var = np.zeros_like(Y)

            mask_0 = (c == 0)
            if np.any(mask_0):
                e = Y[mask_0] - mu_cav[mask_0]
                grad_var[mask_0] = -0.5 / total_var[mask_0] + 0.5 * (e ** 2) / (total_var[mask_0] ** 2)

            mask_1 = (c == 1)
            if np.any(mask_1):
                z = (mu_cav[mask_1] - Y[mask_1]) / sigma_tot[mask_1]
                r = derivLogCdfNormal(z)
                grad_var[mask_1] = -0.5 * r * z / total_var[mask_1]

            mask_neg1 = (c == -1)
            if np.any(mask_neg1):
                z = (Y[mask_neg1] - mu_cav[mask_neg1]) / sigma_tot[mask_neg1]
                r = derivLogCdfNormal(z)
                grad_var[mask_neg1] = -0.5 * r * z / total_var[mask_neg1]

            return np.atleast_2d(grad_var).T

    def log_predictive_density(self, y_test, mu_star, var_star, Y_metadata=None):
        sigma2 = var_star + self.variance
        sigma = np.sqrt(sigma2)
        c = self._get_censoring(Y_metadata)

        if np.ndim(c) > 0 and c.shape != y_test.shape:
            if c.size == y_test.size:
                c = c.reshape(y_test.shape)

        res = np.zeros_like(y_test)

        mask_0 = (c == 0)
        if np.any(mask_0):
            res[mask_0] = -0.5 * np.log(2 * np.pi * sigma2[mask_0]) \
                          - 0.5 * ((y_test[mask_0] - mu_star[mask_0]) ** 2) / sigma2[mask_0]

        mask_1 = (c == 1)
        if np.any(mask_1):
            z = (mu_star[mask_1] - y_test[mask_1]) / sigma[mask_1]
            res[mask_1] = logCdfNormal(z)

        mask_neg1 = (c == -1)
        if np.any(mask_neg1):
            z = (y_test[mask_neg1] - mu_star[mask_neg1]) / sigma[mask_neg1]
            res[mask_neg1] = logCdfNormal(z)

        return res

        # --- MOMENTS (Fixed Underflow) ---
    def moments_match_ep(self, Y_i, tau_i, v_i, censoring_i, Y_metadata_i=None):
            variance = self.variance.values[0]
            sigma2_cav = 1. / tau_i
            mu_cav = v_i / tau_i
            total_var = variance + sigma2_cav
            sigma_tot = np.sqrt(total_var)

            if np.ndim(censoring_i) > 0:
                censoring_i = censoring_i.item()

            if censoring_i == 0:
                # RETURN LOG Z directly to avoid underflow
                log_Z_hat = -0.5 * np.log(2 * np.pi * total_var) - 0.5 * ((Y_i - mu_cav) ** 2) / total_var

                denom = variance + sigma2_cav
                mu_hat = (variance * mu_cav + sigma2_cav * Y_i) / denom
                sigma2_hat = (variance * sigma2_cav) / denom
                return log_Z_hat, mu_hat, sigma2_hat

            # Censored Cases
            if censoring_i == 1:  # Right
                z = (mu_cav - Y_i) / sigma_tot
                sign = 1.0
            elif censoring_i == -1:  # Left
                z = (Y_i - mu_cav) / sigma_tot
                sign = -1.0
            else:
                return 0.0, mu_cav, sigma2_cav

            log_Z_hat = logCdfNormal(z)
            r = derivLogCdfNormal(z)

            shift = (sigma2_cav / sigma_tot) * sign * r
            mu_hat = mu_cav + shift

            delta = r * (r + z)
            rho = 1.0 - (sigma2_cav / total_var) * delta
            if rho < 1e-6: rho = 1e-6
            sigma2_hat = sigma2_cav * rho

            # RETURN LOG Z directly
            return log_Z_hat, mu_hat, sigma2_hat

        # --- EP GRADIENTS (Exact Analytical closed-form, no Quadrature!) ---

    # --- DERIVATIVES ---
    def logpdf_link(self, link_f, y, Y_metadata=None):
        c = self._get_censoring(Y_metadata)
        sigma = np.sqrt(self.variance)

        # Scalar wrapper
        if np.size(c) == 1:
            c_scalar = c.item() if np.ndim(c) > 0 else c
            if c_scalar == 0:
                return -0.5 * np.log(2 * np.pi * self.variance) - 0.5 * ((y - link_f) ** 2) / self.variance
            elif c_scalar == 1:
                return logCdfNormal((link_f - y) / sigma)
            elif c_scalar == -1:
                return logCdfNormal((y - link_f) / sigma)

        input_shape = link_f.shape
        log_pdf = np.zeros(link_f.size)
        c = np.asarray(c).flatten()
        y = np.asarray(y).flatten()
        f_flat = np.asarray(link_f).flatten()

        mask_0 = (c == 0)
        if np.any(mask_0):
            log_pdf[mask_0] = -0.5 * np.log(2 * np.pi * self.variance) - 0.5 * (
                        (y[mask_0] - f_flat[mask_0]) ** 2) / self.variance
        mask_1 = (c == 1)
        if np.any(mask_1):
            log_pdf[mask_1] = logCdfNormal((f_flat[mask_1] - y[mask_1]) / sigma)
        mask_neg1 = (c == -1)
        if np.any(mask_neg1):
            log_pdf[mask_neg1] = logCdfNormal((y[mask_neg1] - f_flat[mask_neg1]) / sigma)

        return log_pdf.reshape(input_shape)

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        c = self._get_censoring(Y_metadata)
        sigma = np.sqrt(self.variance)

        if np.size(c) == 1:
            c_scalar = c.item() if np.ndim(c) > 0 else c
            if c_scalar == 0:
                return (y - link_f) / self.variance
            elif c_scalar == 1:
                return (1. / sigma) * derivLogCdfNormal((link_f - y) / sigma)
            elif c_scalar == -1:
                return (-1. / sigma) * derivLogCdfNormal((y - link_f) / sigma)

        input_shape = link_f.shape
        grad = np.zeros(link_f.size)
        c = np.asarray(c).flatten()
        y = np.asarray(y).flatten()
        f_flat = np.asarray(link_f).flatten()

        mask_0 = (c == 0)
        if np.any(mask_0): grad[mask_0] = (y[mask_0] - f_flat[mask_0]) / self.variance
        mask_1 = (c == 1)
        if np.any(mask_1):
            grad[mask_1] = (1. / sigma) * derivLogCdfNormal((f_flat[mask_1] - y[mask_1]) / sigma)
        mask_neg1 = (c == -1)
        if np.any(mask_neg1):
            grad[mask_neg1] = (-1. / sigma) * derivLogCdfNormal((y[mask_neg1] - f_flat[mask_neg1]) / sigma)

        return grad.reshape(input_shape)

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        c = self._get_censoring(Y_metadata)
        sigma = np.sqrt(self.variance)

        def censored_hessian(z_vals):
            r = derivLogCdfNormal(z_vals)
            return (1. / self.variance) * (-r * (z_vals + r))

        if np.size(c) == 1:
            c_scalar = c.item() if np.ndim(c) > 0 else c
            if c_scalar == 0:
                return np.full_like(link_f, -1. / self.variance)
            elif c_scalar == 1:
                return censored_hessian((link_f - y) / sigma)
            elif c_scalar == -1:
                return censored_hessian((y - link_f) / sigma)

        input_shape = link_f.shape
        hess = np.zeros(link_f.size)
        c = np.asarray(c).flatten()
        y = np.asarray(y).flatten()
        f_flat = np.asarray(link_f).flatten()

        mask_0 = (c == 0)
        if np.any(mask_0): hess[mask_0] = -1. / self.variance
        mask_1 = (c == 1)
        if np.any(mask_1):
            hess[mask_1] = censored_hessian((f_flat[mask_1] - y[mask_1]) / sigma)
        mask_neg1 = (c == -1)
        if np.any(mask_neg1):
            hess[mask_neg1] = censored_hessian((y[mask_neg1] - f_flat[mask_neg1]) / sigma)

        return hess.reshape(input_shape)

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        c = self._get_censoring(Y_metadata)
        sigma = np.sqrt(self.variance)

        def censored_d3(z_vals, sign_flip):
            r = derivLogCdfNormal(z_vals)
            poly = r * (2 * (r ** 2) + 3 * z_vals * r + (z_vals ** 2 - 1.0))
            return (sign_flip ** 3 / (sigma ** 3)) * poly

        if np.size(c) == 1:
            c_scalar = c.item() if np.ndim(c) > 0 else c
            if c_scalar == 0:
                return np.zeros_like(link_f)
            elif c_scalar == 1:
                return censored_d3((link_f - y) / sigma, 1.0)
            elif c_scalar == -1:
                return censored_d3((y - link_f) / sigma, -1.0)

        input_shape = link_f.shape
        d3 = np.zeros(link_f.size)
        c = np.asarray(c).flatten()
        y = np.asarray(y).flatten()
        f_flat = np.asarray(link_f).flatten()

        mask_1 = (c == 1)
        if np.any(mask_1):
            d3[mask_1] = censored_d3((f_flat[mask_1] - y[mask_1]) / sigma, 1.0)
        mask_neg1 = (c == -1)
        if np.any(mask_neg1):
            d3[mask_neg1] = censored_d3((y[mask_neg1] - f_flat[mask_neg1]) / sigma, -1.0)

        return d3.reshape(input_shape)

    def dlogpdf_link_dvar(self, link_f, y, Y_metadata=None):
        c = self._get_censoring(Y_metadata)
        sigma = np.sqrt(self.variance)

        def censored_grad(z_val):
            return (-0.5 / self.variance) * z_val * derivLogCdfNormal(z_val)

        if np.size(c) == 1:
            c_scalar = c.item() if np.ndim(c) > 0 else c
            if c_scalar == 0:
                e = y - link_f
                return -0.5 / self.variance + 0.5 * (e ** 2) / (self.variance ** 2)
            elif c_scalar == 1:
                return censored_grad((link_f - y) / sigma)
            elif c_scalar == -1:
                return censored_grad((y - link_f) / sigma)

        input_shape = link_f.shape
        grad_var = np.zeros(link_f.size)
        c = np.asarray(c).flatten()
        y = np.asarray(y).flatten()
        f_flat = np.asarray(link_f).flatten()

        mask_0 = (c == 0)
        if np.any(mask_0):
            e = y[mask_0] - f_flat[mask_0]
            grad_var[mask_0] = -0.5 / self.variance + 0.5 * (e ** 2) / (self.variance ** 2)

        mask_1 = (c == 1)
        if np.any(mask_1):
            grad_var[mask_1] = censored_grad((f_flat[mask_1] - y[mask_1]) / sigma)

        mask_neg1 = (c == -1)
        if np.any(mask_neg1):
            grad_var[mask_neg1] = censored_grad((y[mask_neg1] - f_flat[mask_neg1]) / sigma)

        return grad_var.reshape(input_shape)

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        dlik_dvar = self.dlogpdf_link_dvar(f, y, Y_metadata=Y_metadata)
        return dlik_dvar.reshape(1, *f.shape)

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        c = self._get_censoring(Y_metadata)
        sigma = np.sqrt(self.variance)

        def censored_mixed(z_vals, sign_flip):
            r = derivLogCdfNormal(z_vals)
            return (sign_flip * r / (2 * sigma ** 3)) * (z_vals ** 2 + z_vals * r - 1.0)

        if np.size(c) == 1:
            c_scalar = c.item() if np.ndim(c) > 0 else c
            if c_scalar == 0:
                return -1.0 * (y - f) / (self.variance ** 2)
            elif c_scalar == 1:
                return censored_mixed((f - y) / sigma, 1.0)
            elif c_scalar == -1:
                return censored_mixed((y - f) / sigma, -1.0)

        mixed = np.zeros(f.size)
        c = np.asarray(c).flatten()
        y = np.asarray(y).flatten()
        f_flat = np.asarray(f).flatten()

        mask_0 = (c == 0)
        if np.any(mask_0):
            mixed[mask_0] = -1.0 * (y[mask_0] - f_flat[mask_0]) / (self.variance ** 2)

        mask_1 = (c == 1)
        if np.any(mask_1):
            mixed[mask_1] = censored_mixed((f_flat[mask_1] - y[mask_1]) / sigma, 1.0)

        mask_neg1 = (c == -1)
        if np.any(mask_neg1):
            mixed[mask_neg1] = censored_mixed((y[mask_neg1] - f_flat[mask_neg1]) / sigma, -1.0)

        return mixed.reshape(1, *f.shape)

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        c = self._get_censoring(Y_metadata)
        if np.size(c) == 1:
            c_scalar = c.item() if np.ndim(c) > 0 else c
            if c_scalar == 0:
                return np.full_like(f, 1. / (self.variance ** 2))
            else:
                return np.zeros_like(f)

        mixed2 = np.zeros(f.size)
        c = np.asarray(c).flatten()
        mask_0 = (c == 0)
        if np.any(mask_0):
            mixed2[mask_0] = 1. / (self.variance ** 2)
        return mixed2.reshape(1, *f.shape)

    def predictive_mean(self, mu, variance, Y_metadata=None):
        return mu

    def predictive_variance(self, mu, variance, pred_mean, Y_metadata=None):
        return variance + self.variance