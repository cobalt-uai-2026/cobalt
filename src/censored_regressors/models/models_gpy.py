import numpy as np
import GPy
from GPy.inference.latent_function_inference import Laplace

from censored_regressors.models.censored_model_gpy import GPCensoredRegression
from censored_regressors.likelihoods.censored_likelihood_gpy import CensoredGaussian
from censored_regressors.latent_inference.ep_gpy import EPCensored
from censored_regressors.models.models_base import BaseGPyModel

class GP(BaseGPyModel):
    base_name = 'GP_Gauss'
    def __init__(self, kernel_type='lin_rbf', name='GP_Gauss'):
        super().__init__(kernel_type=kernel_type, name=name)

    def _fit(self, train_data, num_restarts=10, init_params=None):
        if len(train_data) == 3:
            inputs, labels, _ = train_data
        else:
            inputs, labels = train_data

        kernel = self._get_kernel(inputs.shape[-1])
        self.model = GPy.models.GPRegression(inputs, labels, kernel=kernel)

        # Default smart init
        self.model.likelihood.variance[:] = labels.var() * 0.01

        # Apply User Init (Overwrites default if provided)
        self._apply_init_params(init_params)

        # Constraints must be applied AFTER init
        self._apply_constraints()

        # Optimize
        # GPy's optimize_restarts uses the current state as run #1
        self.model.optimize_restarts(num_restarts=num_restarts, optimizer="BFGS", max_iters=1000, verbose=False)
        return True

class TruncGP(BaseGPyModel):
    base_name = 'TruncGP'

    def __init__(self, kernel_type='lin_rbf', name='TruncGP'):
        super().__init__(kernel_type=kernel_type, name=name)

    def _fit(self, train_data, num_restarts=10, init_params=None):
        if len(train_data) != 3:
            raise ValueError("TruncGP requires (X, Y, Censoring) data tuple.")
        inputs_, labels_, censoring = train_data

        # Filter data
        inputs = inputs_[censoring.flatten() == 0]
        labels = labels_[censoring.flatten() == 0]

        kernel = self._get_kernel(inputs.shape[-1])
        self.model = GPy.models.GPRegression(inputs, labels, kernel=kernel)
        self.model.likelihood.variance[:] = labels.var() * 0.01

        self._apply_init_params(init_params)
        self._apply_constraints()

        self.model.optimize_restarts(num_restarts=num_restarts, optimizer="BFGS", max_iters=1000, verbose=False)
        return True


class CensoredGP_Laplace(BaseGPyModel):
    base_name = 'CensoredGP_Laplace'

    def __init__(self, kernel_type='lin_rbf', name='CensoredGP_Laplace'):
        super().__init__(kernel_type=kernel_type, name=name)

    def _fit(self, train_data, num_restarts=10, init_params=None):
        if len(train_data) != 3:
            raise ValueError("CensoredGP requires (X, Y, Censoring) data tuple.")
        inputs, labels, censoring = train_data

        kernel = self._get_kernel(inputs.shape[-1])
        likelihood = CensoredGaussian(censoring=censoring, variance=labels.var() * 0.01)
        inference = Laplace()

        self.model = GPCensoredRegression(
            X=inputs, Y=labels, censoring=censoring,
            kernel=kernel, likelihood=likelihood,
            inference_method=inference
        )

        self._apply_init_params(init_params)
        self._apply_constraints()

        self.model.optimize_restarts(
            num_restarts=num_restarts,
            optimizer="BFGS",
            max_iters=1000,
            verbose=False,
            robust=True
        )
        return True


class CensoredGP_EP(BaseGPyModel):
    base_name = 'CensoredGP_EP'

    def __init__(self, kernel_type='lin_rbf', name='CensoredGP_EP'):
        super().__init__(kernel_type=kernel_type, name=name)

    def _fit(self, train_data, num_restarts=2, init_params=None):
        if len(train_data) != 3:
            raise ValueError("CensoredGP requires (X, Y, Censoring) data tuple.")
        inputs, labels, censoring = train_data

        kernel = self._get_kernel(inputs.shape[-1])
        likelihood = CensoredGaussian(censoring=censoring, variance=labels.var() * 0.01)
        inference = EPCensored()

        self.model = GPCensoredRegression(
            X=inputs, Y=labels, censoring=censoring,
            kernel=kernel, likelihood=likelihood, inference_method=inference
        )

        # Apply Init & Constraints
        self._apply_init_params(init_params)
        self._apply_constraints()

        best_log_likelihood = -np.inf
        best_model_state = None

        for i in range(num_restarts):
            try:
                # A. Reset Inference
                self.model.inference_method = EPCensored()

                # B. Randomize (Only if i > 0)
                if i > 0:
                    self.model.randomize()
                    # Re-apply constraints after randomization
                    self._apply_constraints()

                # Note: For i==0, we retain the state set by _apply_init_params

                # C. Optimize
                #self.model.optimize(optimizer='adam', max_iters=2500, messages=True)
                self.model.optimize(optimizer='bfgs', max_iters=1000, messages=False)

                current_ll = self.model.log_likelihood()
                if current_ll > best_log_likelihood:
                    best_log_likelihood = current_ll
                    best_model_state = self.model.param_array.copy()

            except (np.linalg.LinAlgError, RuntimeWarning) as e:
                continue

        if best_model_state is not None:
            self.model.param_array[:] = best_model_state
            self.model.inference_method = EPCensored()
            self.model.optimize(max_iters=0)
            return True
        else:
            raise RuntimeError("All restarts failed to converge.")
