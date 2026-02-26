# Copyright (c) 2013, the GPy Authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPy.core import GP
from GPy import likelihoods

from GPy import kern
import numpy as np
from GPy.inference.latent_function_inference.expectation_propagation import EP
from censored_regressors.likelihoods.censored_likelihood_gpy import CensoredGaussian
from censored_regressors.latent_inference.ep_gpy import EPCensored

class GPCensoredRegression(GP):
    """
    Gaussian Process Censored Regression
    """

    def __init__(self, X, Y, censoring, kernel=None, Y_metadata=None, mean_function=None, inference_method=None,
                 likelihood=None, normalizer=False):

        # 1. Ensure self.censoring is 1D (Best for your custom EP loop)
        self.censoring = np.asarray(censoring).flatten()

        # 2. Pack censoring into Y_metadata as 2D (Critical for GPy slicing)
        # GPy's ep_gradients slices metadata as [i, :], so it must be (N, 1)
        if Y_metadata is None:
            Y_metadata = {}
        Y_metadata['censoring'] = self.censoring.reshape(-1, 1)

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        if likelihood is None:
            # Pass the 1D censoring to the likelihood init
            likelihood = CensoredGaussian(censoring=self.censoring)

        if inference_method is None:
            inference_method = EPCensored()
            print("defaulting to " + str(inference_method) + " for latent function inference")

        self.inference_method = inference_method

        GP.__init__(self, X=X, Y=Y, kernel=kernel, likelihood=likelihood,
                    inference_method=inference_method,
                    mean_function=mean_function,
                    name='gp_censored_regression',
                    normalizer=normalizer,
                    Y_metadata=Y_metadata)

    @staticmethod
    def from_gp(gp):
        from copy import deepcopy
        gp = deepcopy(gp)
        GPCensoredRegression(gp.X, gp.Y, gp.cens, gp.kern, gp.likelihood, gp.inference_method,
                             gp.mean_function, name='gp_censored_regression')

    def to_dict(self, save_data=True):
        model_dict = super(GPCensoredRegression, self).to_dict(save_data)
        model_dict["class"] = "GPy.models.GPCensoredRegression"
        return model_dict

    @staticmethod
    def from_dict(input_dict, data=None):
        import GPy
        m = GPy.core.model.Model.from_dict(input_dict, data)
        return GPCensoredRegression.from_gp(m)

    def save_model(self, output_filename, compress=True, save_data=True):
        self._save_model(output_filename, compress=True, save_data=True)

    @staticmethod
    def _build_from_input_dict(input_dict, data=None):
        input_dict = GPCensoredRegression._format_input_dict(input_dict, data)
        input_dict.pop('name', None)
        return GPCensoredRegression(**input_dict)

    def parameters_changed(self):
        """
        Called when parameters change. Re-runs inference.
        """
        if isinstance(self.inference_method, EPCensored):
            # Pass self.censoring (1D) explicitly to your custom EP
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(
                self.kern, self.X, self.likelihood, self.Y_normalized,
                self.censoring,
                self.mean_function, self.Y_metadata
            )
        else:
            # Standard GPy Inference (Laplace, Exact, etc.) uses Y_metadata
            self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(
                self.kern, self.X, self.likelihood, self.Y_normalized,
                self.mean_function, self.Y_metadata
            )

        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.update_gradients_full(self.grad_dict['dL_dK'], self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)