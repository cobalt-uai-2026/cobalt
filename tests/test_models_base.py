import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import gpytorch
import GPy

# Robust import logic
try:
    from censored_regressors.models.models_base import (
        RegressionMethod,
        BaseGPyModel,
        BaseGPyTorchModel,
        GPyTorchOptimizer
    )
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.censored_regressors.models.models_base import (
        RegressionMethod,
        BaseGPyModel,
        BaseGPyTorchModel,
        GPyTorchOptimizer
    )


# --- Concrete Helpers ---
class ConcreteRegression(RegressionMethod):
    def _fit(self, train_data, **kwargs): return "fitted"

    def _predict(self, test_data): return {'f_mean': test_data}


class ConcreteGPyModel(BaseGPyModel):
    def _fit(self, train_data, **kwargs): pass

    def _predict(self, test_data): return {}


class ConcreteGPyTorchModel(BaseGPyTorchModel):
    def _fit(self, train_data, **kwargs): pass

    def _predict(self, test_data): return {}


class MockGP(gpytorch.models.ApproximateGP):
    """Satisfies isinstance(model, GP) check in VariationalELBO."""

    def __init__(self):
        dist = gpytorch.variational.CholeskyVariationalDistribution(10)
        strat = gpytorch.variational.VariationalStrategy(self, torch.randn(10, 1), dist, learn_inducing_locations=False)
        super().__init__(strat)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


# --- TESTS ---

class TestRegressionMethod(unittest.TestCase):
    def setUp(self):
        self.model = ConcreteRegression()
        self.X = np.array([[1.0], [2.0], [3.0]])
        self.Y = np.array([[10.0], [20.0], [30.0]])
        self.censoring = np.array([0, 1, 0])

    def test_preprocess_train_xy(self):
        X_n, Y_n = self.model._preprocess((self.X, self.Y), train=True)
        self.assertTrue(np.allclose(X_n.mean(), 0.0, atol=1e-5))

    def test_preprocess_train_xy_censored(self):
        X_n, Y_n, C_n = self.model._preprocess((self.X, self.Y, self.censoring), train=True)
        self.assertTrue(np.array_equal(C_n, self.censoring))

    def test_reverse_trans_labels_dict(self):
        self.model._preprocess((self.X, self.Y), train=True)
        dummy_res = {'f_mean': np.array([[0.0]]), 'f_var': np.array([[1.0]])}
        unscaled = self.model._reverse_trans_labels(dummy_res)
        self.assertAlmostEqual(unscaled['f_mean'][0, 0], 20.0, places=4)


class TestBaseGPyModel(unittest.TestCase):
    def setUp(self):
        self.model_wrapper = ConcreteGPyModel(kernel_type='rbf')
        self.model_wrapper.model = MagicMock()

    def test_get_kernel_types(self):
        k = self.model_wrapper._get_kernel(input_dim=1)
        self.assertIsInstance(k, GPy.kern.RBF)

    def test_apply_init_params(self):
        self.model_wrapper.model.likelihood.variance = MagicMock()
        self.model_wrapper.model.kern = MagicMock()
        self.model_wrapper._apply_init_params({'noise': 0.5})
        self.model_wrapper.model.likelihood.variance.__setitem__.assert_called()


class TestBaseGPyTorchModel(unittest.TestCase):
    def setUp(self):
        self.wrapper = ConcreteGPyTorchModel(kernel_type='rbf')
        self.wrapper.model = MagicMock()
        self.wrapper.likelihood = MagicMock()

    def test_get_kernel_factory(self):
        k = self.wrapper._get_kernel(input_dim=2)
        self.assertIsInstance(k, gpytorch.kernels.ScaleKernel)

    def test_apply_init_simple(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.wrapper.model = MockModule()
        self.wrapper.likelihood = MagicMock()
        self.wrapper._apply_init({'outputscale': 5.0, 'noise': 0.1})
        self.assertEqual(self.wrapper.model.covar.outputscale.item(), 5.0)


class TestGPyTorchOptimizer(unittest.TestCase):
    def setUp(self):
        self.model = MockGP()
        self.likelihood = MagicMock(spec=gpytorch.likelihoods.Likelihood)
        self.x = torch.randn(10, 1)
        self.y = torch.randn(10, 1)
        self.optimizer_wrapper = GPyTorchOptimizer(self.model, self.likelihood, self.x, self.y)
        self.optimizer_wrapper.mll = MagicMock(return_value=torch.tensor(1.0, requires_grad=True))

    def test_train_ngd_logic(self):
        """Verify NGD splits parameters correctly."""

        # FIX: Patch NGD in the specific module where it is imported/used
        module_name = GPyTorchOptimizer.__module__

        # We use a context manager to ensure patches are applied to the correct namespace
        with patch(f'{module_name}.NGD') as mock_ngd, \
                patch('torch.optim.Adam') as mock_adam, \
                patch('torch.optim.lr_scheduler.ReduceLROnPlateau') as mock_scheduler:
            # Run one iteration
            self.optimizer_wrapper.train(optimizer_name='ngd', max_iters=1)

            # Verify NGD was called
            mock_ngd.assert_called()

            # Verify Adam was called
            mock_adam.assert_called()

            # Verify Scheduler was called
            mock_scheduler.assert_called()

    def test_train_invalid_optimizer(self):
        with self.assertRaises(ValueError):
            self.optimizer_wrapper.train(optimizer_name='invalid_opt')


if __name__ == '__main__':
    unittest.main()