# Censored Regressors

This is a robust library for **Censored Gaussian Process Regression** (Tobit Model). This package provides custom likelihoods and probability distributions to handle left, right, and interval-censored data across major Gaussian Process frameworks.

It is designed to be numerically stable even in extreme tail regions (deep censoring) by using log-space arithmetic (`log_ndtr`) and Gauss-Hermite quadrature for variational integration.

## Features

* **Multi-Backend Support**:
    * **GPyTorch**: Analytic variational likelihood using differentiable Gauss-Hermite quadrature.
    * **Pyro**: Variational likelihood compatible with SVI, using quadrature to reduce gradient variance compared to Monte Carlo.
    * **GPy**: Exact inference support (EP/Laplace) with analytical derivatives and Hessians.
* **Numerically Robust**: Implements "Tobit-style" log-probabilities that clamp gradients in the tails to prevent `NaNs` during training.
* **Active Learning**: Includes **Censored BALD** (Bayesian Active Learning by Disagreement) acquisition function.

## Installation

```bash
git clone [https://github.com/cobalt-uai-2026/cobalt.git](https://github.com/cobalt-uai-2026/cobalt.git)
cd cobolt
pip install -e .
```

## Usage Examples
### 1. GPyTorch

Use CensoredGaussianLikelihoodAnalytic for variational GPs.
```
import torch
import gpytorch
from censored_regressors.likelihoods.censored_likelihood import CensoredGaussianLikelihoodAnalytic

# Define bounds (-inf or inf for open intervals)
# Example: Right-censored at y=1.0
likelihood = CensoredGaussianLikelihoodAnalytic(
    low=None, 
    high=1.0, 
    integration_type='gauss_hermite', 
    n_points=20
)

# In your model forward pass:
# dist = likelihood(f_mean) 
# loss = -mll(dist, y_train)
```

### 2. Pyro

Use CensoredHomoscedGaussian for Pyro's gp module.

```
import torch
from censored_regressors.likelihoods.censored_likelihood_pyro import CensoredHomoscedGaussian

# Initialize likelihood
likelihood = CensoredHomoscedGaussian(
    low=-1.0, 
    high=1.0, 
    num_quad_points=20
)

# Inside your model/guide:
# pyro.sample("obs", likelihood(f_loc, f_var), obs=y)
```

### 3. Bayesian Active Learning (BALD)

Compute information gain scores that account for censoring limits using CensoredBALD.

```
from censored_regressors.active_learning.censored_bald import CensoredBALD

# Wrapper for your trained model
scorer = CensoredBALD(
    model=my_gp_model, 
    noise_std=0.1, 
    low=-1.0, 
    high=1.0
)

# Compute scores for a candidate pool (using robust quadrature)
scores = scorer.get_score(X_pool, method='gauss_hermite')
```

### 4. Testing

The package includes a comprehensive test suite validating gradients, numerical accuracy against SciPy quad, and corner cases (e.g., infinite bounds).
```
# Run all tests
python -m unittest discover tests

# Run specific suite
python -m unittest tests/test_censored_likelihood.py
```

### 5. Project Structure

```
censored_regressors/
├── diagnostics/                # Scripts for checking numerical stability
├── examples/
│   └── notebooks/              # Notebooks demostraing usage of the package
├── src/
│   └── censored_regressors/    # Main package source
│       ├── active_learning/    # Acquisition functions (e.g., CensoredBALD)
│       ├── data/               # Dataset storage
│       ├── distributions/      # Core PyTorch/Pyro distributions (CensoredNormal)
│       ├── latent_inference/   # Latent function inference  (GPy - Censored EP)
│       ├── likelihoods/        # Likelihood implementations (GPyTorch, Pyro, GPy)
│       ├── losses/             # Tobit Loss (Torch)
│       ├── metrics/            # Metrics for experiments evaluation
│       ├── models/             # Models implementations ((GPyTorch, GPy)
│       ├── utils/              # Util functions (Test functions data generators, helpers)
│       └── __init__.py
├── tests/                      # Unit tests for gradients & integration accuracy
├── .gitignore                  # Ignored files (pycache, build artifacts)
├── README.md                   # Project documentation
└── pyproject.toml              # Installation script
```