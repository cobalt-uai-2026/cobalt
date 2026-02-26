import torch

__all__ = ['print_module', 'apply_module']


def rgetattr(o, k_list):
    for k in k_list:
        o = getattr(o, k)
    return o


def apply_module(func, module):
    with torch.no_grad():
        for name, _ in module.named_parameters():
            name = name.replace('raw_', '')
            param = rgetattr(module, name.split('.'))
            func(name=name, param=param)

def print_module(module):
    apply_module(lambda name, param: print(f'{name:35} {tuple(param.shape)}\n{param.numpy().round(10)}'), module)


import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_gradients(x_range, func_handle, label="Function Value"):
    """
    Args:
        func_handle: The Python function to evaluate (e.g., normal_log_cdf_approx)
        x_range: A torch tensor of input values (e.g., torch.linspace(...))
        label: Title for the plot
    """

    # 1. Setup X with gradient tracking
    # We clone and detach to ensure we have a clean leaf tensor
    test_values = x_range.clone().detach().requires_grad_(True)

    # 2. Run the Forward Pass INSIDE this context
    # This builds the graph linking test_values -> output
    output = func_handle(test_values)

    # 3. Compute First Derivative
    grad_output_ones = torch.ones_like(output)
    first_derivative = torch.autograd.grad(
        outputs=output,
        inputs=test_values,
        grad_outputs=grad_output_ones,
        create_graph=True  # Crucial for second derivative
    )[0]

    # 4. Compute Second Derivative
    second_derivative = torch.autograd.grad(
        outputs=first_derivative,
        inputs=test_values,
        grad_outputs=grad_output_ones,
        create_graph=False  # No need for graph here unless you want 3rd derivative
    )[0]

    # 5. Plotting
    C1_PY, C2_PY = -5.5, -4.5  # Hardcoded bounds from your logic

    # Convert to numpy for matplotlib (must detach first)
    x_np = test_values.detach().numpy()
    y_np = output.detach().numpy()
    y_prime_np = first_derivative.detach().numpy()
    y_double_np = second_derivative.detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Function Value
    axes[0].plot(x_np, y_np)
    axes[0].set_title(label)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$f(x)$")

    # Plot 2: First Derivative
    axes[1].plot(x_np, y_prime_np)
    axes[1].set_title("First Derivative: $f'(x)$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$f'(x)$")

    # Plot 3: Second Derivative
    axes[2].plot(x_np, y_double_np)
    axes[2].set_title("Second Derivative: $f''(x)$")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$f''(x)$")

    # Add vertical lines and formatting
    for ax in axes:
        ax.axvline(C1_PY, color='r', linestyle='--', label=f'C1 = {C1_PY}')
        ax.axvline(C2_PY, color='g', linestyle='--', label=f'C2 = {C2_PY}')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.show()
