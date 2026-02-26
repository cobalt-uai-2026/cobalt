import torch
import time
import pandas as pd
import numpy as np
from torch.autograd import grad

from .censored_normal import CensoredNormal
from .integrate_atoms import integrate_log_Phi, normal_log_cdf_scipylike


# [Insert your provided code imports and function definitions here]
# ... (CensoredNormal, integrate_log_Phi, normal_log_cdf_scipylike, etc.) ...
# Ensure the code provided in the prompt is available in the scope or imported.

def run_diagnostics():
    print(f"{'=' * 80}")
    print(f"{'INTEGRATION METHOD DIAGNOSTICS':^80}")
    print(f"{'=' * 80}\n")

    # --- 1. Define Test Cases ---
    test_cases = [
        # (Name, a, b, Description)
        ("Standard", 0.0, 1.0, "Center of distribution"),
        ("Right Tail", 2.0, 1.0, "High probability region"),
        ("Left Tail", -2.0, 1.0, "Low probability region"),
        ("Deep Left", -10.0, 1.0, "Extreme rare event (Log-CDF linear asymptote)"),
        ("Deepest Left", -50.0, 1.0, "Numerical breakdown zone"),
        ("High Var", 0.0, 0.1, "High uncertainty (b small)"),
        ("Low Var", 0.0, 10.0, "Sharp transition (b large)"),
        ("Mixed Tail", -5.0, 5.0, "Deep tail + Sharp transition")
    ]

    # --- 2. Define Methods to Test ---
    methods = [
        # (Name, kwargs)
        ("ScipyQuad (Gold)", {'integration_type': 'scipyquad'}),
        ("GH (n=10)", {'integration_type': 'gauss_hermite', 'n': 10}),
        ("GH (n=20)", {'integration_type': 'gauss_hermite', 'n': 20}),
        ("GH (n=100)", {'integration_type': 'gauss_hermite', 'n': 100}),
        ("Trapez (n=50)", {'integration_type': 'trapez', 'n': 50, 'width': 6.}),
        ("Simpson (n=50)", {'integration_type': 'simpson', 'n': 50, 'width': 6.}),
    ]

    results = []

    for case_name, a_val, b_val, desc in test_cases:
        print(f"Testing Case: {case_name} (a={a_val}, b={b_val})")

        # Prepare Inputs (require grad for stability check)
        a = torch.tensor([a_val], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([b_val], dtype=torch.float32, requires_grad=True)

        # 1. Get Baseline (ScipyQuad)
        try:
            start_t = time.perf_counter()
            baseline_val = integrate_log_Phi(a, b, integration_type='scipyquad')
            baseline_time = (time.perf_counter() - start_t) * 1000  # ms

            # Baseline Gradients
            base_grad = grad(baseline_val, (a, b), create_graph=False)
            base_grad_norm = torch.norm(torch.stack(base_grad)).item()
            baseline_res = baseline_val.item()

        except Exception as e:
            print(f"  [CRITICAL] Baseline failed: {e}")
            baseline_res = np.nan
            continue

        # 2. Test Candidates
        for method_name, kwargs in methods:
            # Reset gradients
            if a.grad is not None: a.grad.zero_()
            if b.grad is not None: b.grad.zero_()

            try:
                # Forward Pass
                t0 = time.perf_counter()
                val = integrate_log_Phi(a, b, **kwargs)
                t1 = time.perf_counter()

                # Backward Pass (Stability Check)
                g = grad(val, (a, b), create_graph=True)
                g_norm = torch.norm(torch.stack(g)).item()

                # Metrics
                val_item = val.item()
                abs_err = abs(val_item - baseline_res)
                rel_err = abs_err / (abs(baseline_res) + 1e-9)
                dur = (t1 - t0) * 1000  # ms

                grad_err = abs(g_norm - base_grad_norm)

                results.append({
                    "Case": case_name,
                    "Method": method_name,
                    "Value": val_item,
                    "Error (Abs)": abs_err,
                    "Time (ms)": dur,
                    "Grad Norm": g_norm,
                    "Grad Error": grad_err
                })

            except Exception as e:
                results.append({
                    "Case": case_name,
                    "Method": method_name,
                    "Value": np.nan,
                    "Error (Abs)": np.nan,
                    "Time (ms)": np.nan,
                    "Grad Norm": np.nan,
                    "Grad Error": np.nan
                })
                print(f"  Method {method_name} failed: {e}")

    # --- 3. Display Results ---
    df = pd.DataFrame(results)

    # Format for display
    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)

    for case_name in df['Case'].unique():
        print(f"\n>>> Case: {case_name}")
        subset = df[df['Case'] == case_name].copy()
        subset = subset[['Method', 'Value', 'Error (Abs)', 'Time (ms)', 'Grad Error']]
        print(subset.to_string(index=False, float_format=lambda x: "{:.2e}".format(x)))


if __name__ == "__main__":
    # Ensure GH cache is populated if needed or just run
    run_diagnostics()