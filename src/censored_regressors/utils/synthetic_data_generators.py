import numpy as np
import copy


class SineWaveGenerator:
    def __init__(self, n_samples=100, seed=10, noise_scale=0.1):
        self.n_samples = n_samples
        self.seed = seed
        self.noise_scale = noise_scale

    def generate(self):
        """
        Returns:
            X: Input features
            y_obs: The 'observed' values (containing censored data where applicable)
            censoring: The boolean mask (1 if censored, 0 if observed)
            y_true: The ground truth noiseless function (for validation)
        """
        np.random.seed(self.seed)

        # 1. Define underlying function
        x = np.linspace(0, 10, self.n_samples)
        y_true = 0.5 * np.sin(2 * x) + 2 + x / 10

        # 2. Generate noisy observations (Ground Truth + Noise)
        y_obs_raw = y_true + np.random.normal(loc=0, scale=self.noise_scale, size=x.shape[0])

        # 3. Apply Censoring Logic
        # Your specific logic: if function value >= 2, we might censor it
        censoring_mask = np.int32(0.5 * np.sin(2 * x) + 2 >= 2)

        # Create the final observed vector (initially valid)
        y_obs_final = copy.deepcopy(y_obs_raw)

        # Calculate random reduction for censored points
        n_censored = np.sum(censoring_mask == 1)
        if n_censored > 0:
            p_c = np.random.uniform(low=0.2, high=0.3, size=n_censored)
            # Apply reduction: Observed = True * (1 - random_drop)
            y_obs_final[censoring_mask == 1] = y_obs_raw[censoring_mask == 1] * (1 - p_c)

        return x, y_obs_final, censoring_mask, y_true