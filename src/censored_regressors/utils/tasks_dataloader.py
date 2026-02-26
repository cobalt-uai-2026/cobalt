import abc
import os
import numpy as np
import requests
from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import load_diabetes

def download_url(url, store_directory, save_name=None, messages=True):
    """
    Robust downloader that replaces GPy's version.
    Args:
        url: Web URL of the file.
        store_directory: Local folder to save into.
        save_name: (Optional) Force a specific filename.
        messages: (Optional) Dummy argument to maintain compatibility with your existing calls.
    """
    if not os.path.exists(store_directory):
        os.makedirs(store_directory)

    # Determine filename
    if save_name:
        filename = save_name
    else:
        filename = os.path.basename(url)

    file_path = os.path.join(store_directory, filename)

    # Check if file already exists
    if os.path.exists(file_path):
        if messages:
            print(f"File already exists: {file_path}")
        return

    if messages:
        print(f"Downloading {filename}...")

    # Download
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        with open(file_path, 'wb') as f:
            f.write(response.content)

        if messages:
            print(f"Download complete: {file_path}")

    except Exception as e:
        print(f"Download failed: {e}")
        # Clean up partial file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

class RegressionTask(object):
    """
    Base class for Regression Tasks (Datasets).
    Handles loading, censoring simulation, and cross-validation splitting.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, datapath='./data', lower_bound=None, upper_bound=None, is_natural=False):
        """
        Args:
            datapath (str): Path to store/load datasets.
            lower_bound (float): Threshold for Left Censoring.
            upper_bound (float): Threshold for Right Censoring.
            is_natural (bool):
                If True: Assumes data is already censored (Dataset is naturally bounded).
                If False: Simulates censoring by clipping Y.
        """
        self.datapath = datapath
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_natural = is_natural

        self.X = None
        self.Y_obs = None  # Observed values (clipped if not natural)
        self.Y_true = None  # Ground truth (uncensored)
        self.C = None  # Censoring indicator: 0=Observed, 1=Right, -1=Left

    @abc.abstractmethod
    def load_data(self):
        """Implement this in subclasses to populate X, Y_obs, C, and Y_true."""
        pass

    def _process_censoring(self, Y_raw):
        """
        Generates censoring mask and observation values based on bounds.
        Returns: (Y_obs, C, Y_true)
        """
        Y_raw = Y_raw.reshape(-1, 1)
        C = np.zeros_like(Y_raw, dtype=int)
        Y_obs = Y_raw.copy()

        # --- 1. Right Censoring ---
        if self.upper_bound is not None:
            mask_upper = (Y_raw >= self.upper_bound)
            C[mask_upper] = 1

            # If synthetic/simulated, we clip the observed value to the bound
            if not self.is_natural:
                Y_obs[mask_upper] = self.upper_bound

        # --- 2. Left Censoring ---
        if self.lower_bound is not None:
            mask_lower = (Y_raw <= self.lower_bound)
            C[mask_lower] = -1

            if not self.is_natural:
                Y_obs[mask_lower] = self.lower_bound

        return Y_obs, C, Y_raw

    def _pack_tuple(self, indices):
        """Helper to return the correct data tuple for a set of indices."""
        X_sub = self.X[indices]
        Y_obs_sub = self.Y_obs[indices]
        C_sub = self.C[indices]

        if self.is_natural:
            # Natural data: We don't have ground truth for censored points
            return (X_sub, Y_obs_sub, C_sub)
        else:
            # Simulated data: We retain ground truth for evaluation
            Y_true_sub = self.Y_true[indices]
            return (X_sub, Y_obs_sub, C_sub, Y_true_sub)

    def get_cv_folds(self, n_folds=5, shuffle=True, seed=42):
        """Returns a list of (train_data, test_data) tuples."""
        if self.X is None: raise ValueError("Data not loaded. Call load_data() first.")

        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
        folds = []

        for train_idx, test_idx in kf.split(self.X):
            train_data = self._pack_tuple(train_idx)
            test_data = self._pack_tuple(test_idx)
            folds.append((train_data, test_data))

        return folds

    def get_train_test_split(self, test_size=0.2, shuffle=True, seed=42):
        """Returns (train_data, test_data)."""
        if self.X is None: raise ValueError("Data not loaded. Call load_data() first.")

        indices = np.arange(len(self.X))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, shuffle=shuffle, random_state=seed)

        return (self._pack_tuple(train_idx), self._pack_tuple(test_idx))


# --- Concrete Implementations ---

class Housing(RegressionTask):
    name = 'Housing'
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    filename = 'housing.data'

    def load_data(self):
        # Ensure we are using absolute paths to avoid notebook relative path issues
        # (Optional but recommended if running from different notebook folders)
        if not os.path.isabs(self.datapath):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Adjust '..' if your code is in src/utils and data is in root
            # For now, we trust self.datapath relative to CWD
            pass

        full_path = os.path.join(self.datapath, self.filename)

        # Force download check using our LOCAL function, not GPy's
        if not os.path.exists(full_path):
            download_url(self.url, self.datapath, self.filename)

        # Double check existence to prevent confusing error messages
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Download seemingly succeeded, but {full_path} is missing.")

        data = np.loadtxt(full_path)

        self.X = data[:, :-1]
        Y_raw = data[:, -1:]

        self.Y_obs, self.C, self.Y_true = self._process_censoring(Y_raw)
        return True

class SyntheticTask(RegressionTask):
    name = 'SyntheticData'

    def __init__(self, generator, datapath='./data', lower_bound=None, upper_bound=None, is_natural=False):
        super().__init__(datapath, lower_bound, upper_bound, is_natural)
        self.generator = generator

    def load_data(self):
        # 1. Generate Data
        x_raw, y_obs_raw, c_raw, y_true_raw = self.generator.generate()

        # 2. Reshape to (N, 1) matrices
        self.X = x_raw.reshape(-1, 1)
        self.Y_obs = y_obs_raw.reshape(-1, 1)
        self.C = c_raw.reshape(-1, 1)
        self.Y_true = y_true_raw.reshape(-1, 1)

        print(f"Loaded {len(self.X)} synthetic samples.")
        print(f"Censoring Rate: {np.mean(self.C != 0):.2%}")
        return True


class Housing(RegressionTask):
    name = 'Housing'
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    filename = 'housing.data'

    def load_data(self):
        full_path = os.path.join(self.datapath, self.filename)

        if not os.path.exists(full_path):
            print(f"Downloading {self.name} dataset...")
            download_url(self.url, self.datapath, messages=True)

        data = np.loadtxt(full_path)

        # Features are all cols except last; Target is last col
        self.X = data[:, :-1]
        Y_raw = data[:, -1:]

        # Apply censoring logic
        self.Y_obs, self.C, self.Y_true = self._process_censoring(Y_raw)
        return True


class WineQuality(RegressionTask):
    name = 'WineQuality'
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    filename = 'winequality-red.csv'

    def load_data(self, target_idx=-1):
        full_path = os.path.join(self.datapath, self.filename)

        if not os.path.exists(full_path):
            print(f"Downloading {self.name} dataset...")
            download_url(self.url, self.datapath, messages=True)

        # Load CSV (Wine quality uses semi-colon delimiter)
        data = np.loadtxt(full_path, skiprows=1, delimiter=';')

        # Handle target extraction flexibly
        n_cols = data.shape[1]
        if np.isscalar(target_idx):
            t_idx = target_idx if target_idx >= 0 else n_cols + target_idx
            target_indices = [t_idx]
        else:
            target_indices = [i if i >= 0 else n_cols + i for i in target_idx]

        Y_raw = data[:, target_indices]
        self.X = np.delete(data, target_indices, axis=1)

        self.Y_obs, self.C, self.Y_true = self._process_censoring(Y_raw)
        return True


class Diabetes(RegressionTask):
    name = 'Diabetes'

    def load_data(self, target_idx=-1):
        # Load from sklearn
        diabetes = load_diabetes()

        # X is (442, 10), y is (442,)
        X_base = diabetes.data
        y_base = diabetes.target.reshape(-1, 1)

        # If target_idx is -1 (default), we just use the standard target
        if target_idx == -1 or target_idx == 10:
            self.X = X_base
            Y_raw = y_base
        else:
            # If user wants to treat a feature as the target (e.g. BMI), we must swap
            data_combined = np.hstack((X_base, y_base))

            # Resolve index
            n_cols = data_combined.shape[1]
            if np.isscalar(target_idx):
                t_idx = target_idx if target_idx >= 0 else n_cols + target_idx
                target_indices = [t_idx]
            else:
                target_indices = [i if i >= 0 else n_cols + i for i in target_idx]

            Y_raw = data_combined[:, target_indices]
            self.X = np.delete(data_combined, target_indices, axis=1)

        self.Y_obs, self.C, self.Y_true = self._process_censoring(Y_raw)
        return True