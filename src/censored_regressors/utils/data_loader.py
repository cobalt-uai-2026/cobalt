import os
import pandas as pd
import numpy as np
import importlib.resources as pkg_resources


def load_csv_dataset(filename: str, target_col=None) -> tuple:
    """
    Loads a dataset from the package or local development path.

    Args:
        filename (str): Name of the file (e.g. 'housing.csv').
        target_col (str or int, optional):
            If provided, the function splits the data into (X, y).
            - Can be a string name of the column.
            - Can be an integer index (e.g. -1 for the last column).

    Returns:
        (X, y): Tuple of numpy arrays if target_col is provided.
        df: Pandas DataFrame if target_col is None.
    """

    # 1. Attempt to load from installed package resources
    # This works if you installed via pip (even as a zip)
    try:
        # Check if the resource exists within the package scope
        ref = pkg_resources.files('censored_regressors.data') / filename
        with pkg_resources.as_file(ref) as path:
            if not path.exists():
                raise FileNotFoundError
            df = pd.read_csv(path)

    except (ImportError, FileNotFoundError):
        # 2. Fallback: Local Development Path (Relative to this script)
        # Assumes structure: src/censored_regressors/utils/data_loader.py
        # We need to go up to: censored_regressors/data/

        # Current dir: src/censored_regressors/utils
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up two levels to src/censored_regressors, then into data
        data_dir = os.path.join(current_dir, '..', 'data')
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            # Try project root (common in some IDE configurations)
            project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
            file_path = os.path.join(project_root, 'data', filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find '{filename}' in package resources or at '{file_path}'")

        df = pd.read_csv(file_path)

    # 3. Handle Splitting (X, y)
    if target_col is not None:
        # Handle Integer Index (e.g. -1 for last column)
        if isinstance(target_col, int):
            target_col_name = df.columns[target_col]
        else:
            target_col_name = target_col

        if target_col_name not in df.columns:
            raise ValueError(f"Target column '{target_col_name}' not found in dataset columns: {df.columns.tolist()}")

        # Extract Target (y)
        y = df[target_col_name].to_numpy()

        # Extract Features (X) - Drop target and convert to numpy (strips index)
        X = df.drop(columns=[target_col_name]).to_numpy()

        return X, y

    # Return raw dataframe if no split requested
    return df