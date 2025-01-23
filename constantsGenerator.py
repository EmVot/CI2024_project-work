import numpy as np
from typing import Dict, Tuple

EPSILON = 1e-8  #  global constant to avoid zero divisions

def coefficient_scaler(problem: np.ndarray) -> np.ndarray:
    """
    Compute scaling coefficients for each feature based on the standard deviation of y.

    Parameters
    ----------
    problem : np.ndarray
        A dictionary-like object containing:
        - 'x' : np.ndarray of shape (n_features, n_samples)
        - 'y' : np.ndarray of shape (n_samples,)

    Returns
    -------
    np.ndarray
        Array of scaling coefficients for each feature.
    """
    x_values = problem['x']
    y_values = problem['y']

    y_std = np.std(y_values) + EPSILON  # Avoid 0 division
    x_std = np.std(x_values, axis=1) + EPSILON  # Per-feature standard deviation

    return y_std / x_std


def coefficient_range(problem: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Compute a reasonable range of coefficients for each feature.

    Parameters
    ----------
    problem : np.ndarray
        A dictionary-like object containing:
        - 'x' : np.ndarray of shape (n_features, n_samples)
        - 'y' : np.ndarray of shape (n_samples,)

    Returns
    -------
    dict[str, tuple[float, float]]
        Dictionary mapping feature names to (min, max) coefficient range.
    """
    x_values = problem['x']
    variables = [f"x{index}" for index in range(x_values.shape[0])]

    scaled_coefficients = coefficient_scaler(problem)

    coefficient_ranges = [(-sc * 10, sc * 10) for sc in scaled_coefficients]

    return dict(zip(variables, coefficient_ranges))


