import numpy as np


def eigen_detect(Omega, delta_star):
    """
    Performs the eigenvalue detection scheme by checking if the relative
    change in eigenvalue average between consecutive windows exceeds a
    prescribed threshold `delta_star`.

    Parameters
    ----------
    Omega : ndarray, shape (r, num_windows)
        Moduli of the eigenvalues in each window.
    delta_star : float
        Detection threshold for the relative change in average eigenvalue.

    Returns
    -------
    Detect : ndarray, shape (num_windows,)
        Array with 1 at indices where an event is detected, 0 elsewhere.

    Author: Marco Mignacca
    """
    num_windows = Omega.shape[1]
    Detect = np.zeros(num_windows)

    Average = np.mean(Omega, axis=0)

    # Start from index 5 to have a 5-frame lookback window (mirrors MATLAB k=6, 1-based)
    for k in range(5, num_windows):
        delta = abs((Average[k] - Average[k - 1]) / Average[k - 1])
        if delta > delta_star and not np.any(Detect[k - 5:k]):
            Detect[k] = 1

    return Detect
