import numpy as np


def ROC(Omega, delta_star, frames, event_tol):
    """
    Computes the False Positive Rate (FPR) and True Positive Rate (TPR)
    for a given threshold on a single video.

    Parameters
    ----------
    Omega : ndarray, shape (r, num_windows)
        Moduli of the eigenvalues in each window.
    delta_star : float
        Detection threshold for the relative change in average eigenvalue.
    frames : array-like
        0-based window indices of true events.
    event_tol : int
        Tolerance in each direction for event detection (± event_tol windows).

    Returns
    -------
    FPR : float
        False positive rate.
    TPR : float
        True positive rate.

    Author: Marco Mignacca
    """
    num_windows = Omega.shape[1]
    Detect = np.zeros(num_windows)

    Average = np.mean(Omega, axis=0)
    for k in range(1, num_windows):
        delta = abs((Average[k] - Average[k - 1]) / Average[k - 1])
        if delta > delta_star:
            Detect[k] = 1

    fn = 0
    tp = 0
    for i in frames:
        lo = max(0, i - event_tol)
        hi = min(num_windows, i + event_tol + 1)
        slice_ = Detect[lo:hi]
        if np.sum(slice_ == 1) == 0:
            fn += 1
        else:
            tp += 1

    fp = int(np.sum(Detect)) - tp
    tn = (num_windows - 1) - fp - fn - tp

    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return FPR, TPR
