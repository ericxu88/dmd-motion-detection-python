import numpy as np


def error_score(Detect, events, c, d_star):
    """
    Returns the error score for a particular threshold applied to a video.

    Parameters
    ----------
    Detect : ndarray
        Binary detection vector output from eigen_detect (0-based indices).
    events : array-like
        0-based indices of windows in which true events occur.
    c : float
        Weight for false negatives in the error score.
    d_star : int
        Tolerance in window index for detecting events (± d_star windows).

    Returns
    -------
    error : float
        Weighted error: false_pos + c * false_neg.

    Author: Marco Mignacca
    """
    Detect = Detect.copy().astype(float)
    false_neg = 0
    false_pos = 0

    for win in events:
        # Clamp to valid index range
        idx_lo = max(0, win - d_star)
        idx_hi = min(len(Detect), win + d_star + 1)
        indices = np.arange(idx_lo, idx_hi)
        instances = indices[Detect[indices] == 1]

        if len(instances) == 0:          # No detection within tolerance: false negative
            false_neg += 1
        elif len(instances) >= 2:        # Multiple detections: count extras as false positives
            false_pos += len(instances) - 1
            Detect[instances] = 0
        else:                            # Exactly one detection: true positive
            Detect[instances] = 0

    # Any remaining detections not matched to events are false positives
    false_pos += int(np.sum(Detect == 1))

    error = false_pos + c * false_neg
    return error
