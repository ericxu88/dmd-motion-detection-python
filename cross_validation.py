import numpy as np
from eigen_detect import eigen_detect
from error_score import error_score


def cross_validation(windows, num_vids, delta_vals, k, c, dist_star):
    """
    Performs pseudo-k-fold cross-validation on a set of videos whose
    eigenvalue matrices were pre-computed by dmd() and saved as
    'Omega<i>.txt'.

    Parameters
    ----------
    windows : ndarray, shape (max_windows, num_vids)
        Matrix where entry [i, j] == 1 marks the earliest window in video j
        at which motion could be detected (0-based row indexing).
    num_vids : int
        Total number of videos in the dataset.
    delta_vals : array-like
        Threshold values to sweep over.
    k : int
        Number of folds.
    c : float
        Weight for false negatives in the error score.
    dist_star : int
        Leeway in window index for event detection.

    Returns
    -------
    optimal_deltas : ndarray, shape (k,)
        Optimal threshold chosen on the training set for each fold.
    avg_validation_error : ndarray, shape (k,)
        Average error on the validation set for each fold.
    avg_training_error : ndarray, shape (k, len(delta_vals))
        Average training error across all threshold values for each fold.

    Author: Marco Mignacca
    """
    delta_vals = np.asarray(delta_vals)
    num_per_fold = num_vids // k

    # Shuffled 1-based video indices (matching 'Omega<i>.txt' filenames)
    indices = np.random.permutation(num_vids) + 1

    optimal_deltas = np.zeros(k)
    avg_training_error = np.zeros((k, len(delta_vals)))
    avg_validation_error = np.zeros(k)

    for j in range(k):
        # Split into validation and training folds
        val_pos = np.arange(j * num_per_fold, (j + 1) * num_per_fold)
        val_ind = indices[val_pos]
        train_ind = np.delete(indices, val_pos)

        n_train = len(train_ind)
        training_error = np.zeros((n_train, len(delta_vals)))

        for m, index in enumerate(train_ind):
            Omega = np.loadtxt(f'Omega{index}.txt')

            key_frames = np.where(windows[:, index - 1] == 1)[0]
            error_vec = np.zeros(len(delta_vals))
            for iter_, threshold in enumerate(delta_vals):
                Detect = eigen_detect(Omega, threshold)
                error_vec[iter_] = error_score(Detect, key_frames, c, dist_star)

            training_error[m, :] = error_vec

        avg_training_error[j, :] = np.mean(training_error, axis=0)
        best_idx = np.argmin(avg_training_error[j, :])
        optimal_deltas[j] = delta_vals[best_idx]

        # Evaluate on the validation set
        val_err = np.zeros(num_per_fold)
        for m, index in enumerate(val_ind):
            Omega = np.loadtxt(f'Omega{index}.txt')
            key_frames = np.where(windows[:, index - 1] == 1)[0]
            Detect = eigen_detect(Omega, optimal_deltas[j])
            val_err[m] = error_score(Detect, key_frames, c, dist_star)

        avg_validation_error[j] = np.mean(val_err)

    return optimal_deltas, avg_validation_error, avg_training_error
