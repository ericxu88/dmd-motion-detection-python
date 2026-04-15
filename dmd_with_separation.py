import numpy as np
import cv2


def dmd_from_frames(frames_array, T, r, p):
    """
    Performs background/foreground separation on a pre-loaded array of frames,
    combining compressed and sliding window DMD techniques.

    Parameters
    ----------
    frames_array : ndarray, shape (num_frames, height, width)
        Greyscale frames as float values.
    T : int
        Prescribed window width for sliding window DMD.
    r : int
        Target rank for the rank-reduction procedure.
    p : int
        Target dimensionality reduction for compressed DMD.

    Returns
    -------
    X_background : ndarray, shape (height, width, num_frames)
        Isolated background frames.
    X_foreground : ndarray, shape (height, width, num_frames)
        Isolated foreground frames.
    video_full : ndarray, shape (height, width, num_frames)
        Original greyscale video tensor.
    Omega_continuous : ndarray, shape (r, num_windows)
        Matrix of continuous (complex) eigenvalues per window.

    Author: Marco Mignacca
    """
    num_frames, height, width = frames_array.shape
    dim = height * width
    video_mat = frames_array.reshape(num_frames, dim).T  # (dim, num_frames)
    return _dmd_separation_core(video_mat, height, width, num_frames, T, r, p)


def dmd_with_separation(video, T, r, p):
    """
    Performs background/foreground separation on a video file, combining
    compressed and sliding window DMD techniques.

    Parameters
    ----------
    video : str
        Path to the video file.
    T : int
        Prescribed window width for sliding window DMD.
    r : int
        Target rank for the rank-reduction procedure.
    p : int
        Target dimensionality reduction for compressed DMD.

    Returns
    -------
    X_background : ndarray, shape (height, width, num_frames)
        Isolated background frames.
    X_foreground : ndarray, shape (height, width, num_frames)
        Isolated foreground frames.
    video_full : ndarray, shape (height, width, num_frames)
        Original greyscale video tensor.
    Omega_continuous : ndarray, shape (r, num_windows)
        Matrix of continuous (complex) eigenvalues per window.

    Author: Marco Mignacca
    """
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dim = height * width

    video_mat = np.zeros((dim, num_frames))
    for j in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        video_mat[:, j] = gray.flatten()
    cap.release()

    return _dmd_separation_core(video_mat, height, width, num_frames, T, r, p)


def _dmd_separation_core(video_mat, height, width, num_frames, T, r, p):
    """Shared DMD separation computation used by both public entry points."""
    dim = height * width
    num_windows = num_frames - T
    x_bg = np.zeros((dim, num_frames), dtype=complex)
    GaussSum = np.zeros(num_frames)
    tgrid = np.arange(num_frames, dtype=float)
    sig = T / 8.0

    C = np.random.randn(p, dim)
    video_comp = C @ video_mat

    Omega = np.zeros((r, num_windows))
    Omega_continuous = np.zeros((r, num_windows), dtype=complex)

    for k in range(num_windows):
        X1 = video_mat[:, k:k + T]
        X2 = video_mat[:, k + 1:k + T + 1]
        Y1 = video_comp[:, k:k + T]
        Y2 = video_comp[:, k + 1:k + T + 1]

        U, s, Vh = np.linalg.svd(Y1, full_matrices=False)
        Ur = U[:, :r]
        Sr_diag = s[:r]
        Vr = Vh[:r, :].T  # (T x r)

        Sinv = np.diag(1.0 / Sr_diag)
        Atilde = Ur.T @ Y2 @ Vr @ Sinv
        eVals, eVecs = np.linalg.eig(Atilde)
        Phi = X2 @ Vr @ Sinv @ eVecs  # DMD modes

        omega = np.log(eVals)
        Omega_continuous[:, k] = omega
        Omega[:, k] = np.abs(omega)

        # Foreground/background separation via frequency thresholding
        cutoff = 0.01
        bg = np.where(np.abs(omega) < cutoff)[0]

        bg_ev = 1j * np.imag(np.log(eVals[bg]))
        Phi_bg = Phi[:, bg]

        win_mid = (tgrid[k] + tgrid[k + T - 1]) / 2.0

        b_bg, _, _, _ = np.linalg.lstsq(Phi_bg, X1[:, 0], rcond=None)

        # Weighted reconstruction with Gaussian window
        for t in range(len(b_bg)):
            exp_term = np.exp(
                -(tgrid - win_mid) ** 2 / sig ** 2 + bg_ev[t] * tgrid
            )
            x_bg += b_bg[t] * np.outer(Phi_bg[:, t], exp_term)

        GaussSum += np.exp(-(tgrid - win_mid) ** 2 / sig ** 2)

    x_bg = x_bg / (GaussSum[np.newaxis, :] + 1e-10)

    # Make result real and non-negative
    x_fg = video_mat - np.abs(x_bg)
    neg_mask = x_fg < 0
    R = x_fg * neg_mask
    X_background = np.abs(x_bg)
    X_foreground = x_fg - R

    X_background = X_background.reshape(height, width, num_frames)
    X_foreground = X_foreground.reshape(height, width, num_frames)
    video_full = video_mat.reshape(height, width, num_frames)

    return X_background, X_foreground, video_full, Omega_continuous
