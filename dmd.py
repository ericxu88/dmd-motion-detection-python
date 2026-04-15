import numpy as np
import cv2


def dmd(T, r, video, C):
    """
    Performs sliding window DMD on a given video incorporating the rank
    reduction procedure and compressed DMD.

    Parameters
    ----------
    T : int
        Prescribed window width.
    r : int
        Target rank of the rank-reduction step.
    video : str
        Path to the video file.
    C : ndarray, shape (p, height*width)
        Measurement matrix for compressed DMD.

    Returns
    -------
    Omega : ndarray, shape (r, num_windows)
        Moduli of the eigenvalues for each window (one column per window).

    Author: Marco Mignacca
    """
    cap = cv2.VideoCapture(video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_mat = np.zeros((height * width, num_frames))

    for j in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        video_mat[:, j] = gray.flatten()

    cap.release()

    # Apply Sliding Window DMD
    num_windows = num_frames - T
    Video_comp = C @ video_mat  # Compressed video matrix
    Omega = np.zeros((r, num_windows))

    for k in range(num_windows):
        Y1 = Video_comp[:, k:k + T]
        Y2 = Video_comp[:, k + 1:k + T + 1]

        U, s, Vh = np.linalg.svd(Y1, full_matrices=False)
        Ur = U[:, :r]
        Sr = np.diag(s[:r])
        Vr = Vh[:r, :].T  # (T x r)

        Atilde = Ur.T @ Y2 @ Vr @ np.diag(1.0 / s[:r])
        eVals, _ = np.linalg.eig(Atilde)

        omega = np.log(eVals)
        Omega[:, k] = np.abs(omega)

    return Omega
