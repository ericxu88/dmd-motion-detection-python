# -------------------------------------------------------------------------
# Generates the eigenvalue modulus matrices 'Omega<i>.txt' from videos
# labelled 'vid_<i>.mp4'. Modify num_vids and the video naming convention
# to suit your own database.
#
# Author: Marco Mignacca
# -------------------------------------------------------------------------

import numpy as np
from dmd import dmd

num_vids = 20
T = 80
r = 5
p = 20

for i in range(1, num_vids + 1):
    video = f'vid_{i}.mp4'

    # Read the first frame to determine dimensions for the measurement matrix
    import cv2
    cap = cv2.VideoCapture(video)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    dim = height * width
    C = np.random.randn(p, dim)

    Omega = dmd(T, r, video, C)
    np.savetxt(f'Omega{i}.txt', Omega)
