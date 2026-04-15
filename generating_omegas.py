# -------------------------------------------------------------------------
# Generates the eigenvalue modulus matrices 'Omega<i>.txt' from videos
# labelled 'vid_<i>.mp4'. Modify num_vids and the video naming convention
# to suit your own database.
#
# Author: Marco Mignacca
# -------------------------------------------------------------------------

import cv2
import numpy as np
from dmd import dmd

num_vids = 20
T = 80
r = 5
p = 20

# Read dimensions from the first video and generate C once so the same
# measurement matrix is used across all videos (required for comparability).
cap = cv2.VideoCapture('vid_1.mp4')
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.release()

C = np.random.randn(p, height * width)

for i in range(1, num_vids + 1):
    video = f'vid_{i}.mp4'
    Omega = dmd(T, r, video, C)
    np.savetxt(f'Omega{i}.txt', Omega)
