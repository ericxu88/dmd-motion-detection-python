# -------------------------------------------------------------------------
# Optimizes the threshold parameter for videos in the Microsoft Wallflower
# database. Calculates the optimal threshold for each video and the
# corresponding error score.
#
# Author: Marco Mignacca
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from eigen_detect import eigen_detect
from error_score import error_score

# Parameters
num_vids = 6   # Bootstrap video excluded
T = 80
r = 5
p = 20
P = np.arange(0.001, 1.001, 0.001)
c = 100
d_star = 30   # Leeway in time (frames)

# Events matrix: row = window index (0-based), col = video index (0-based)
# A 1 marks a window in which an event occurs.
events_MW = np.zeros((5000, num_vids))

# Camouflage (353 frames)
events_MW[241 - T - 1, 0] = 1

# ForegroundAperture (2113 frames)
events_MW[505 - 1, 1] = 1
events_MW[919 - T - 1, 1] = 1
events_MW[1509 - 1, 1] = 1

# LightSwitch (2714 frames)
events_MW[796 - T - 1, 2] = 1
events_MW[829 - 1, 2] = 1
events_MW[1844 - T - 1, 2] = 1
events_MW[2202 - 1, 2] = 1

# MovedObject (1744 frames)
events_MW[637 - T - 1, 3] = 1
events_MW[891 - 1, 3] = 1
events_MW[1389 - T - 1, 3] = 1
events_MW[1502 - 1, 3] = 1

# TimeOfDay (5889 frames)
events_MW[1831 - T - 1, 4] = 1
events_MW[1918 - 1, 4] = 1
events_MW[3072 - T - 1, 4] = 1
events_MW[3244 - 1, 4] = 1
events_MW[4739 - T - 1, 4] = 1
events_MW[4933 - 1, 4] = 1

# WavingTrees (286 frames)
events_MW[242 - T - 1, 5] = 1

# Load pre-generated eigenvalue matrices
Omega_MW = []
for i in range(1, num_vids + 1):
    Omega_MW.append(np.loadtxt(f'Omega_MW{i}.txt'))

# Optimize threshold for each video
optimal_params = np.zeros((2, num_vids))
error = np.zeros((num_vids, len(P)))
detected_windows = []

for i in range(num_vids):
    Detects = []
    omega = Omega_MW[i]
    key_frames = np.where(events_MW[:, i] == 1)[0]

    for j, p_val in enumerate(P):
        Detect = eigen_detect(omega, p_val)
        error[i, j] = error_score(Detect, key_frames, c, d_star)
        Detects.append(Detect)

    err_min = np.min(error[i, :])
    idx = np.argmin(error[i, :])
    optimal_params[:, i] = [err_min, P[idx]]
    detected_windows.append(np.where(Detects[idx] == 1)[0])

# Display results (row 0: error, row 1: optimal threshold)
print(optimal_params)

# --- Plot error curve for video 4 (MovedObject, index 3) ---
fig1, ax1 = plt.subplots()
ax1.plot(P, error[3, :])
ax1.set_xlabel('Threshold value')
ax1.set_ylabel('Error score')

# --- Detection plot for video 4 (MovedObject, index 3) ---
actual_frames = np.where(events_MW[:, 3] == 1)[0]
temp = detected_windows[3]
omega = Omega_MW[3]
omega_mean = np.mean(omega, axis=0)

detections = -1 * np.ones(omega.shape[1])
detections[temp] = omega_mean[temp]

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(1, omega.shape[1] + 1), omega_mean, '-')
ax2.plot(np.arange(1, omega.shape[1] + 1), detections, '*', linewidth=2)
ax2.set_ylim([0, np.max(omega_mean) + 0.5])
ax2.set_xlabel('Window number')
ax2.set_ylabel('Eigenvalue modulus')

plt.show()
