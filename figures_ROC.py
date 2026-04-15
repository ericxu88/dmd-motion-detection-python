# -------------------------------------------------------------------------
# Plots the mean ROC curve (Figure 7 in the paper), assessing performance
# of classifying windows as foreground action vs. no action.
#
# Author: Marco Mignacca
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ROC import ROC

T = 80
event_tol = 15   # Tolerance in event detection (± 15 windows)
num_vids = 20

# Frame indices of true events per video (0-based)
frames = [None] * num_vids
frames[0]  = [756 - T - 1, 921 - 1]
frames[1]  = [366 - T - 1, 648 - 1]
frames[2]  = [333 - T - 1, 786 - 1, 1086 - T - 1, 1494 - 1]
frames[3]  = [660 - T - 1, 966 - 1, 1365 - T - 1, 1638 - 1]
frames[4]  = [393 - T - 1, 444 - 1]
frames[5]  = [351 - T - 1, 567 - 1, 987 - 1]
frames[6]  = [477 - T - 1, 594 - 1, 621 - 1]
frames[7]  = [426 - T - 1, 762 - 1]
frames[8]  = [543 - T - 1, 657 - 1]
frames[9]  = [354 - T - 1, 876 - 1, 1590 - 1, 1617 - T - 1, 1728 - 1, 2067 - 1]
frames[10] = [423 - T - 1, 669 - 1, 1017 - T - 1, 1290 - 1, 1545 - T - 1, 1620 - 1]
frames[11] = [759 - T - 1, 948 - T - 1, 1413 - T - 1, 1836 - T - 1, 1917 - 1, 2175 - 1, 2346 - 1, 2469 - 1]
frames[12] = [456 - T - 1, 720 - T - 1, 1269 - T - 1, 1335 - 1]
frames[13] = [327 - T - 1, 534 - 1, 987 - T - 1, 1131 - 1, 1479 - T - 1, 1572 - 1, 1617 - T - 1, 1707 - 1]
frames[14] = [1269 - 1, 1296 - 1, 1299 - 1, 1656 - T - 1, 1794 - 1, 2280 - T - 1, 2340 - 1]
frames[15] = [180 - T - 1, 984 - T - 1, 1059 - 1, 1089 - 1, 1572 - T - 1, 1722 - 1]
frames[16] = [312 - T - 1, 573 - T - 1, 888 - T - 1, 1332 - 1, 1671 - 1, 1728 - 1]
frames[17] = [132 - T - 1, 630 - T - 1, 687 - 1, 1107 - 1, 1668 - T - 1, 1725 - 1,
              2052 - T - 1, 2109 - 1, 2403 - T - 1, 2559 - 1, 3126 - T - 1]
frames[18] = [732 - T - 1, 1407 - T - 1, 1500 - 1, 1941 - 1]
frames[19] = [738 - T - 1, 822 - T - 1, 981 - 1, 1002 - 1, 1287 - T - 1, 1560 - 1]

# Load eigenvalue matrices
Omega = [np.loadtxt(f'Omega{i + 1}.txt') for i in range(num_vids)]

# Threshold sweep (logarithmic, matching MATLAB 10.^(-10:0.01:10))
vals = 10.0 ** np.arange(-10, 10.01, 0.01)

TPR = np.zeros((num_vids, len(vals)))
FPR = np.zeros((num_vids, len(vals)))

for i in range(num_vids):
    for iter_, threshold in enumerate(vals):
        FPR[i, iter_], TPR[i, iter_] = ROC(Omega[i], threshold, frames[i], event_tol)

AUC = abs(np.trapz(np.mean(TPR, axis=0), np.mean(FPR, axis=0)))
print(f'AUC:  {AUC}')

fig, ax = plt.subplots()
ax.plot(np.mean(FPR, axis=0), np.mean(TPR, axis=0), linewidth=2)
ax.set_xlim([0, 1])
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_aspect('equal')
ax.tick_params(labelsize=16)

plt.show()
