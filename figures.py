# -------------------------------------------------------------------------
# Generates various figures from the paper:
#
# Figures 1-3  →  Figure 1 in paper: background, foreground, and original
#                 frame 100 of the gate video.
# Figure 4     →  Figure 3 in paper: continuous eigenvalues in window 100.
# Figure 5     →  Part of Figure 2 in paper: eigenvalue moduli per window
#                 for video 3 in the database.
# Figure 6     →  Part of Figure 6 in paper: mean eigenvalue modulus per
#                 window for video 3.
#
# Author: Marco Mignacca
# -------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dmd_with_separation import dmd_with_separation

video = 'gate_low.mp4'
T = 100
r = 20   # For plotting purposes (use r=5 for separation only)
p = 20

X_background, X_foreground, video_full, Omega_continuous = \
    dmd_with_separation(video, T, r, p)

fig1, ax1 = plt.subplots()
ax1.imshow(X_foreground[:, :, 99].astype(np.uint8), cmap='gray')
ax1.set_title('Foreground – frame 100')
ax1.axis('off')

fig2, ax2 = plt.subplots()
ax2.imshow(X_background[:, :, 99].astype(np.uint8), cmap='gray')
ax2.set_title('Background – frame 100')
ax2.axis('off')

fig3, ax3 = plt.subplots()
ax3.imshow(video_full[:, :, 99].astype(np.uint8), cmap='gray')
ax3.set_title('Original – frame 100')
ax3.axis('off')

# Continuous eigenvalues in window 100 (plot real vs imaginary parts)
eigenvals = np.sort(Omega_continuous[:, 99])
fig4, ax4 = plt.subplots()
ax4.scatter(eigenvals.real, eigenvals.imag, color='r', marker='*', linewidths=4,
            label='Background')
ax4.scatter(eigenvals[1:].real, eigenvals[1:].imag, color='b', marker='*',
            linewidths=4, label='Foreground')
ax4.grid(True)
ax4.set_xlabel('Real Part')
ax4.set_ylabel('Imaginary Part')
ax4.tick_params(labelsize=16)

# Eigenvalue moduli per window (from pre-computed file)
Omega = np.loadtxt('Omega3.txt')

fig5, ax5 = plt.subplots()
for row in Omega:
    ax5.plot(np.arange(1, Omega.shape[1] + 1), row, '*')
ax5.set_xlim([0, Omega.shape[1]])
ax5.set_xlabel('Window Number')
ax5.set_ylabel('Eigenvalue Modulus')

fig6, ax6 = plt.subplots()
omega_mean = np.mean(Omega, axis=0)
ax6.plot(np.arange(1, Omega.shape[1] + 1), omega_mean, '-', linewidth=1.5)
ax6.set_xlim([0, Omega.shape[1]])
ax6.set_ylim([0, np.max(omega_mean)])
ax6.set_xlabel('Window Number')
ax6.set_ylabel('Eigenvalue Average')

plt.show()
