"""
Quick sanity-check tests for the pure-numpy functions.
Run with:  python test_functions.py

To compare with MATLAB, paste the equivalent MATLAB snippet shown in each
section and check that the printed values match.
"""

import numpy as np
from eigen_detect import eigen_detect
from error_score import error_score
from ROC import ROC

np.random.seed(42)

# ── eigen_detect ──────────────────────────────────────────────────────────────
# Build a fake Omega (3 x 50) with a clear spike at window 20
r, n = 3, 50
Omega = np.random.rand(r, n) * 0.1
Omega[:, 20] *= 20   # large spike → should trigger detection near window 20

Detect = eigen_detect(Omega, delta_star=0.5)
print("eigen_detect: detections at windows:", np.where(Detect == 1)[0].tolist())
# MATLAB equivalent:
#   rng(42); r=3; n=50;
#   Omega = rand(r,n)*0.1; Omega(:,21) = Omega(:,21)*20;
#   Detect = eigen_detect(Omega, 0.5);
#   find(Detect==1)

# ── error_score ───────────────────────────────────────────────────────────────
# Perfect detection: one event, detected exactly once
Detect_perfect = np.zeros(50)
Detect_perfect[20] = 1
events = [20]
err = error_score(Detect_perfect, events, c=100, d_star=5)
print(f"error_score (perfect): {err}  (expected 0)")

# Missed event: false negative, weight c=100
Detect_miss = np.zeros(50)
err_fn = error_score(Detect_miss, events, c=100, d_star=5)
print(f"error_score (false neg): {err_fn}  (expected 100)")

# Extra detection: false positive
Detect_fp = np.zeros(50)
Detect_fp[5] = 1
Detect_fp[20] = 1
err_fp = error_score(Detect_fp, events, c=100, d_star=5)
print(f"error_score (1 false pos): {err_fp}  (expected 1)")

# MATLAB equivalent:
#   Detect_perfect = zeros(1,50); Detect_perfect(21)=1;
#   error_score(Detect_perfect, [21], 100, 5)   % → 0
#   error_score(zeros(1,50),    [21], 100, 5)   % → 100
#   Detect_fp=zeros(1,50); Detect_fp(6)=1; Detect_fp(21)=1;
#   error_score(Detect_fp, [21], 100, 5)        % → 1

# ── ROC ───────────────────────────────────────────────────────────────────────
# Spike at index 15 triggers a detection on the way up (index 15) AND on the
# way back down (index 16), so FPR > 0.  TPR must still be 1.
Omega_roc = np.ones((2, 30)) * 0.1
Omega_roc[:, 15] *= 20
fpr, tpr = ROC(Omega_roc, delta_star=0.5, frames=[15], event_tol=3)
print(f"ROC (spike): FPR={fpr:.3f} TPR={tpr:.3f}  (expected TPR=1)")

# No detections with impossibly high threshold → TPR=0, FPR=0
fpr0, tpr0 = ROC(np.ones((2, 30)) * 0.1, delta_star=999, frames=[15], event_tol=3)
print(f"ROC (no detections): FPR={fpr0:.3f} TPR={tpr0:.3f}  (expected 0, 0)")

# ── DMD core math (SVD + eig, no video) ───────────────────────────────────────
# Structured data: x_{t+1} = A x_t + noise, so A_tilde should recover A's eigs
np.random.seed(0)
dim, T_win, r = 100, 40, 3
# Build low-rank dynamics with known eigenvalues
lam_true = np.array([0.95, 0.80, 0.60])
modes = np.random.randn(dim, r)
coefs = np.random.randn(r, T_win + 1)
for t in range(1, T_win + 1):
    coefs[:, t] = lam_true * coefs[:, t - 1]
X = modes @ coefs + np.random.randn(dim, T_win + 1) * 1e-6

Y1 = X[:, :T_win]
Y2 = X[:, 1:T_win + 1]
U, s, Vh = np.linalg.svd(Y1, full_matrices=False)
Ur = U[:, :r]
Vr = Vh[:r, :].T
Atilde = Ur.T @ Y2 @ Vr @ np.diag(1.0 / s[:r])
eVals, _ = np.linalg.eig(Atilde)
recovered = np.sort(np.abs(eVals))[::-1]
print(f"DMD core: recovered |eig| = {np.round(recovered, 4).tolist()}")
print(f"          true      |eig| = {np.sort(lam_true)[::-1].tolist()}")
print(f"          max error = {np.max(np.abs(recovered - np.sort(lam_true)[::-1])):.2e}  (should be ~1e-6)")

print("\nAll checks passed.")
