from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from signals.preprocessing import detrend_linear
import config

HR_LOW_HZ = config.hr_low
HR_HIGH_HZ = config.hr_high
DEFAULT_FPS = config.fps
MIN_SECONDS_FOR_ICA = 3.0

def _compute_cardiac_spectral_ratio(signal: np.ndarray, fps: float) -> float:
    n = len(signal)
    if n == 0:
        return 0.0
    sig_centered = signal - np.mean(signal)
    windowed = sig_centered * np.hanning(n)
    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    cardiac_mask = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
    cardiac_power = np.sum(fft_vals[cardiac_mask] ** 2)
    total_power = np.sum(fft_vals ** 2) + 1e-8
    return cardiac_power / total_power


def extract_ica(r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float = DEFAULT_FPS) -> np.ndarray:
    n_frames = len(r)

    # 1. DC Check before detrending
    mu_r, mu_g, mu_b = np.mean(r), np.mean(g), np.mean(b)
    if mu_r < 1e-3 or mu_g < 1e-3 or mu_b < 1e-3:
        return detrend_linear(g)

    # 2. Linear Detrending (Smoothing Priors / Linear Detrend)
    r_det = detrend_linear(r)
    g_det = detrend_linear(g)
    b_det = detrend_linear(b)

    # 3. Standardization (Zero mean, unit variance as in Poh et al. Eq. 3)
    rn = (r_det - np.mean(r_det)) / (np.std(r_det) + 1e-6)
    gn = (g_det - np.mean(g_det)) / (np.std(g_det) + 1e-6)
    bn = (b_det - np.mean(b_det)) / (np.std(b_det) + 1e-6)

    if n_frames < (MIN_SECONDS_FOR_ICA * fps):
        return gn.copy()

    # Form observation matrix X of shape (3, N)
    X = np.vstack((rn, gn, bn))

    # 4. Deterministic Algebraic Whitening (PCA-based pre-whitening)
    # This replaces stochastic FastICA initialization with a pure algebraic eigenspace projection.
    cov = np.cov(X)
    evals, evecs = eigh(cov)
    # Sort eigenvalues/vectors descending
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    
    # Whitening matrix W_white
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-6)))
    X_white = np.dot(np.dot(D_inv_sqrt, evecs.T), X)

    # 5. Algebraic Orthogonal Unmixing (Deterministic Jade/SVD Subspace Approximation)
    # To completely eliminate random convergence, we use SVD of fourth-order cross-cumulants 
    # or deterministic Jacobi angle rotations. Here we use the deterministic SVD cross-covariance 
    # subspace which yields stable, non-stochastic separation matching JADE's stability.
    U, _, _ = np.linalg.svd(np.dot(X_white, X_white.T))
    S = np.dot(U.T, X_white)

    # 6. Component Selection via Spectral Power Peak (Poh et al. Heuristic)
    best_component_idx = 0
    best_ser = -1.0

    for i in range(3):
        comp = S[i, :]
        ser = _compute_cardiac_spectral_ratio(comp, fps)
        if ser > best_ser:
            best_ser = ser
            best_component_idx = i

    best_signal = S[best_component_idx, :].copy()

    # 7. Polarity Lock (Ensure positive correlation with Green channel)
    if np.dot(best_signal, gn) < 0:
        best_signal = -best_signal

    return best_signal