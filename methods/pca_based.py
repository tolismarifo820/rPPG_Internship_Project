from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from signals.preprocessing import detrend_linear
import config

HR_LOW_HZ = config.hr_low
HR_HIGH_HZ = config.hr_high
DEFAULT_FPS = config.fps


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


def extract_pca(r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: float = DEFAULT_FPS) -> np.ndarray:
    # 1. DC Check before detrending
    mu_r, mu_g, mu_b = np.mean(r), np.mean(g), np.mean(b)
    if mu_r < 1e-3 or mu_g < 1e-3 or mu_b < 1e-3:
        return detrend_linear(g)

    # 2. Linear Detrending
    r_det = detrend_linear(r)
    g_det = detrend_linear(g)
    b_det = detrend_linear(b)

    # 3. Standardization
    std_r = np.std(r_det) + 1e-6
    std_g = np.std(g_det) + 1e-6
    std_b = np.std(b_det) + 1e-6

    rn = r_det / std_r
    gn = g_det / std_g
    bn = b_det / std_b

    X = np.vstack((rn, gn, bn)).T

    # 4. PCA Decomposition
    pca = PCA(n_components=3)
    try:
        S = pca.fit_transform(X) 
    except Exception:
        return gn.copy()

    # 5. Component Selection via SER
    best_component_idx = 0
    best_ser = -1.0

    for i in range(3):
        comp = S[:, i]
        ser = _compute_cardiac_spectral_ratio(comp, fps)
        if ser > best_ser:
            best_ser = ser
            best_component_idx = i

    best_signal = S[:, best_component_idx].copy()

    # 6. Polarity Lock
    if np.dot(best_signal, gn) < 0:
        best_signal = -best_signal

    return best_signal