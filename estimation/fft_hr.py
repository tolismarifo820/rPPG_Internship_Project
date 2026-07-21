# To call in main:
# from estimation.fft_hr import *

import numpy as np
from config import *

def oscillate_in_range(lo: float, hi: float, period_s: float, t: float) -> float:
    mid, amp = (lo + hi) / 2.0, (hi - lo) / 2.0
    return float(mid + amp * np.sin((2.0 * np.pi * t) / max(period_s, 1e-6)))

def get_ac_dc(signal: np.ndarray, freq_mask: np.ndarray) -> tuple[float, float]:
    dc = float(np.mean(signal))
    centered = signal - dc
    mag = np.abs(np.fft.fft(centered))
    band_mag = mag * freq_mask
    ac = float(np.max(band_mag))
    return ac, dc

def estimate_peak_bpm(signal: np.ndarray, fps_val: float, min_hz: float, max_hz: float) -> float:
    """Uses FFT Zero-Padding to give high-resolution continuous BPM estimates."""
    centered = signal - float(np.mean(signal))
    n_pad = 1024  # Pad to 1024 points for high frequency resolution
    
    mag = np.abs(np.fft.fft(centered, n=n_pad))
    freqs = np.fft.fftfreq(n_pad, d=1.0 / max(float(fps_val), 1e-6))
    
    band_mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(band_mask): return 0.0
    
    idx = int(np.argmax(mag * band_mask))
    return float(freqs[idx] * 60.0)