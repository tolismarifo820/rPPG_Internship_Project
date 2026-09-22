import numpy as np
from scipy.signal import welch
import pywt

def calculate_snr(signal, fps, hr_low=0.7, hr_high=2.5):
    if len(signal) == 0: return 0.0
    freqs, psd = welch(signal, fs=fps, nperseg=len(signal))
    
    cardiac_mask = (freqs >= hr_low) & (freqs <= hr_high)
    if not np.any(cardiac_mask): return 0.0
    
    peak_idx = np.argmax(psd[cardiac_mask])
    peak_freq = freqs[cardiac_mask][peak_idx]
    
    signal_mask = (freqs >= peak_freq - 0.1) & (freqs <= peak_freq + 0.1)
    noise_mask = ~signal_mask
    
    # CHANGE HERE: Use np.trapezoid instead of np.trapz
    signal_power = np.trapezoid(psd[signal_mask], freqs[signal_mask])
    noise_power = np.trapezoid(psd[noise_mask], freqs[noise_mask])
    
    return float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else 0.0

def calculate_temporal_variance(bpm_array):
    valid_bpms = [b for b in bpm_array if b > 0 and not np.isnan(b)]
    if len(valid_bpms) < 2: return 0.0
    return float(np.std(valid_bpms))

def track_cwt_ridge(signal, fps):
    if len(signal) == 0: return 0.0
    scales = np.arange(1, 31)
    # Using Complex Morlet (cmor) for clear phase/amplitude separation
    coefficients, _ = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1.0/fps)
    power_spectrum = np.abs(coefficients)**2
    dominant_scales = np.argmax(power_spectrum, axis=0)
    return float(np.var(dominant_scales))

def cross_method_clustering(bpm_dict):
    valid_bpms = [bpm for bpm in bpm_dict.values() if bpm > 0 and not np.isnan(bpm)]
    if len(valid_bpms) < 2: return 0.0
    return float(np.std(valid_bpms))