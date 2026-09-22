# To call in main:
# from signals.preprocessing import *

import cv2
import numpy as np
from config import *

def fuse_rppg_signals(signals_list):
    """Z-score normalizes and averages multiple rPPG waveforms."""
    if not signals_list:
        return np.array([])
    if len(signals_list) == 1:
        return signals_list[0]
        
    normalized_signals = []
    for sig in signals_list:
        std_val = np.std(sig)
        if std_val > 1e-6:
            normalized_signals.append((sig - np.mean(sig)) / std_val)
        else:
            normalized_signals.append(sig - np.mean(sig))
            
    return np.mean(normalized_signals, axis=0)

def detrend_linear(signal: np.ndarray) -> np.ndarray:
    """Removes linear drift (like camera auto-exposure adjustments) from a signal."""
    if len(signal) < 2: return signal
    x = np.arange(len(signal))
    slope, intercept = np.polyfit(x, signal, 1)
    return signal - (slope * x + intercept)

def compute_weighted_mean(roi_bgr, mask_float):
    """Calculates spatial RGB means using fractional pixel weights."""
    mask_3d = np.expand_dims(mask_float, axis=-1)
    weighted_sum = np.sum(roi_bgr.astype(np.float32) * mask_3d, axis=(0, 1))
    weight_total = np.sum(mask_float)
    
    if weight_total <= 1e-6:
        return 0.0, 0.0, 0.0
        
    return float(weighted_sum[0]), float(weighted_sum[1]), float(weighted_sum[2])