# To call in main:
# from estimation.sqi import *

import cv2
import numpy as np
from collections import deque
from config import *

def robust_mean(values: deque[float]) -> float | None:
    if len(values) < 3: return None
    arr = np.asarray(values, dtype=np.float32)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) + 1e-6
    keep = np.abs(arr - med) < 2.5 * mad
    if not np.any(keep): return None
    return float(np.mean(arr[keep]))

def compute_motion_score(prev_roi_gray, current_roi_gray):
    if prev_roi_gray is None or current_roi_gray is None: return 0.0
    try:
        if prev_roi_gray.shape != current_roi_gray.shape: current_roi_gray = cv2.resize(current_roi_gray, (prev_roi_gray.shape[1], prev_roi_gray.shape[0]))
        diff = cv2.absdiff(prev_roi_gray, current_roi_gray)
        return float(np.clip(100.0 - float(np.mean(diff)) * 6.0, 0.0, 100.0))
    except Exception: return 0.0

def compute_brightness_score(brightness_values):
    if brightness_values is None or len(brightness_values) < 10: return 0.0
    arr = np.asarray(list(brightness_values), dtype=np.float32)
    std_val, mean_val = float(np.std(arr)), float(np.mean(arr))
    penalty = (45 - mean_val) * 1.2 if mean_val < 45 else (mean_val - 220) * 1.2 if mean_val > 220 else 0.0
    return float(np.clip(100.0 - std_val * 4.0 - penalty, 0.0, 100.0))

def compute_periodicity_score(signal_values, fps_value):
    if signal_values is None or len(signal_values) < 45: return 0.0
    x = np.asarray(list(signal_values), dtype=np.float32)
    x = x - float(np.mean(x))
    if float(np.std(x)) < 1e-6: return 0.0
    freqs_local = np.fft.rfftfreq(len(x), d=1.0 / max(float(fps_value), 1e-6))
    mag = np.abs(np.fft.rfft(x))
    band = (freqs_local >= 0.7) & (freqs_local <= 3.0)
    if not np.any(band): return 0.0
    dominance = float(np.max(mag[band])) / (float(np.mean(mag[band])) + 1e-6)
    return float(np.clip((dominance - 1.0) / 3.0 * 100.0, 0.0, 100.0))

def compute_signal_quality_index(motion_score, brightness_score, periodicity_score, face_detected_now, vitals_enabled):
    if not face_detected_now: return 0.0, "NO FACE"
    score = 0.30 * float(motion_score) + 0.25 * float(brightness_score) + 0.45 * float(periodicity_score)
    if not vitals_enabled: score = min(score, 60.0)
    score = float(np.clip(score, 0.0, 100.0))
    if score >= SQI_GOOD_THRESHOLD: return score, "GOOD SIGNAL"
    elif score >= SQI_FAIR_THRESHOLD: return score, "FAIR SIGNAL"
    return score, "POOR SIGNAL"