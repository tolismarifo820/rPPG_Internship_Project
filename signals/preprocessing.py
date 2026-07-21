# To call in main:
# from signals.preprocessing import *

import cv2
import numpy as np
from config import *

def detrend_linear(signal: np.ndarray) -> np.ndarray:
    """Removes linear drift (like camera auto-exposure adjustments) from a signal."""
    if len(signal) < 2: return signal
    x = np.arange(len(signal))
    slope, intercept = np.polyfit(x, signal, 1)
    return signal - (slope * x + intercept)