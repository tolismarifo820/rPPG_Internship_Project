import numpy as np
from signals.preprocessing import detrend_linear

def extract_green(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Flatten out ambient light/exposure shifts before processing
    g_detrend = detrend_linear(g)
    return g_detrend.copy()