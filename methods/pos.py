import numpy as np
from signals.preprocessing import detrend_linear

def extract_pos(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    r, g, b = detrend_linear(r), detrend_linear(g), detrend_linear(b)
    
    mu_r, mu_g, mu_b = np.mean(r), np.mean(g), np.mean(b)
    if mu_r < 1e-3 or mu_g < 1e-3 or mu_b < 1e-3: return g.copy()
        
    rn, gn, bn = r / mu_r, g / mu_g, b / mu_b
    
    x = rn - gn
    y = 0.5 * rn + 0.5 * gn - bn
    std_x, std_y = np.std(x), np.std(y)
    alpha_val = std_x / (std_y + 1e-6)
    
    return x + alpha_val * y