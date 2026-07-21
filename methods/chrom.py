import numpy as np
from signals.preprocessing import detrend_linear

def extract_chrom(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    r, g, b = detrend_linear(r), detrend_linear(g), detrend_linear(b)
    
    mu_r, mu_g, mu_b = np.mean(r), np.mean(g), np.mean(b)
    if mu_r < 1e-3 or mu_g < 1e-3 or mu_b < 1e-3: return g.copy()
        
    rn, gn, bn = r / mu_r, g / mu_g, b / mu_b
    
    xs = 3 * rn - 2 * gn
    ys = 1.5 * rn + gn - 1.5 * bn
    std_xs, std_ys = np.std(xs), np.std(ys)
    alpha_val = std_xs / (std_ys + 1e-6)
    
    return xs - alpha_val * ys