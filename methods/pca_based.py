import numpy as np
from sklearn.decomposition import PCA
from signals.preprocessing import detrend_linear

def extract_pca(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    r, g, b = detrend_linear(r), detrend_linear(g), detrend_linear(b)
    
    mu_r, mu_g, mu_b = np.mean(r), np.mean(g), np.mean(b)
    if mu_r < 1e-3 or mu_g < 1e-3 or mu_b < 1e-3: return g.copy()
    
    # Normalize channels to give PCA equal variance scaling
    rn = (r - mu_r) / (np.std(r) + 1e-6)
    gn = (g - mu_g) / (np.std(g) + 1e-6)
    bn = (b - mu_b) / (np.std(b) + 1e-6)
    
    X = np.vstack((rn, gn, bn)).T
    
    pca = PCA(n_components=3)
    try:
        S = pca.fit_transform(X) 
    except Exception:
        return gn.copy()
        
    # Heuristic: Select the component with the highest absolute correlation to the Green channel
    correlations = [np.abs(np.corrcoef(S[:, i], gn)[0, 1]) for i in range(3)]
    best_component_idx = np.argmax(correlations)
    
    best_signal = S[:, best_component_idx]
    if np.corrcoef(best_signal, gn)[0, 1] < 0:
        best_signal = -best_signal
        
    return best_signal