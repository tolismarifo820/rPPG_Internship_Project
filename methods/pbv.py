import numpy as np
import config

def extract_pbv(r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: int = 15) -> np.ndarray:
    """
    Vectorized PBV (Pulse Blood Volume) rPPG algorithm with sliding window overlap-add.
    Implements the Signature-based method from de Haan 2014 using the vector from config.
    """
    N = len(r)
    l = int(2 * fps)  
    
    if N < l:
        return np.zeros(N)

    # Retrieve and normalize the Blood Volume Pulse signature vector directly from config
    pbv_vector = np.array(config.PBV_VECTOR, dtype=np.float32)
    eps = 1e-6
    pbv_vector = pbv_vector / (np.linalg.norm(pbv_vector) + eps)

    # 1. Create sliding window views: shape (N - l + 1, l)
    r_win = np.lib.stride_tricks.sliding_window_view(r, l)
    g_win = np.lib.stride_tricks.sliding_window_view(g, l)
    b_win = np.lib.stride_tricks.sliding_window_view(b, l)

    # 2. Compute mean-centered normalized color signals
    rn = r_win / (np.mean(r_win, axis=1, keepdims=True) + eps) - 1.0
    gn = g_win / (np.mean(g_win, axis=1, keepdims=True) + eps) - 1.0
    bn = b_win / (np.mean(b_win, axis=1, keepdims=True) + eps) - 1.0

    num_windows = rn.shape[0]

    # 3. Stack into the C_n matrix: shape (num_windows, 3, l)
    Cn = np.stack((rn, gn, bn), axis=1)

    # 4. Compute the covariance matrix Q = C_n * (C_n)^T
    Q = np.matmul(Cn, np.transpose(Cn, (0, 2, 1)))
    Q = Q + np.eye(3) * eps

    # 5. Compute weights W_PBV = k * P_bv * Q^-1
    # Pass pbv_vector directly! NumPy will auto-broadcast it across all windows.
    W_raw = np.linalg.solve(Q, pbv_vector)
    W_norm = W_raw / (np.linalg.norm(W_raw, axis=1, keepdims=True) + eps)

    # 6. Extract the raw pulse per window: S = W_PBV * C_n
    h = np.einsum('ni, nil -> nl', W_norm, Cn)
    h_zero_mean = h - np.mean(h, axis=1, keepdims=True)

    # 7. Fast Overlap-Add assembly
    H = np.zeros(N)
    for i in range(l):
        H[i : i + num_windows] += h_zero_mean[:, i]

    return H