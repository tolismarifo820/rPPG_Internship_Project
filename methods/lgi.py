import numpy as np

def extract_lgi(r: np.ndarray, g: np.ndarray, b: np.ndarray, fps: int = 15) -> np.ndarray:
    """
    Vectorized Local Group Invariance (LGI) rPPG algorithm.
    Implementation based on Pilz et al. (CVPR 2018).
    """
    N = len(r)
    # Hyperparameter 1: Local Window Length (l)
    # The paper evaluates local transformations over a short temporal interval.
    # 1 to 1.5 seconds is optimal for local invariance without capturing global drift.
    l = int(fps * 1.5)  
    
    if N < l:
        return np.zeros(N)

    # Step 1: Create 2D sliding window views: shape (num_windows, l)
    r_win = np.lib.stride_tricks.sliding_window_view(r, l)
    g_win = np.lib.stride_tricks.sliding_window_view(g, l)
    b_win = np.lib.stride_tricks.sliding_window_view(b, l)
    num_windows = r_win.shape[0]

    # Stack raw observation vectors x(t): shape (num_windows, l, 3)
    X = np.stack((r_win, g_win, b_win), axis=-1)

    # Step 2: Approximate local group transformations (Eq. 4 & 5)
    # We compute the temporal derivatives (frame-to-frame differences) 
    # to capture the immediate action of nuisance factors (motion/illumination).
    dX = X[:, 1:, :] - X[:, :-1, :]  # shape: (num_windows, l-1, 3)

    # Step 3: Compute Covariance matrix C of the transformations (Eq. 7)
    # C = dX^T * dX. shape: (num_windows, 3, 3)
    C = np.matmul(np.transpose(dX, (0, 2, 1)), dX)

    # Step 4: Symmetric eigenvalue problem (Eq. 8)
    # np.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Extract eigenvector corresponding to the largest variance (the nuisance noise)
    # V shape: (num_windows, 3, 1)
    V_noise = eigenvectors[:, :, -1:] 

    # Step 5: Construct projection operator P with corank k=1 (Eq. 9)
    # P = I - V * V^T
    I = np.eye(3).reshape(1, 3, 3)
    P = I - np.matmul(V_noise, np.transpose(V_noise, (0, 2, 1)))

    # Step 6: Project raw features onto the complementary subspace (Eq. 10)
    # X_tilde = X * P. shape: (num_windows, l, 3)
    X_tilde = np.matmul(X, P)

    # Step 7: Isolate the 1D pulse from the remaining concentrated energy
    # We mean-center the projected features and extract the primary remaining component.
    X_tilde_mc = X_tilde - np.mean(X_tilde, axis=1, keepdims=True)
    C_tilde = np.matmul(np.transpose(X_tilde_mc, (0, 2, 1)), X_tilde_mc)
    _, evecs_tilde = np.linalg.eigh(C_tilde)
    
    # Extract the dominant remaining pulse axis
    U_pulse = evecs_tilde[:, :, -1:]  # (num_windows, 3, 1)
    
    # Project onto 1D signal: shape (num_windows, l)
    h = np.matmul(X_tilde_mc, U_pulse).reshape(num_windows, l)

    # Fast Overlap-Add assembly (similar to POS/CHROM)
    h_zero_mean = h - np.mean(h, axis=1, keepdims=True)
    
    # Phase alignment: PCA can arbitrarily flip the signal phase (+/-). 
    # We force the green channel (index 1) to always be the positive reference.
    g_alignment = np.sign(U_pulse[:, 1, 0])
    g_alignment[g_alignment == 0] = 1
    h_zero_mean = h_zero_mean * g_alignment[:, np.newaxis]

    H = np.zeros(N)
    for i in range(l):
        H[i : i + num_windows] += h_zero_mean[:, i]

    return H