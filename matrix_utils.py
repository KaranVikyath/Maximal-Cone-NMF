import numpy as np
import random
def generate_matrices(m, r, n, sigma=0.0, seed=None):
    """
    Generate a clean low‑rank matrix X = U V and a noisy version X_hat = X + N(0, sigma^2).
    
    Args:
      m (int): ambient dimension
      r (int): rank
      n (int): number of data points
      sigma (float): standard deviation of added Gaussian noise
    
    Returns:
      X (ndarray of shape (m, n)): clean data
      X_hat (ndarray of shape (m, n)): noisy data
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    W = np.abs(np.random.randn(m, r))
    H = np.abs(np.random.randn(r, n))
    X_clean = W @ H
    noise = sigma * np.abs(np.random.randn(m, n))
    
    return X_clean, X_clean + noise
