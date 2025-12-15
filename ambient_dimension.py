import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import pandas as pd
def ambient_dimension_elbow(
    X,
    title="X",
    center=True,
    normalize_singular_values=False,
    curve="convex",
    direction="decreasing",
    show=True,
):
    """
    Estimate ambient/intrinsic dimension from the elbow of the singular values curve.

    Returns:
        elbow_dim (int): suggested ambient dimension (1-based index)
        S (np.ndarray): singular values
    """
    X = np.asarray(X)

    # Flatten samples if X is image/tensor-like: (n, h, w, ...) -> (n, d)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if center:
        X = X - X.mean(axis=0, keepdims=True)

    # SVD singular values
    S = np.linalg.svd(X, full_matrices=False, compute_uv=False)

    # Optionally normalize so different datasets are comparable in scale
    S_plot = (S / (S[0] + 1e-12)) if normalize_singular_values else S

    x = np.arange(1, len(S_plot) + 1)  # 1-based index for dimensions

    knee = KneeLocator(x, S_plot, curve=curve, direction=direction)
    elbow_dim = int(knee.knee) if knee.knee is not None else -1

    print(f"[{title}] Elbow (ambient dimension) = {elbow_dim}")

    if show:
        plt.figure(figsize=(8, 5))
        plt.plot(x, S_plot, marker='o', markersize=3, linewidth=1, label="Singular values")
        if elbow_dim != -1:
            plt.axvline(elbow_dim, linestyle="--", label=f"Elbow @ {elbow_dim}")
        plt.title(f"{title} - Singular Values Elbow")
        plt.xlabel("Component (dimension index)")
        plt.ylabel("Normalized singular value" if normalize_singular_values else "Singular value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return elbow_dim, S

# Modify below code to import your dataset and find approximate rank
df_iris = pd.read_csv(r"iris.csv")
data_iris = df_iris.iloc[:, :-1].values
X = data_iris.T  # shape: (features, samples)
ambient_dimension_elbow(
        X,
        title="iris",
        center=True,
        normalize_singular_values=True,  # helps plotting consistency
        show=True
    )
