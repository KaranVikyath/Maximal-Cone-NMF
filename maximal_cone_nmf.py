"""Directional method for finding facets in Non-negative Matrix Factorization.

This module implements an efficient directional Gauss-Jordan elimination approach
for discovering all valid facets of dimension m - r + 1 in NMF problems. The algorithm
explores the orthant structure by following directional vectors and identifying
facet boundaries.

Key components:
- Orthant crossing detection and facet exploration
- capital_delta matrix construction with orthonormalization
- Iterative facet dimension exploration using BFS
- Main entry point: directional_method()

"""
from typing import Optional, Tuple, Set, List
from collections import deque
import numpy as np
from scipy.linalg import qr

def find_orthant_crossing(
    w_1: np.ndarray,
    w_2: np.ndarray,
    omega: List[int]
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Move from vector w_1 along direction w_2 until hitting a facet boundary.
    
    Computes the first positive alpha that zeros out at least one coordinate
    when moving from w_1 in direction w_2, indicating a facet crossing.
    
    Args:
        w_1: Starting vector (current point in space).
        w_2: Direction vector to move along.
        omega: List of indices that are already zero and constrained.
    
    Returns:
        A tuple containing:
            - w_k: Updated vector after the step (None if no valid crossing found).
            - idx: Index of coordinate that became zero (None if no valid crossing).
    
    Notes:
        Returns (None, None) if no valid positive step exists or if the direction
        would violate non-negativity constraints.
    """
    # Find indices where w_2 is non-zero (potential crossing points)
    valid_indices = np.where(w_2 != 0)[0]
    
    # Calculate step sizes that would zero out each coordinate
    alphas = -w_1[valid_indices] / w_2[valid_indices]
    positive_alphas = alphas[alphas > 0]
    
    # Check if direction violates constraints at already-zero coordinates
    mask_zero = (w_1 == 0)
    if (positive_alphas.size == 0) or np.any(w_2[mask_zero] < 0):
        return None, None
    
    # Find minimum positive step size (first facet crossing)
    alpha_k = np.min(positive_alphas)
    idx = valid_indices[np.where(alphas == alpha_k)[0][0]]
    
    # Compute new position and enforce zero constraints
    w_k = w_1 + alpha_k * w_2
    w_k[omega] = 0
    w_k[idx] = 0
    
    return w_k, idx


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.
    
    Args:
        vec: Input vector to normalize.
    
    Returns:
        Normalized vector with L2 norm equal to 1.
    """
    return vec / np.linalg.norm(vec)


def orthonormal_basis(
    w_k: np.ndarray,
    r: int,
    W: np.ndarray
) -> np.ndarray:
    """Construct an orthonormal basis starting from w_k within the span of W.
    
    Args:
        w_k: Starting vector (will be normalized and used as first basis vector).
        r: Desired rank (number of basis vectors).
        W: Matrix whose column space contains the basis.
    
    Returns:
        Matrix with r orthonormal columns, first column is normalize(phi).
    """
    basis = [normalize(w_k)]
    
    for _ in range(r - 1):
        # Generate random vector in span of W
        random_vec = np.random.randn(r)
        random_vec = W @ random_vec
        
        # Gram-Schmidt orthogonalization
        for b in basis:
            random_vec -= np.dot(random_vec, b) * b
        
        basis.append(normalize(random_vec))
    
    return np.column_stack(basis)


def construct_phi_matrix(
    w_k: np.ndarray,
    r: int,
    W: np.ndarray,
    cached_qr: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct capital_delta matrix with w_k as first vector and modified remaining vectors.
    
    Creates an orthonormal basis with w_k as the first column, then modifies
    the remaining columns to zero out coordinates where w_k is zero.
    
    Args:
        w_k: Starting vector for first column.
        r: Target rank (number of columns in result).
        W: Matrix defining the column space.
        cached_qr: Pre-computed QR decomposition of W (optional, for efficiency).
    
    Returns:
        A tuple containing:
            - PHI: Constructed matrix of shape (m, r) with modified orthonormal columns.
            - Q_full: QR decomposition result (for caching in subsequent calls).
    """
    m = len(w_k)
    w_k_norm = normalize(w_k)
    zero_idx = np.where(w_k == 0)[0]
    
    # Use cached QR or compute new one
    if cached_qr is None:
        Q_full, _ = qr(W, mode='economic')
    else:
        Q_full = cached_qr
    
    # Build orthonormal basis using Gram-Schmidt
    Q = [w_k_norm]
    for vec in Q_full.T:
        # Project out components along existing basis vectors
        proj = sum(np.dot(vec, q) * q for q in Q)
        candidate = vec - proj
        
        Q.append(normalize(candidate))
        
        if len(Q) == r:
            break
    
    capital_delta = np.column_stack(Q)
    
    # Modify columns 2-r to zero out the same coordinates as w_k
    tail = capital_delta[zero_idx, 1:]
    identity_target = np.eye(len(zero_idx), r - 1)
    
    try:
        transform = np.linalg.lstsq(tail, identity_target, rcond=None)[0]
    except np.linalg.LinAlgError:
        transform = np.zeros((tail.shape[1], identity_target.shape[1]))
    
    capital_delta[:, 1:] = capital_delta[:, 1:] @ transform

    # Identity block for columns 1...r-1
    capital_delta[zero_idx, 1:] = np.eye(len(zero_idx), r - 1)

    # Column 0 must be exactly zero on zero_idx
    capital_delta[zero_idx, 0] = 0.0

    return capital_delta, Q_full


def find_direction(
    w_vectors: np.ndarray,
    omega: List[int]
) -> np.ndarray:
    """Compute directional vector preserving zero constraints in omega.
    
    Args:
        w_vectors: Matrix of direction vectors (columns are individual directions).
        omega: List of coordinate indices that must remain zero.
    
    Returns:
        Direction vector that preserves constraints in omega.
    """
    i = len(omega)
    
    if i == 0:
        return w_vectors[:, 0]
    
    omega_array = np.array(omega, dtype=int)
    gamma = -np.linalg.pinv(w_vectors[omega_array, :i]) @ w_vectors[omega_array, i]
    delta_k = w_vectors[:, :i] @ gamma + w_vectors[:, i]
    delta_k[omega_array] = 0
    
    return delta_k


def find_first_facet(
    w_k: np.ndarray,
    ix: int,
    w_vectors: np.ndarray,
    depth: int,
    current_facet: Tuple[int, ...],
    facets: Set[Tuple[int, ...]],
    omega: List[int]
) -> Tuple[Set[Tuple[int, ...]], np.ndarray, List[int]]:
    """Recursively explore facet tree to find first facet of target dimension.
    
    Explores the tree of facets by stepping in direction vectors and removing
    coordinates as they hit zero boundaries.
    
    Args:
        w_k: Current position vector.
        ix: Current index (unused, kept for compatibility).
        w_vectors: Matrix of direction vectors.
        depth: Number of steps to take (typically r - 1).
        current_facet: Current facet as tuple of active indices.
        facets: Set of discovered facets (modified in place).
        omega: List of zero coordinate indices (modified in place).
    
    Returns:
        A tuple containing:
            - facets: Updated set of discovered facets.
            - new_w_k: Final position vector after exploration.
            - omega: Updated list of zero constraints.
    """
    for ix in range(depth):
        delta_k = find_direction(w_vectors, omega)
        delta_k[omega] = 0
        new_w_k, neg_coord = find_orthant_crossing(w_k, delta_k, omega)
        
        if neg_coord is not None:
            new_facet = tuple(x for x in current_facet if x != neg_coord)
            omega.append(neg_coord)
            current_facet = new_facet
            w_k = new_w_k
    
    facets.add(current_facet)
    return facets, new_w_k, omega


def get_facet(m: int, omega: List[int]) -> Tuple[int, ...]:
    """Return a facet tuple by removing omega indices from [0, ..., m-1].
    
    Args:
        m: Total dimension size.
        omega: Indices to exclude from the facet.
    
    Returns:
        Sorted tuple of indices representing the facet.
    """
    return tuple(sorted(set(range(m)) - set(omega)))


def explore_facet_dimension(
    first_w: np.ndarray,
    principal_W: np.ndarray,
    facets: Set[Tuple[int, ...]],
    visited_facets: Set[Tuple[int, ...]],
    W: List[np.ndarray],
    cached_qr: Optional[np.ndarray] = None
) -> Set[Tuple[int, ...]]:
    """Iteratively explore all facets using breadth-first search.
    
    Uses a queue-based approach to process each phi vector, exploring all
    possible facet directions and discovering new facets.
    
    Args:
        first_w: Starting vector for exploration.
        principal_W: Basis matrix defining the column space.
        facets: Set of discovered facets (modified in place).
        visited_facets: Set of already-visited facets to avoid duplicates.
        W: List of phi vectors (modified in place as new ones are found).
        cached_qr: Pre-computed QR decomposition of W (optional).
    
    Returns:
        Updated set of all discovered facets.
    """
    m = len(first_w)
    r = principal_W.shape[1]
    
    # Initialize BFS queue with starting phi
    L = deque([first_w])
    
    while L:
        current_w = L.popleft()
        
        # Construct capital_delta matrix for current position
        capital_delta, cached_qr = construct_phi_matrix(
            current_w, r, principal_W, cached_qr
        )
        w_k = capital_delta[:, 0]
        small_delta = capital_delta[:, 1:]
        
        # Explore all direction vectors in both directions
        for delta_l in small_delta.T:
            for sign in [1, -1]:
                direction = sign * delta_l
                
                # Determine constrained coordinates (omega)
                omega = [ i for i in range(m) if w_k[i] == 0 and direction[i] == 0 ]
                facet = get_facet(m, omega)
                
                # Move until hitting a new facet boundary
                new_w, neg_coord = find_orthant_crossing(w_k, direction, omega)
                if new_w is None or neg_coord is None:
                    continue
                
                new_facet = tuple(x for x in facet if x != neg_coord)
                
                # Skip already-visited facets
                if new_facet in visited_facets:
                    continue
                
                # Register new facet and add to exploration queue
                facets.add(new_facet)
                visited_facets.add(new_facet)
                W.append(new_w)
                L.append(new_w)
    
    return facets


def directional_method(
    X: np.ndarray,
    r: int
) -> Tuple[Set[Tuple[int, ...]], np.ndarray]:
    """Find all valid facets of dimension m - r + 1 using the directional method.
    
    Main entry point for the directional facet exploration algorithm. Computes
    the SVD of input matrix X, then systematically explores the orthant structure
    to discover all facets of the target dimension.
    
    Args:
        X: Input data matrix of shape (m, n).
        r: Target rank for factorization.

    
    Returns:
        A tuple containing:
            - facets: Set of all discovered facets, each represented as a tuple of indices.
            - W_hat: Matrix of shape (m, k) where columns are phi vectors, k is number
                      of discovered facets.
    
    Examples:
        >>> X = np.random.rand(10, 20)
        >>> facets, W_hat = directional_method(X, r=3)
        >>> len(facets)  # Number of facets discovered
        45
    """
    # Compute SVD and extract left singular vectors
    U, S, Vt = np.linalg.svd(X, full_matrices=True)
    principal_W = U[:, :r] # principal r-dimensional subapce of X
    
    # Ensure consistent sign convention
    if principal_W[0, 0] < 0:
        principal_W = -principal_W
    
    # Initialize with first column of W
    w_1 = principal_W[:, 0]
    m = len(w_1)
    w_vectors = np.array([principal_W[:, i] for i in range(1, r)])
    
    # Find initial facet
    facets: Set[Tuple[int, ...]] = set()
    initial_facet = tuple(range(m))
    omega: List[int] = []
    
    facets, first_w, omega = find_first_facet(
        w_1, 0, np.array(w_vectors).T,
        r - 1, initial_facet, facets, omega
    )
    
    # Explore all facets starting from initial phi
    W = [first_w]
    facets = explore_facet_dimension(
        first_w, principal_W, facets,
        visited_facets=set(facets),
        W=W
    )
    
    # Stack all w_k vectors into a matrix
    W_hat = np.column_stack(W) if W else np.empty((m, 0))
    
    return facets, W_hat
