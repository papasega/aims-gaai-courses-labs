"""Step-by-step scaled dot-product attention on a tiny 3-token example.

The point is to make every intermediate matrix visible so the formula
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
becomes concrete.
"""

import math
import numpy as np


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def manual_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, verbose: bool = True):
    """Compute attention(Q, K, V) step by step on numpy arrays.

    Q, K, V: shape (T, d_k).
    Returns the (T, d_k) output and prints every intermediate matrix.
    """
    d_k = Q.shape[-1]

    if verbose:
        print(f"Q (queries, T={Q.shape[0]}, d_k={d_k}):\n{Q}\n")
        print(f"K (keys, same shape):\n{K}\n")
        print(f"V (values, same shape):\n{V}\n")

    scores = Q @ K.T
    if verbose:
        print(f"step 1, scores = Q K^T:\n{scores}\n")

    scaled = scores / math.sqrt(d_k)
    if verbose:
        print(f"step 2, scaled = scores / sqrt(d_k) (d_k={d_k}, sqrt={math.sqrt(d_k):.3f}):\n{scaled}\n")

    weights = softmax_rows(scaled)
    if verbose:
        print(f"step 3, weights = softmax(scaled), each row sums to 1:\n{weights}\n")
        print(f"row sums: {weights.sum(axis=-1)}\n")

    output = weights @ V
    if verbose:
        print(f"step 4, output = weights V:\n{output}\n")

    return output, weights


def example_3_tokens():
    """Return a fixed 3-token example with simple integer Q, K, V matrices."""
    Q = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    K = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    V = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    # Note: V has d_v=2 here (different from d_k=4), which is fine.
    # Recompute Q/K with d_k matching for the simplest version:
    return Q, K, V
