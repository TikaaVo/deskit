"""
Normalization functions for Dynamic Ensemble Selection.

All functions operate row-wise: each row corresponds to a query instance,
and columns are the base models. They return a matrix of the same shape,
with values in a suitable range (usually [0,1]) and "higher = better".

Built‑in options:
    'minmax'   : classic min‑max scaling per row → [0,1]
    'zscore'   : z‑score per row (mean 0, std 1) – may be negative
    'bestrel'  : ratio‑to‑best, automatically handles sign -> [0,1]:
                 - if scores are all negative (e.g., -errors): best / score
                 - if scores are all non‑negative (e.g., V/IV): score / best
    'softmax'  : softmax with temperature=1 (optional)
    'rank'     : rank‑based (0 for worst, 1 for best) – robust to outliers

Custom functions can be passed via the `normalization` parameter in DEWS.
"""

import numpy as np

EPS = 1e-12

def minmax(x):
    """Row‑wise min‑max scaling to [0,1]."""
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    x_range = x_max - x_min
    # Avoid division by zero (if range is 0, set to 1.0)
    safe_range = np.where(x_range > 0, x_range, 1.0)
    return (x - x_min) / safe_range

def zscore(x):
    """Row‑wise z‑score (standard score) with bias‑corrected std."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True, ddof=1)  # sample std
    safe_std = np.where(std > EPS, std, 1.0)
    return (x - mean) / safe_std

def bestrel(x):
    """
    Ratio‑to‑best normalization.

    """
    max_val = x.max(axis=1, keepdims=True)
    # Determine per‑row which normalization to use based on the max
    eps = 1e-12
    safe_x = np.minimum(x, -eps)
    norm_neg = max_val / safe_x   # this will be <=1 because max is the least negative
    safe_max = np.where(max_val > eps, max_val, eps)
    norm_pos = x / safe_max       # will be in [0,1]

    neg_mask = max_val < -eps
    result = np.where(neg_mask, norm_neg, norm_pos)
    return result

def softmax(x, temperature=1.0):
    """Row‑wise softmax with optional temperature (higher temperature → more uniform)."""
    x_scaled = x / temperature
    x_shifted = x_scaled - x_scaled.max(axis=1, keepdims=True)  # stability
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def rank_norm(x):
    """
    Convert each row to fractional ranks: worst → 0, best → 1.
    Ties are averaged.
    """
    # Get ranks along axis=1 (method='average' gives average rank for ties)
    ranks = x.argsort(axis=1).argsort(axis=1) + 1  # ranks from 1 to M
    # Convert to [0,1] with 1 for best (highest original value)
    M = x.shape[1]
    # For 'higher is better', we invert: rank 1 (lowest original) gets 0, rank M (highest) gets 1
    return (ranks - 1) / (M - 1) if M > 1 else np.ones_like(x) * 0.5

_NORMS = {
    'minmax': minmax,
    'z': zscore,
    'bestrel': bestrel,
    'softmax': softmax,
    'rank_norm': rank_norm
}