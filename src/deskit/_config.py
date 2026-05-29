"""
Internal helpers shared across algorithm classes.
Not part of the public API.
"""
import numpy as np
from deskit.metrics import _METRICS, _PROB_METRICS, _SCALAR_METRICS
from deskit.utils import to_numpy


# ---------------------------------------------------------------------------
# Speed / accuracy presets
# ---------------------------------------------------------------------------

SPEED_PRESETS = {
    'exact': {
        'description': 'Exact nearest neighbors — slowest but 100% accurate',
        'finder': 'knn',
        'kwargs': {},
    },
    'balanced': {
        'description': 'Good balance of speed and accuracy (~98% recall)',
        'finder': 'faiss',
        'kwargs': {'index_type': 'ivf', 'n_probes': 50},
    },
    'fast': {
        'description': 'Fast queries with good accuracy (~95% recall)',
        'finder': 'faiss',
        'kwargs': {'index_type': 'ivf', 'n_probes': 30},
    },
    'turbo': {
        'description': 'Maximum speed, exact results — FAISS flat index',
        'finder': 'faiss',
        'kwargs': {'index_type': 'flat'},
    },
    'high_dim_balanced': {
        'description': 'High-dimensional data (>100D), balanced',
        'finder': 'hnsw',
        'kwargs': {'backend': 'hnswlib', 'M': 32, 'ef_construction': 400, 'ef_search': 200},
    },
    'high_dim_fast': {
        'description': 'High-dimensional data (>100D), fast',
        'finder': 'hnsw',
        'kwargs': {'backend': 'hnswlib', 'M': 16, 'ef_construction': 200, 'ef_search': 100},
    },
}


def list_presets():
    """Print all available presets with descriptions and parameters."""
    print("\nAvailable Speed/Accuracy Presets:")
    print("=" * 70)
    for name, config in SPEED_PRESETS.items():
        print(f"\n{name.upper()}\n  {config['description']}\n  Finder: {config['finder']}")
        if config['kwargs']:
            print(f"  Parameters: {config['kwargs']}")
    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

def resolve_metric(metric):
    """
    Convert a metric string or callable to (name_or_None, callable).

    Returns
    -------
    metric_name : str or None
        String name if metric was passed as a string; None for callables.
        Used later in validate_fit_inputs to check prediction shape.
    metric_fn : callable
        The actual scoring function.
    """
    if isinstance(metric, str):
        name = metric.lower()
        if name not in _METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Built-in options: {sorted(_METRICS)}. "
                f"Pass a callable for custom metrics."
            )
        return name, _METRICS[name]
    return None, metric


# ---------------------------------------------------------------------------
# Neighbor finder construction
# ---------------------------------------------------------------------------

def make_finder(preset, k, finder=None, distance_metric='euclidean', **kwargs):
    """
    Create a NeighborFinder from a preset name or custom finder string.

    Parameters
    ----------
    preset : str
        One of the keys in SPEED_PRESETS, or 'custom'.
    k : int
        Number of neighbors.
    finder : str, optional
        Required when preset='custom'. One of 'knn', 'faiss', 'annoy', 'hnsw'.
    distance_metric : str
        Distance function to use. Default: 'euclidean'. See
        neighbors.list_distance_metrics() for all options and per-backend
        availability.

        Quick guide:
          'euclidean'  — Best default for most tabular data.
          'manhattan'  — More robust to outliers; good for moderate-to-high dims.
          'chebyshev'  — Max abs diff; use preset='exact' (KNN only).
          'minkowski'  — L1/L2 generalisation; use preset='exact' (KNN only).
          'cosine'     — Direction-based; ideal for embeddings (NLP, vision).
    **kwargs
        Forwarded to the finder constructor (e.g. index_type, n_probes).
    """
    if preset == 'custom':
        if finder is None:
            raise ValueError("Must specify 'finder' when using preset='custom'.")
        finder_type = finder.lower()
        finder_kwargs = {'k': k, 'distance_metric': distance_metric, **kwargs}
    else:
        if preset not in SPEED_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {sorted(SPEED_PRESETS)}. "
                f"Or use preset='custom' with an explicit finder."
            )
        config = SPEED_PRESETS[preset]
        finder_type = config['finder']
        finder_kwargs = {**config['kwargs'], 'k': k, 'distance_metric': distance_metric, **kwargs}

    if finder_type == 'knn':
        from deskit.neighbors import KNNNeighborFinder
        return KNNNeighborFinder(**finder_kwargs)
    elif finder_type == 'faiss':
        from deskit.neighbors import FaissNeighborFinder
        return FaissNeighborFinder(**finder_kwargs)
    elif finder_type == 'annoy':
        from deskit.neighbors import AnnoyNeighborFinder
        return AnnoyNeighborFinder(**finder_kwargs)
    elif finder_type == 'hnsw':
        from deskit.neighbors import HNSWNeighborFinder
        return HNSWNeighborFinder(**finder_kwargs)
    else:
        raise ValueError(f"Unknown finder '{finder_type}'.")

# fit() input validation

def prep_fit_inputs(features, y, preds_dict, metric_name):
    """
    Convert all fit() inputs to numpy arrays and validate consistency.

    Returns
    -------
    features, y, preds_dict — all as numpy arrays, ready for KNNBase.fit().
    """
    features = to_numpy(features)
    y = to_numpy(y)
    preds_dict = {name: to_numpy(p) for name, p in preds_dict.items()}

    n = len(y)

    if len(features) != n:
        raise ValueError(
            f"features has {len(features)} rows but y has {n} samples. "
            f"All arrays passed to fit() must have the same number of rows."
        )

    for name, preds in preds_dict.items():
        if len(preds) != n:
            raise ValueError(
                f"preds_dict['{name}'] has {len(preds)} samples but y has {n}. "
                f"Every prediction array must align row-for-row with y."
            )

    if metric_name is not None:
        sample_preds = next(iter(preds_dict.values()))
        pred_ndim    = np.asarray(sample_preds).ndim

        if metric_name in _PROB_METRICS and pred_ndim != 2:
            raise ValueError(
                f"Metric '{metric_name}' expects probability arrays of shape "
                f"(n_samples, n_classes), but received a {pred_ndim}D array. "
                f"Pass predict_proba() output instead of predict() output."
            )

        if metric_name in _SCALAR_METRICS and pred_ndim != 1:
            raise ValueError(
                f"Metric '{metric_name}' expects scalar predictions of shape "
                f"(n_samples,), but received a {pred_ndim}D array. "
                f"Pass predict() output, or switch to 'log_loss' / 'prob_correct' "
                f"if you intended to use class probabilities."
            )

    return features, y, preds_dict