import numpy as np
import warnings

# FAISS IVF k-means needs at least this many training samples per cell to converge.
_FAISS_MIN_SAMPLES_PER_CELL = 40


# ---------------------------------------------------------------------------
# Distance metric registry
# ---------------------------------------------------------------------------

# Metrics supported by each backend.
# 'euclidean' is the universal default and always available.
#
# Choosing a distance metric:
#   euclidean  – The standard L2 norm. Best default for most tabular data.
#   manhattan  – L1 norm (sum of absolute differences). More robust to outliers
#                and tends to work better in moderately high-dimensional spaces
#                because it doesn't square large differences.
#   chebyshev  – L∞ norm (maximum absolute difference across features). Useful
#                when a single feature dominating the distance is acceptable;
#                common in game-grid / chess-style distance problems.
#   minkowski  – Generalisation of L1/L2 (controlled by p). p=1 → manhattan,
#                p=2 → euclidean. Use when you want to tune between them.
#   cosine     – Angle between vectors, ignoring magnitude. Excellent for
#                embeddings (text, image, audio) where direction matters more
#                than raw scale.

# Metrics that every backend supports natively.
_UNIVERSAL_METRICS = {'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine'}

# Per-backend metric availability.
# KNN (sklearn) supports all scipy metrics — this is the complete curated list.
_KNN_METRICS = _UNIVERSAL_METRICS | {'correlation', 'hamming', 'canberra', 'braycurtis'}

# FAISS only has built-in L2 and inner-product (cosine via normalization).
# All others fall back to a manual compute-then-search path.
_FAISS_NATIVE_METRICS = {'euclidean', 'cosine'}
_FAISS_METRICS = _UNIVERSAL_METRICS  # remainder handled via sklearn fallback

# Annoy metric names (library-specific).
_ANNOY_METRIC_MAP = {
    'euclidean': 'euclidean',
    'manhattan': 'manhattan',
    'cosine':    'angular',
    'hamming':   'hamming',
    # chebyshev and minkowski are not natively supported; we warn the user.
}
_ANNOY_METRICS = set(_ANNOY_METRIC_MAP)

# HNSW (hnswlib) space names.
_HNSW_METRIC_MAP = {
    'euclidean': 'l2',
    'cosine':    'cosine',
    # Others not natively supported in hnswlib; we warn and fall back to l2.
}
_HNSW_METRICS = _UNIVERSAL_METRICS  # partial — see fit() for fallback note

# All metrics callable from the public API.
ALL_METRICS = _KNN_METRICS


def list_distance_metrics():
    """Print all available distance metrics with per-backend availability."""
    print("\nAvailable Distance Metrics:")
    print("=" * 70)
    rows = [
        ("euclidean",  "Default. L2 norm. Best for most tabular data.",                       "all"),
        ("manhattan",  "L1 norm. More robust to outliers; good for high-dim data.",            "all"),
        ("chebyshev",  "L∞ norm. Max absolute diff across features.",                         "KNN only (exact preset)"),
        ("minkowski",  "Generalises L1/L2 via p-param. Set minkowski_p=<float>.",             "KNN only (exact preset)"),
        ("cosine",     "Angle between vectors. Ideal for embeddings (NLP, vision).",           "all"),
        ("correlation","Pearson correlation distance. Good for time series.",                  "KNN only (exact preset)"),
        ("hamming",    "Fraction of differing components. For binary/categorical data.",       "KNN, Annoy"),
        ("canberra",   "Weighted L1. Sensitive to small values near zero.",                    "KNN only (exact preset)"),
        ("braycurtis", "Normalised L1 bounded to [0,1]. Ecological data.",                    "KNN only (exact preset)"),
    ]
    for name, desc, backends in rows:
        print(f"\n  {name:<14}  {desc}")
        print(f"  {'':14}  Backends: {backends}")
    print("\n" + "=" * 70)


class NeighborFinder:
    """Base class for neighbor search backends."""

    def fit(self, X):
        raise NotImplementedError

    def kneighbors(self, X, k=None):
        raise NotImplementedError


class KNNNeighborFinder(NeighborFinder):
    """
    Exact nearest neighbors via sklearn NearestNeighbors.

    Supports all distance metrics in deskit (euclidean, manhattan, chebyshev,
    minkowski, cosine, correlation, hamming, canberra, braycurtis).
    """

    def __init__(self, k=10, distance_metric='euclidean', minkowski_p=2, **kwargs):
        """
        Parameters
        ----------
        k : int
            Number of neighbors.
        distance_metric : str
            Distance function to use. One of the metrics returned by
            list_distance_metrics(). Default: 'euclidean'.
        minkowski_p : float
            The p-parameter for the Minkowski metric (p=1 → manhattan,
            p=2 → euclidean). Ignored for all other metrics.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        metric = distance_metric.lower()
        if metric not in _KNN_METRICS:
            raise ValueError(
                f"distance_metric='{distance_metric}' is not supported by KNNNeighborFinder. "
                f"Available: {sorted(_KNN_METRICS)}."
            )
        self.n_neighbors = k
        self.distance_metric = metric
        self.minkowski_p = minkowski_p
        self.kwargs = kwargs
        self.model = None

    def fit(self, X):
        from sklearn.neighbors import NearestNeighbors
        X = np.atleast_2d(X)
        if X.shape[0] < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{X.shape[0]} samples. Reduce k to at most {X.shape[0]}."
            )
        metric_kwargs = {}
        if self.distance_metric == 'minkowski':
            metric_kwargs['p'] = self.minkowski_p
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.distance_metric,
            metric_params=metric_kwargs if metric_kwargs else None,
            **self.kwargs,
        )
        self.model.fit(X)
        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X)
        if X.shape[0] == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=np.int64)
        return self.model.kneighbors(X, n_neighbors=k)


class FaissNeighborFinder(NeighborFinder):
    """
    Approximate nearest neighbors via FAISS (flat, IVF, or HNSW index).

    Natively supports 'euclidean' and 'cosine'. All other metrics in
    _UNIVERSAL_METRICS fall back to a sklearn-based exact search with a
    warning, so you can still use them without switching presets.
    """

    def __init__(self, k=10, index_type='flat', n_cells=None, n_probes=50,
                 hnsw_M=32, hnsw_efConstruction=400, hnsw_efSearch=200,
                 distance_metric='euclidean'):
        """
        Parameters
        ----------
        distance_metric : str
            'euclidean' (default) or 'cosine'. Other metrics fall back to
            exact sklearn search with a warning — use preset='exact' to avoid
            the overhead.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        metric = distance_metric.lower()
        if metric not in _FAISS_METRICS:
            raise ValueError(
                f"distance_metric='{distance_metric}' is not supported by FaissNeighborFinder. "
                f"Available: {sorted(_FAISS_METRICS)}. "
                f"For other metrics use preset='exact' (KNNNeighborFinder)."
            )
        self.n_neighbors = k
        self.index_type = index_type.lower()
        self.n_cells = n_cells
        self.n_probes = n_probes
        self.hnsw_M = hnsw_M
        self.hnsw_efConstruction = hnsw_efConstruction
        self.hnsw_efSearch = hnsw_efSearch
        self.distance_metric = metric
        self.index_ = None
        self._fallback_finder = None   # used for non-native metrics
        self._check_availability()

    def _check_availability(self):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not found. Install with: pip install faiss-cpu")

    @staticmethod
    def _l2_normalize(X):
        """Row-wise L2 normalisation in-place (for cosine similarity)."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms.astype(np.float32)

    def fit(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        n_samples, dim = X.shape

        if n_samples < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )

        # Non-native metrics: delegate entirely to KNNNeighborFinder.
        if self.distance_metric not in _FAISS_NATIVE_METRICS:
            warnings.warn(
                f"distance_metric='{self.distance_metric}' is not natively supported by "
                f"FAISS. Falling back to exact sklearn KNN for this metric. "
                f"Use preset='exact' to avoid this overhead.",
                UserWarning,
            )
            self._fallback_finder = KNNNeighborFinder(
                k=self.n_neighbors, distance_metric=self.distance_metric
            )
            self._fallback_finder.fit(X)
            return self

        # Cosine similarity: normalise all vectors, then use inner-product index.
        if self.distance_metric == 'cosine':
            X = self._l2_normalize(X)

        if self.index_type == 'flat':
            if dim <= 2:
                warnings.warn(
                    f"FAISS Flat may have floating-point precision issues in {dim}D. "
                    f"Consider KNNNeighborFinder for low-dimensional data.",
                    UserWarning
                )
            if self.distance_metric == 'cosine':
                self.index_ = self.faiss.IndexFlatIP(dim)   # inner product on normalised vecs
            else:
                self.index_ = self.faiss.IndexFlatL2(dim)
            self.index_.add(X)

        elif self.index_type == 'ivf':
            if self.n_cells is None:
                self.n_cells = min(int(np.sqrt(n_samples)), 4096)

            min_required = self.n_cells * _FAISS_MIN_SAMPLES_PER_CELL
            if n_samples < min_required:
                safe_cells = max(1, n_samples // _FAISS_MIN_SAMPLES_PER_CELL)
                warnings.warn(
                    f"n_cells={self.n_cells} requires {min_required} samples but only "
                    f"{n_samples} provided. Reducing to {safe_cells} to prevent hanging. "
                    f"Consider index_type='flat' or KNNNeighborFinder for small datasets.",
                    UserWarning
                )
                self.n_cells = safe_cells

            effective_probes = min(self.n_probes, self.n_cells)
            if effective_probes < self.n_probes:
                warnings.warn(
                    f"n_probes={self.n_probes} exceeds n_cells={self.n_cells}. "
                    f"Clamping to {self.n_cells}.",
                    UserWarning
                )
            if effective_probes < self.n_cells * 0.1:
                warnings.warn(
                    f"n_probes={effective_probes} is below 10% of n_cells={self.n_cells}. "
                    f"Recall may be poor. Consider n_probes >= {max(1, int(self.n_cells * 0.1))}.",
                    UserWarning
                )

            if self.distance_metric == 'cosine':
                quantizer = self.faiss.IndexFlatIP(dim)
                self.index_ = self.faiss.IndexIVFFlat(
                    quantizer, dim, self.n_cells, self.faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = self.faiss.IndexFlatL2(dim)
                self.index_ = self.faiss.IndexIVFFlat(quantizer, dim, self.n_cells)
            self.index_.train(X)
            self.index_.add(X)
            self.index_.nprobe = effective_probes

        elif self.index_type == 'hnsw':
            if n_samples >= 10000 and self.hnsw_efConstruction < 300:
                warnings.warn(
                    f"ef_construction={self.hnsw_efConstruction} may be too low for "
                    f"{n_samples} samples. Consider ef_construction >= 400.",
                    UserWarning
                )
            self.index_ = self.faiss.IndexHNSWFlat(dim, self.hnsw_M)
            self.index_.hnsw.efConstruction = self.hnsw_efConstruction
            self.index_.hnsw.efSearch = self.hnsw_efSearch
            self.index_.add(X)

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X).astype(np.float32)
        if X.shape[0] == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        # Non-native metric fallback.
        if self._fallback_finder is not None:
            return self._fallback_finder.kneighbors(X, k=k)

        if self.distance_metric == 'cosine':
            X = self._l2_normalize(X)
            scores, indices = self.index_.search(X, k)
            # Inner product on normalised vectors: similarity ∈ [-1, 1].
            # Convert to a proper distance (0 = identical, 2 = opposite).
            distances = 1.0 - scores
        else:
            distances, indices = self.index_.search(X, k)
            # FAISS returns squared L2; clamp to 0 before sqrt.
            distances = np.sqrt(np.maximum(distances, 0))

        return distances.astype(np.float32), indices


class AnnoyNeighborFinder(NeighborFinder):
    """
    Approximate nearest neighbors via Annoy.

    Supports: euclidean, manhattan, cosine, hamming.
    chebyshev and minkowski are not available in Annoy — use preset='exact' for those.
    """

    def __init__(self, k=10, n_trees=100, distance_metric='euclidean', search_k=-1):
        """
        Parameters
        ----------
        distance_metric : str
            One of 'euclidean', 'manhattan', 'cosine', 'hamming'. Default: 'euclidean'.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        metric = distance_metric.lower()
        if metric not in _ANNOY_METRICS:
            raise ValueError(
                f"distance_metric='{distance_metric}' is not supported by AnnoyNeighborFinder. "
                f"Available: {sorted(_ANNOY_METRICS)}. "
                f"For chebyshev or minkowski use preset='exact' (KNNNeighborFinder)."
            )
        self.k = k
        self.n_trees = n_trees
        self.distance_metric = metric
        # Annoy's recommended default
        self.search_k = n_trees * k if search_k == -1 else search_k
        self.index_ = None
        self.n_samples_ = None
        self._check_availability()

    def _check_availability(self):
        try:
            from annoy import AnnoyIndex
            self.AnnoyIndex = AnnoyIndex
        except ImportError:
            raise ImportError("Annoy not found. Install with: pip install annoy")

    def fit(self, X):
        X = np.atleast_2d(X)
        n_samples, dim = X.shape
        self.n_samples_ = n_samples

        if n_samples < self.k:
            raise ValueError(
                f"Cannot find {self.k} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )
        if dim <= 3:
            warnings.warn(
                f"Annoy tree structure can degenerate in {dim}D. "
                f"Consider KNNNeighborFinder for low-dimensional data.",
                UserWarning
            )

        self.index_ = self.AnnoyIndex(dim, _ANNOY_METRIC_MAP[self.distance_metric])
        for i, vec in enumerate(X):
            self.index_.add_item(i, vec.tolist())
        self.index_.build(self.n_trees)

        # Verify the index returns the expected number of neighbors
        test_vec = X[0].tolist()
        test_result = self.index_.get_nns_by_vector(test_vec, self.k, search_k=self.search_k)
        if len(test_result) < self.k:
            raise RuntimeError(
                f"Annoy index returned {len(test_result)} neighbors but {self.k} were "
                f"requested. This is a known Annoy bug on Apple Silicon (M1/M2/M3) — "
                f"the package does not work correctly on ARM64. "
                f"Use preset='fast' (FAISS IVF) or preset='exact' (sklearn KNN) instead."
            )

        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.k
        X = np.atleast_2d(X)
        if X.shape[0] == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=np.int64)

        all_indices, all_distances = [], []
        for vec in X:
            idx, dist = self.index_.get_nns_by_vector(
                vec.tolist(), k, search_k=self.search_k, include_distances=True
            )
            if len(idx) != k:
                raise ValueError(
                    f"Annoy returned {len(idx)} neighbors but {k} were requested. "
                    f"Try increasing n_trees (current: {self.n_trees}) or search_k "
                    f"(current: {self.search_k}), or use KNNNeighborFinder."
                )
            all_indices.append(idx)
            all_distances.append(dist)

        return np.array(all_distances), np.array(all_indices)


class HNSWNeighborFinder(NeighborFinder):
    """
    Approximate nearest neighbors via HNSW (hnswlib or nmslib backend).

    Natively supports 'euclidean' and 'cosine'. Manhattan, chebyshev, and
    minkowski are not available in hnswlib/nmslib — use preset='exact' for those.
    """

    def __init__(self, k=10, M=32, ef_construction=400,
                 ef_search=200, backend='hnswlib', distance_metric='euclidean'):
        """
        Parameters
        ----------
        distance_metric : str
            'euclidean' (default) or 'cosine'. Other metrics are not supported
            natively and will raise an error — use preset='exact' instead.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got k={k}")
        metric = distance_metric.lower()
        if metric not in _HNSW_METRIC_MAP:
            raise ValueError(
                f"distance_metric='{distance_metric}' is not natively supported by "
                f"HNSWNeighborFinder. Available: {sorted(_HNSW_METRIC_MAP)}. "
                f"For other metrics use preset='exact' (KNNNeighborFinder)."
            )
        self.n_neighbors = k
        self.distance_metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.backend = backend.lower()
        self.index_ = None
        self._check_availability()

    def _check_availability(self):
        if self.backend == 'hnswlib':
            try:
                import hnswlib
                self.hnswlib = hnswlib
            except ImportError:
                raise ImportError("hnswlib not found. Install with: pip install hnswlib")
        elif self.backend == 'nmslib':
            try:
                import nmslib
                self.nmslib = nmslib
            except ImportError:
                raise ImportError("nmslib not found. Install with: pip install nmslib")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def fit(self, X):
        X = np.atleast_2d(X).astype(np.float32)
        n_samples, dim = X.shape

        if n_samples < self.n_neighbors:
            raise ValueError(
                f"Cannot find {self.n_neighbors} neighbors in a dataset with only "
                f"{n_samples} samples. Reduce k to at most {n_samples}."
            )
        if n_samples >= 10000 and self.ef_construction < 300:
            warnings.warn(
                f"ef_construction={self.ef_construction} may be too low for "
                f"{n_samples} samples. Consider ef_construction >= 400.",
                UserWarning
            )

        if self.backend == 'hnswlib':
            space = _HNSW_METRIC_MAP[self.distance_metric]
            self.index_ = self.hnswlib.Index(space=space, dim=dim)
            self.index_.init_index(
                max_elements=n_samples, M=self.M, ef_construction=self.ef_construction
            )
            self.index_.set_ef(self.ef_search)
            self.index_.add_items(X, np.arange(n_samples))

        else:  # nmslib
            nmslib_space_map = {
                'euclidean': 'l2',
                'cosine':    'cosinesimil',
            }
            self.index_ = self.nmslib.init(
                method='hnsw',
                space=nmslib_space_map.get(self.distance_metric, 'l2'),
                data_type=self.nmslib.DataType.DENSE_VECTOR
            )
            self.index_.addDataPointBatch(X)
            self.index_.createIndex(
                {'M': self.M, 'efConstruction': self.ef_construction, 'post': 0},
                print_progress=False
            )
            self.index_.setQueryTimeParams({'efSearch': self.ef_search})

        return self

    def kneighbors(self, X, k=None):
        """Return (distances, indices) of shape (batch_size, k)."""
        if k is None:
            k = self.n_neighbors
        X = np.atleast_2d(X).astype(np.float32)
        if X.shape[0] == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)

        if self.backend == 'nmslib':
            results = self.index_.knnQueryBatch(X, k=k)
            return (
                np.array([dist for _, dist in results]),
                np.array([idx  for idx, _ in results]),
            )

        # hnswlib returns (indices, distances).
        indices, distances = self.index_.knn_query(X, k=k)
        return distances, indices