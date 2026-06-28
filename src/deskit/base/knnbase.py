from deskit.base.base import BaseRouter
from deskit.base.predictbase import PredictBase
import numpy as np


class KNNBase(PredictBase, BaseRouter):
    """
    Base for KNN-based DES algorithms.

    Inheriting PredictBase gives every subclass the public
    predict() and predict_weights() API automatically.
    Subclasses must implement _weights_batch().
    """

    def __init__(self, metric, mode='max', neighbor_finder=None, task='classification'):
        """
        Parameters
        ----------
        metric : callable
            Per-sample scoring function: (y_true, y_pred) -> float.
        mode : str
            'max' if higher scores are better, 'min' if lower.
        neighbor_finder : NeighborFinder
            Backend used for neighborhood queries.
        """
        self.metric          = metric
        self.mode            = mode
        self.model           = neighbor_finder
        self.matrix          = None   # (n_val, n_models); higher is always better
        self.models          = None   # ordered list of model names
        self.task = task

    def _compute_scores(self, y, preds):
        """
        Return a 1D array of per-sample metric scores.

        preds may be 1D (scalar predictions) or 2D (probability arrays, one
        row per sample).
        """
        preds = np.asarray(preds)
        if preds.ndim == 2:
            return np.array([self.metric(y[i], preds[i]) for i in range(len(y))])
        return np.vectorize(self.metric)(y, preds)

    def fit(self, features, y, preds_dict):
        """
        Build the score matrix and fit the neighbor index.

        This method expects pre-validated numpy arrays.
        """
        self.models = list(preds_dict.keys())
        n_val       = len(y)
        n_models    = len(self.models)
        self.matrix = np.zeros((n_val, n_models))

        for j, name in enumerate(self.models):
            scores = self._compute_scores(y, preds_dict[name])
            self.matrix[:, j] = scores if self.mode == 'max' else -scores

        if self.task == 'classification':
            self.classes_ = np.unique(y)
        else:
            self.classes_ = None

        self.model.fit(features)

    def _kneighbors(self, x, k=None, loo=False):
        """
        Query the fitted neighbor index, with optional leave-one-out (LOO)
        exclusion of each query point's own occurrence in the DSEL.

        Set loo=True when ``x`` is (part of) the same data this model was
        fit on -- e.g. while tuning k / threshold / temperature directly on
        the DSEL -- so a point doesn't end up neighboring itself at distance
        0, which would otherwise dominate the routing.

        Parameters
        ----------
        x : np.ndarray, shape (batch, n_features)
        k : int, optional
            Neighborhood size. None defers to the finder's own default.
        loo : bool
            If True, query one extra neighbor per row and drop the
            zero-distance match (the point itself) when present, so the
            returned neighborhood still has the size a normal call would
            have produced. Rows with no zero-distance match (e.g. ``x``
            isn't actually part of the fitted DSEL) fall back to dropping
            the farthest neighbor instead, so shapes stay consistent.

        Returns
        -------
        distances, indices : np.ndarray, each shape (batch, k_eff)
        """
        if not loo:
            return self.model.kneighbors(x, k=k)

        # Different backends store their default k under different
        # attribute names (n_neighbors vs k), so rather than guessing,
        # resolve the effective k with one cheap probe call when it isn't
        # given explicitly. Costs one extra query only in that case.
        if k is None:
            probe_distances, _ = self.model.kneighbors(x, k=k)
            k = probe_distances.shape[1]

        distances, indices = self.model.kneighbors(x, k=k + 1)
        return _drop_self_match(distances, indices, k)


def _drop_self_match(distances, indices, k, eps=1e-6):
    """
    Drop one zero-distance neighbor per row from (batch, k+1) neighbor
    results, returning (batch, k) arrays.

    A query point's own occurrence in the DSEL, if present, is always the
    closest possible neighbor (distance 0 is the global minimum for any
    proper distance metric), so it's identified per row as whichever
    column is nearest, gated on that distance being ~0. Rows without a
    zero-distance match (point not actually in the DSEL) drop the
    farthest neighbor instead, so every row keeps exactly k entries.

    Note: if the DSEL contains true duplicate feature rows, only one
    occurrence is dropped per query point -- the duplicates remain valid,
    distinct neighbors. Distance order within each row need not be
    sorted; this works either way.
    """
    batch = distances.shape[0]
    rows = np.arange(batch)

    nearest_col = np.argmin(distances, axis=1)
    is_self_match = distances[rows, nearest_col] < eps

    farthest_col = np.argmax(distances, axis=1)
    drop_col = np.where(is_self_match, nearest_col, farthest_col)

    keep = np.ones(distances.shape, dtype=bool)
    keep[rows, drop_col] = False

    new_distances = distances[keep].reshape(batch, k)
    new_indices = indices[keep].reshape(batch, k)
    return new_distances, new_indices