"""
KNORA-U: K-Nearest Oracles — Union.
"""
from deskit.base.knnbase import KNNBase
from deskit._config import make_finder, resolve_metric, prep_fit_inputs
import numpy as np


class KNORAU(KNNBase):
    """
    KNORA-U: K-Nearest Oracles — Union.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Recommended: 'log_loss' for classification.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Neighborhood size. Default: 10.
    threshold : float
        Per-neighbor competence cutoff on the [0, 1] normalized scale
        (1.0 = best model on that neighbor, 0.0 = worst).
        Classification with log_loss: 0.5 (default).
        Regression: use 1.0.
    preset : str
        Neighbor search preset. Default: 'balanced'. See list_presets().
    distance_metric : str
        Distance function to use for neighbor search. Default: 'euclidean'. See
        neighbors.list_distance_metrics() for all options and per-backend availability.
    loo: bool
        Enables Leave One Out (LOO) for hyperparameter tuning on the DSEL set. Default: 'false'.
        Ignores closest neighbor with a negligible distance to avoid overfitting.
    """

    def __init__(self, task, metric='mae', mode='min', k=10,
                 threshold=0.5, preset='balanced', distance_metric='euclidean', **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, **kwargs)
        super().__init__(metric=metric_fn, mode=mode, neighbor_finder=finder, task=task)
        self.task = task
        self.threshold = threshold
        self._metric_name = metric_name

    def fit(self, features, y, preds_dict):
        """
        Fit the routing model on validation data.

        Parameters
        ----------
        features : array-like, shape (n_val, n_features)
        y : array-like, shape (n_val,)
        preds_dict : dict[str, array-like]
        """
        features, y, preds_dict = prep_fit_inputs(
            features, y, preds_dict, self._metric_name
        )
        super().fit(features, y, preds_dict)

    def _weights_batch(self, x, temperature=None, threshold=None, k=None, loo=False):
        """
        Core weight computation. x is a 2-D float64 numpy array (batch, n_features).
        Returns (batch, n_models) weight array.
        temperature is accepted for API compatibility but has no effect.
        """
        th = threshold if threshold is not None else self.threshold

        _, indices      = self._kneighbors(x, k=k, loo=loo)
        neighbor_scores = self.matrix[indices]                        # (batch, k, n_models)

        # Normalize per neighbor: best model = 1.0, worst = 0.0
        n_min   = neighbor_scores.min(axis=2, keepdims=True)
        n_max   = neighbor_scores.max(axis=2, keepdims=True)
        n_range = n_max - n_min
        norm    = np.where(n_range > 0,
                           (neighbor_scores - n_min) / n_range,
                           1.0)   # tied → all equally competent

        # votes[b, j] = number of neighbours where model j exceeds the threshold
        votes       = (norm >= th).sum(axis=1).astype(float)         # (batch, n_models)
        total_votes = votes.sum(axis=1, keepdims=True)

        any_votes = total_votes > 0
        weights   = np.where(
            any_votes,
            votes / np.where(any_votes, total_votes, 1.0),
            np.full_like(votes, 1.0 / len(self.models)),
        )
        return weights