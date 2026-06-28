"""
OLA: Overall Local Accuracy.
"""
from deskit.base.knnbase import KNNBase
from deskit._config import make_finder, resolve_metric, prep_fit_inputs
import numpy as np


class OLA(KNNBase):
    """
    OLA: Overall Local Accuracy.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Scoring function.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Neighborhood size. Default: 10.
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
                 preset='balanced', threshold=None, distance_metric='euclidean', **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, **kwargs)
        super().__init__(metric=metric_fn, mode=mode, neighbor_finder=finder, task=task)
        self.task = task
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
        # Global normalization to [0, 1]
        mat_min, mat_max = self.matrix.min(), self.matrix.max()
        if mat_max > mat_min:
            self.matrix = (self.matrix - mat_min) / (mat_max - mat_min)

    def _weights_batch(self, x, temperature=None, threshold=None, k=None, loo=False):
        """
        Core weight computation. x is a 2-D float64 numpy array (batch, n_features).
        Returns (batch, n_models) weight array.
        temperature and threshold are accepted for API compatibility but
        OLA always selects a single model via argmax.
        """
        batch_size = x.shape[0]

        _, indices  = self._kneighbors(x, k=k, loo=loo)
        avg_scores  = self.matrix[indices].mean(axis=1)               # (batch, n_models)
        best_indices = np.argmax(avg_scores, axis=1)

        weights = np.zeros((batch_size, len(self.models)))
        weights[np.arange(batch_size), best_indices] = 1.0
        return weights