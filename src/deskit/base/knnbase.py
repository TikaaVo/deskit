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

        self.model.fit(features)