"""
DEWS-U: K-Nearest Neighbors with Distance-Weighted Softmax.
"""
from deskit.base.knnbase import KNNBase
from deskit._config import make_finder, resolve_metric, prep_fit_inputs
import numpy as np


class DEWSU(KNNBase):
    """
    DEWS-U: K-Nearest Neighbors with Distance-Weighted Softmax.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Scoring function. Use 'log_loss' or 'prob_correct' with predict_proba()
        output for classification; 'mae', 'mse', or 'rmse' for regression.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Neighborhood size. Default: 10.
    threshold : float
        After per-neighborhood normalization (best=1.0, worst=0.0), models
        below this fraction are excluded from softmax. 0.0 disables the gate;
        1.0 reduces to OLA behavior. Default: 0.5.
    temperature : float, optional
        Softmax sharpness. Lower = sharper routing toward the local best model;
        higher = softer blending. If not set, defaults to 0.5 for regression
        (min-metrics) and 1.0 for classification (max-metrics) at predict time.
    preset : str
        Neighbor search preset. Default: 'balanced'. See list_presets().
    """

    def __init__(self, task, metric='mae', mode='min', k=10,
                 threshold=0.5, temperature=None, preset='balanced', **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, **kwargs)
        super().__init__(metric=metric_fn, mode=mode, neighbor_finder=finder, task=task)
        self.task = task
        self.threshold = threshold
        self._temperature = temperature
        self._metric_name = metric_name

    def fit(self, features, y, preds_dict):
        """
        Fit the routing model on validation data.

        Parameters
        ----------
        features : array-like, shape (n_val, n_features)
            Validation features. Must not overlap with train or test data.
        y : array-like, shape (n_val,)
            Validation ground-truth labels or values.
        preds_dict : dict[str, array-like]
            Validation predictions keyed by model name.
            Shape (n_val,) for scalar metrics; (n_val, n_classes) for probability metrics.
        """
        features, y, preds_dict = prep_fit_inputs(
            features, y, preds_dict, self._metric_name
        )
        super().fit(features, y, preds_dict)

    def _weights_batch(self, x, temperature=None, threshold=None):
        """
        Core weight computation. x is a 2-D float64 numpy array (batch, n_features).
        Returns (batch, n_models) weight array.
        """
        t  = temperature if temperature is not None else (
             self._temperature if self._temperature is not None else
             (0.5 if self.mode == 'min' else 1.0))
        th = threshold if threshold is not None else self.threshold

        _, indices = self.model.kneighbors(x)                        # (batch, k)

        # Average each model's scores over the K neighbors
        avg_scores = self.matrix[indices].mean(axis=1)               # (batch, n_models)

        # Normalize per neighborhood: best = 1.0, worst = 0.0
        local_min = avg_scores.min(axis=1, keepdims=True)
        local_max = avg_scores.max(axis=1, keepdims=True)
        local_range = local_max - local_min
        norm_scores = (avg_scores - local_min) / np.where(local_range > 0, local_range, 1.0)

        # Zero out models below threshold; fall back to single best if none pass
        if th > 0:
            gate = norm_scores >= th
            any_pass = gate.any(axis=1, keepdims=True)
            gate = np.where(any_pass, gate, norm_scores == 1.0)
            norm_scores = norm_scores * gate

        # Softmax
        max_scores = norm_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp((norm_scores - max_scores) / t)
        if th > 0:
            exp_scores = exp_scores * gate
        total = exp_scores.sum(axis=1, keepdims=True)
        weights = np.where(total > 0,
                           exp_scores / np.where(total > 0, total, 1.0),
                           np.full_like(exp_scores, 1.0 / len(self.models)))
        return weights