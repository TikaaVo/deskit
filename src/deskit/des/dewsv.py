"""
DEWS-V: Distance-weighted Ensemble with Softmax — Variance-penalised.
"""
from deskit.base.knnbase import KNNBase
from deskit._config import make_finder, resolve_metric, prep_fit_inputs
import numpy as np


_SIGNED_METRICS = {'mae', 'mse'}


def _signed_residual(y_true, y_pred):
    return float(y_true) - float(y_pred)


class DEWSV(KNNBase):
    """
    DEWS-V: Distance-weighted Ensemble with Softmax — Variance-penalised.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Scoring function. 'mae' or 'mse' activate signed-residual variance;
        all other metrics use the score matrix directly for variance.
    mode : str
        'max' if higher scores are better, 'min' if lower.
    k : int
        Neighbourhood size. Default: 10.
    threshold : float
        Competence gate. After per-neighbourhood normalisation (best=1.0,
        worst=0.0), models below this fraction are excluded from softmax.
        0.0 disables the gate; 1.0 reduces to OLA behaviour. Default: 0.5.
    temperature : float, optional
        Softmax sharpness. Lower = sharper routing toward the local best model.
        Defaults to 0.5 for min-metrics, 1.0 otherwise.
    preset : str
        Neighbour search preset. Default: 'balanced'. See list_presets().
    distance_metric : str
        Distance function to use for neighbor search. Default: 'euclidean'. See
        neighbors.list_distance_metrics() for all options and per-backend availability. 
    """

    def __init__(self, task, metric='mae', mode='min', k=10,
                 threshold=0.5, temperature=None, preset='balanced', distance_metric='euclidean', **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, distance_metric=distance_metric, **kwargs)

        self._use_signed = metric_name in _SIGNED_METRICS
        self._metric_name = metric_name

        super().__init__(metric=metric_fn, mode=mode, neighbor_finder=finder, task=task)

        self.task = task
        self.threshold = threshold
        self._temperature = temperature
        self._var_matrix = None   # (n_val, n_models) signed residuals, MAE/MSE only

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
        """
        features, y, preds_dict = prep_fit_inputs(
            features, y, preds_dict, self._metric_name
        )
        super().fit(features, y, preds_dict)

        # Build signed residual matrix for variance (MAE/MSE only)
        if self._use_signed:
            n_val    = len(y)
            n_models = len(self.models)
            self._var_matrix = np.zeros((n_val, n_models))
            for j, name in enumerate(self.models):
                preds = np.asarray(preds_dict[name])
                self._var_matrix[:, j] = np.vectorize(_signed_residual)(y, preds)

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

        # Uniform average of each model's scores over K neighbors
        neighbor_scores = self.matrix[indices]                        # (batch, k, n_models)
        avg_scores      = neighbor_scores.mean(axis=1)                # (batch, n_models)

        # Variance of signed residuals (MAE/MSE) or score matrix
        var_scores = self._var_matrix[indices] if self._use_signed else neighbor_scores
        local_var  = var_scores.var(axis=1)                           # (batch, n_models)

        # Normalize scores to [0, 1]
        local_min   = avg_scores.min(axis=1, keepdims=True)
        local_max   = avg_scores.max(axis=1, keepdims=True)
        local_range = local_max - local_min
        norm_scores = (avg_scores - local_min) / np.where(local_range > 0, local_range, 1.0)

        # Normalize variance to [0, 1]
        var_min   = local_var.min(axis=1, keepdims=True)
        var_max   = local_var.max(axis=1, keepdims=True)
        var_range = var_max - var_min
        norm_var  = (local_var - var_min) / np.where(var_range > 0, var_range, 1.0)

        # Penalise inconsistent models, then re-normalize
        norm_scores = norm_scores / (1.0 + norm_var)
        local_min   = norm_scores.min(axis=1, keepdims=True)
        local_max   = norm_scores.max(axis=1, keepdims=True)
        local_range = local_max - local_min
        norm_scores = (norm_scores - local_min) / np.where(local_range > 0, local_range, 1.0)

        # Zero out models below threshold; fall back to single best if none pass
        if th > 0:
            gate      = norm_scores >= th
            any_pass  = gate.any(axis=1, keepdims=True)
            gate      = np.where(any_pass, gate, norm_scores == 1.0)
            norm_scores = norm_scores * gate

        # Softmax
        max_scores = norm_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp((norm_scores - max_scores) / t)
        if th > 0:
            exp_scores = exp_scores * gate
        total   = exp_scores.sum(axis=1, keepdims=True)
        weights = np.where(total > 0,
                           exp_scores / np.where(total > 0, total, 1.0),
                           np.full_like(exp_scores, 1.0 / len(self.models)))
        return weights