"""
DEWS-T: Distance-weighted Ensemble with Softmax — Trend.
"""
from deskit.base.knnbase import KNNBase
from deskit._config import make_finder, resolve_metric, prep_fit_inputs
import numpy as np


_SIGNED_METRICS = {'mae', 'mse'}


def _signed_residual(y_true, y_pred):
    return float(y_true) - float(y_pred)


class DEWST(KNNBase):
    """
    DEWS-T: Distance-weighted Ensemble with Softmax — Trend.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str or callable
        Scoring function. 'mae' or 'mse' activate signed-residual mode;
        all other metrics are trended directly.
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
    r2_threshold : float
        Minimum weighted R² for the trend line to be trusted. Below this value
        the sample falls back to DEWS-I scoring for that model. Default: 0.7.
    preset : str
        Neighbour search preset. Default: 'balanced'. See list_presets().
    distance_metric : str
        Distance function to use for neighbor search. Default: 'euclidean'. See
        neighbors.list_distance_metrics() for all options and per-backend availability.
    """

    def __init__(self, task, metric='mae', mode='min', k=10,
                 threshold=0.5, temperature=None, r2_threshold=0.7,
                 preset='balanced', distance_metric='euclidean', **kwargs):
        metric_name, metric_fn = resolve_metric(metric)
        finder = make_finder(preset, k, **kwargs)

        self._use_signed = metric_name in _SIGNED_METRICS
        self._metric_name = metric_name
        self._convert = {'mae': np.abs, 'mse': np.square}.get(metric_name)

        # For signed metrics use signed residuals internally
        super().__init__(
            metric=_signed_residual if self._use_signed else metric_fn,
            mode='max' if self._use_signed else mode,
            neighbor_finder=finder, task=task
        )

        self._real_mode = mode
        self.task = task
        self.threshold = threshold
        self._temperature = temperature
        self.r2_threshold = r2_threshold

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

    def _weights_batch(self, x, temperature=None, threshold=None, k=None, r2_threshold=None):
        """
        Core weight computation. x is a 2-D float64 numpy array (batch, n_features).
        Returns (batch, n_models) weight array.
        """
        t  = temperature if temperature is not None else (
             self._temperature if self._temperature is not None else
             (0.5 if self._real_mode == 'min' else 1.0))
        th    = threshold    if threshold    is not None else self.threshold
        r2_th = r2_threshold if r2_threshold is not None else self.r2_threshold

        distances, indices = self.model.kneighbors(x, k=k)          # (batch, k)
        k = distances.shape[1]

        # Inverse-distance weights
        inv_dist   = 1.0 / np.maximum(distances, 1e-8)              # (batch, k)
        inv_dist_w = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        # Scores at each neighbour: (batch, k, n_models)
        neighbor_scores = self.matrix[indices]

        # Weighted least-squares trend — normalize distances to [0, 1]
        d_max  = distances.max(axis=1, keepdims=True)
        d_norm = distances / np.where(d_max > 0, d_max, 1.0)        # (batch, k)

        # Build X^T W X components
        W   = inv_dist_w                                             # (batch, k)
        a   = W.sum(axis=1)                                          # (batch,)
        b   = (W * d_norm).sum(axis=1)
        d_v = (W * d_norm ** 2).sum(axis=1)
        det = a * d_v - b ** 2                                       # (batch,)
        bad_det = np.abs(det) <= 1e-12
        det_safe = np.where(bad_det, 1.0, det)

        # X^T W y for all models: (batch, 2, n_models)
        Wy   = neighbor_scores * inv_dist_w[:, :, np.newaxis]        # (batch, k, n_models)
        Wdy  = Wy * d_norm[:, :, np.newaxis]
        XtWy_0 = Wy.sum(axis=1)                                      # (batch, n_models)
        XtWy_1 = Wdy.sum(axis=1)                                     # (batch, n_models)

        # Closed-form 2×2 inverse
        intercept = (d_v[:, np.newaxis] * XtWy_0 -
                     b[:, np.newaxis]   * XtWy_1) / det_safe[:, np.newaxis]
        slope     = (a[:, np.newaxis]   * XtWy_1 -
                     b[:, np.newaxis]   * XtWy_0) / det_safe[:, np.newaxis]

        # Weighted R²
        y_hat  = (intercept[:, np.newaxis, :] +
                  slope[:, np.newaxis, :] * d_norm[:, :, np.newaxis])  # (batch, k, n_models)
        y_wmean = XtWy_0
        ss_res = (inv_dist_w[:, :, np.newaxis] *
                  (neighbor_scores - y_hat) ** 2).sum(axis=1)
        ss_tot = (inv_dist_w[:, :, np.newaxis] *
                  (neighbor_scores - y_wmean[:, np.newaxis, :]) ** 2).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)
        r2 = np.where(bad_det[:, np.newaxis], 0.0, r2)              # (batch, n_models)

        # DEWS-I fallback scores
        if self._use_signed:
            fallback_raw  = self._convert(neighbor_scores)
            dewsi_scores  = -(fallback_raw * inv_dist_w[:, :, np.newaxis]).sum(axis=1)
        else:
            dewsi_scores  = XtWy_0

        # Convert trend intercept to routing score
        if self._use_signed:
            trend_scores = -self._convert(intercept)
        else:
            trend_scores = intercept

        # Blend: trust trend where R² ≥ threshold, fall back otherwise
        use_trend  = r2 >= r2_th
        avg_scores = np.where(use_trend, trend_scores, dewsi_scores)

        # Standard DEWS softmax
        local_min = avg_scores.min(axis=1, keepdims=True)
        local_max = avg_scores.max(axis=1, keepdims=True)
        local_range = local_max - local_min
        norm_scores = (avg_scores - local_min) / np.where(local_range > 0, local_range, 1.0)

        if th > 0:
            gate = norm_scores >= th
            any_pass = gate.any(axis=1, keepdims=True)
            gate = np.where(any_pass, gate, norm_scores == 1.0)
            norm_scores = norm_scores * gate

        max_scores = norm_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp((norm_scores - max_scores) / t)
        if th > 0:
            exp_scores = exp_scores * gate
        total = exp_scores.sum(axis=1, keepdims=True)
        weights = np.where(total > 0,
                           exp_scores / np.where(total > 0, total, 1.0),
                           np.full_like(exp_scores, 1.0 / len(self.models)))
        return weights