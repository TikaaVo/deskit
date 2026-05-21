"""
LWSE-U: Locally Weighted Stacking Ensemble (Uniform).
"""
from deskit.base.predictbase import PredictBase
from deskit._config import make_finder
from scipy.optimize import nnls
import numpy as np


class LWSEU(PredictBase):
    """
    LWSE-U: Locally Weighted Stacking Ensemble (Uniform).

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    k : int
        Neighbourhood size. Default: 10.
    preset : str
        Neighbour search preset. Default: 'balanced'. See list_presets().
    """

    def __init__(self, task, k=10, preset='balanced', **kwargs):
        self.task    = task
        self.k       = k
        self._finder = make_finder(preset, k, **kwargs)
        self.models  = None

        self._val_preds = None   # (n_val, n_models) or (n_val, n_models, n_classes)
        self._y_val     = None   # (n_val,)
        self._y_onehot  = None   # (n_val, n_classes) for classification
        self._is_proba  = None

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
            Shape (n_val,) for regression; (n_val, n_classes) for
            classification with probability output.
        """
        features = np.asarray(features, dtype=float)
        y        = np.asarray(y)

        self.models    = list(preds_dict.keys())
        first          = np.asarray(list(preds_dict.values())[0])
        self._is_proba = (first.ndim == 2)

        if self._is_proba:
            self._val_preds = np.stack(
                [np.asarray(preds_dict[m], dtype=float) for m in self.models],
                axis=1
            )  # (n_val, n_models, n_classes)
            n_val, _, n_classes = self._val_preds.shape
            self._y_onehot = np.zeros((n_val, n_classes), dtype=float)
            self._y_onehot[np.arange(n_val), y.astype(int)] = 1.0
        else:
            self._val_preds = np.stack(
                [np.asarray(preds_dict[m], dtype=float) for m in self.models],
                axis=1
            )  # (n_val, n_models)

        self._y_val = y
        self._finder.fit(features)

    def _weights_batch(self, x, temperature=None, **kwargs):
        """
        Core weight computation. x is a 2-D float64 numpy array (batch, n_features).
        Returns (batch, n_models) weight array.

        NNLS is solved independently per sample (unavoidable); the rest of
        the pipeline (dict formatting, ensembling) is handled vectorially
        by PredictMixin.
        """
        batch_size = x.shape[0]
        n_models   = len(self.models)
        uniform    = np.full(n_models, 1.0 / n_models)

        distances, indices = self._finder.kneighbors(x)              # (batch, k)
        weights_out        = np.empty((batch_size, n_models))

        for b in range(batch_size):
            idx = indices[b]                                          # (k,)

            if self._is_proba:
                P = self._val_preds[idx]                              # (k, n_models, n_classes)
                k_, _, n_classes = P.shape
                P_flat = P.transpose(0, 2, 1).reshape(k_ * n_classes, n_models)
                y_flat = self._y_onehot[idx].reshape(k_ * n_classes)
                try:
                    coeffs, _ = nnls(P_flat, y_flat, maxiter=10 * n_models)
                except RuntimeError:
                    coeffs = uniform.copy()
            else:
                P      = self._val_preds[idx]                         # (k, n_models)
                y_nbr  = self._y_val[idx]                             # (k,)
                lambda_ = 1e-6
                P_aug  = np.vstack([P, lambda_ * np.eye(n_models)])
                y_aug  = np.concatenate([y_nbr, np.zeros(n_models)])
                try:
                    coeffs, _ = nnls(P_aug, y_aug, maxiter=10 * n_models)
                except RuntimeError:
                    coeffs = uniform.copy()

            total = coeffs.sum()
            weights_out[b] = coeffs / total if total > 1e-10 else uniform

        return weights_out