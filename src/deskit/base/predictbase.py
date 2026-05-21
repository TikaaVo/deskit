"""
PredictBase: unified predict() and predict_weights() for all DES routers.

Any router that inherits this mixin and implements _weights_batch() gets
both public methods automatically.
"""
import numpy as np
from deskit.utils import to_numpy


class PredictBase:
    """
    Mixin that adds predict() and predict_weights() to any DES router.

    Subclasses must implement:

        _weights_batch(x, temperature=None, threshold=None, **kwargs)
            -> np.ndarray of shape (batch_size, n_models)

    where ``x`` is already a 2-D float64 numpy array (atleast_2d + to_numpy
    is handled here, not in _weights_batch).

    Subclasses must also expose:
        self.models : list[str]   ordered model names
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_weights(self, X_test, temperature=None, **kwargs):
        """
        Return per-sample model weights for every point in X_test.

        Parameters
        ----------
        X_test : array-like, shape (n_features,) or (n_samples, n_features)
            Test features.
        temperature : float, optional
            Softmax sharpness. Forwarded to _weights_batch; ignored by
            algorithms that do not use softmax (KNORA-*, OLA).
        **kwargs
            Additional per-call overrides forwarded to _weights_batch.
            Supported by most algorithms: ``threshold=<float>``.

        Returns
        -------
        dict or list[dict]
            Single point  → {model_name: weight}.
            Batch         → list of such dicts, one per sample.
        """
        x, batch_size = _prepare(X_test)
        weights = self._weights_batch(x, temperature=temperature, **kwargs)
        result = [dict(zip(self.models, w)) for w in weights]
        return result[0] if batch_size == 1 else result

    def predict(self, X_test, test_preds, temperature=None, **kwargs):
        """
        Ensemble model predictions for every point in X_test.

        The ensembling strategy depends on prediction shape:

        * **Probability arrays** (shape ``(n_samples, n_classes)`` per model):
          weighted average of probability vectors.  For a hard class label,
          take ``argmax`` on the returned array.
        * **Scalar predictions** (shape ``(n_samples,)`` per model):
          weighted average.  Works for regression and for classification when
          hard labels are passed (returns a float; round or cast as needed).

        Parameters
        ----------
        X_test : array-like, shape (n_features,) or (n_samples, n_features)
            Test features.
        test_preds : dict[str, array-like]
            Test predictions keyed by model name, formatted the same way as
            the ``val_preds`` passed to ``fit()``.
            Regression / hard labels : shape ``(n_samples,)`` per model.
            Probability classification : shape ``(n_samples, n_classes)`` per model.
        temperature : float, optional
            Forwarded to _weights_batch.
        **kwargs
            Forwarded to _weights_batch (e.g. ``threshold=<float>``).

        Returns
        -------
        np.ndarray
            Single point → scalar or 1-D array (probability vector).
            Batch → shape ``(n_samples,)`` or ``(n_samples, n_classes)``.
        """
        x, batch_size = _prepare(X_test)
        weights = self._weights_batch(x, temperature=temperature, **kwargs)  # (batch, n_models)

        preds_list = [np.asarray(test_preds[m], dtype=float) for m in self.models]
        first = preds_list[0]

        if self.task == 'classification':
            # Probability arrays: blend per-class columns.
            # preds_3d : (batch, n_models, n_classes)
            preds_3d = np.stack(preds_list, axis=1)
            result = np.einsum("bm,bmc->bc", weights, preds_3d)  # (batch, n_classes)
        else:
            # Scalar predictions: weighted average.
            preds_2d = np.stack(preds_list, axis=1)              # (batch, n_models)
            result = (weights * preds_2d).sum(axis=1)             # (batch,)

        return result[0] if batch_size == 1 else result


# ------------------------------------------------------------------
# Module-private helper
# ------------------------------------------------------------------

def _prepare(X_test):
    """Convert X_test to a 2-D float64 array and return (x, batch_size)."""
    x = np.atleast_2d(to_numpy(X_test)).astype(float, copy=False)
    return x, x.shape[0]