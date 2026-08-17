# deskit

This is not a usage guide, but an overview of deskit. For a usage guide, see the [documentation](https://TikaaVo.github.io/deskit/usage)

deskit is a flexible, lightweight, and easy-to-use ensembling library that implements
Dynamic Ensemble Selection (DES) algorithms for ensembling multiple ML models
on a given dataset. 

The library works entirely with data, taking as input a validation dataset 
along with precomputed predictions and outputting a dictionary of weights 
per model. This means that it can be used with any library or model without 
requiring any wrappers, including custom models, popular ML libraries, and APIs.

deskit includes several DES algorithms, and it works with both classification
and regression.

See the full documentation [here](https://TikaaVo.github.io/deskit/).

# Dynamic Ensemble Selection

Ensemble learning in machine learning refers to when multiple models trained on a 
single dataset combine their predictions to create a single, more accurate prediction,
usually through weighted voting or picking the best model.

DES refers to techniques where the models or their voting weights are selected dynamically
for every test case. This selection bases on the idea of competence regions, which is the 
concept that there are regions of feature space where certain models perform particularly well,
so every base model can be an expert in a different region.
Only the most competent, or an ensemble of the most competent models is selected for the prediction.

Through empirical studies, DES has been shown to perform best on small-sized, imbalanced, or 
heterogeneous datasets, as well as non-stationary data (concept drift), models that haven't perfected a dataset, 
and when used on an ensemble of models with differing architectures and perspectives.

However, DES is not an automatic improvement. It tends to perform worse when datasets are homogeneous or have low diversity, 
when the validation set isn't a good representation of the test set, when using very high dimensional data or few training samples,
or when a single model dominates a dataset.

---

## Installation

```bash
pip install deskit

# The library runs with Nearest Neighbors from sklearn for exact KNN
pip install scikit-learn

# Alternatively, ANN can be used for faster runtimes at the cost of
# slightly lower accuracy. The following three are supported;
# Install the one you want to use.
pip install faiss-cpu   # FAISS (good default for most datasets)
pip install annoy       # Annoy (memory-efficient, simple)
pip install hnswlib     # HNSW (best for high-dimensional data)
```

---

## Dependencies

Python (>= 3.8)

NumPy (>= 1.21)

---

## Quick start

For a more detailed understanding of how to use the library, consult the [usage guide](https://TikaaVo.github.io/deskit/usage).

```python
from deskit.des.knorau import KNORAU

# 1. Train your models
models = {"rf": rf, "xgb": xgb, "mlp": mlp}

# 2. Get predictions on a held-out validation set
#    Regression: scalar arrays
#    Classification: probability arrays OR hard predictions
val_preds = {name: m.predict_proba(X_val) for name, m in models.items()}

# 3. Fit the router
router = KNORAU(task="classification", metric="accuracy", mode="max", k=20)
router.fit(X_val, y_val, val_preds)

# 4. Get test predictions from your models
test_preds = {name: m.predict_proba(X_test) for name, m in models.items()}

# 5. Route and blend test samples instantly
#    Returns final ensembled predictions for the entire batch
#    Alternatively, predict_weights can be used to get an array of weights dicts
#    Weights dict example:{"rf": 0.7, "xgb": 0.2, "mlp": 0.1}
predictions = router.predict(X_test, test_preds, temperature=0.1)
```

For classification with probability arrays, blend the output the same way to
get a final probability distribution, then take the argmax.

---

### Batch vs. Streaming Usage

`deskit` is fully polymorphic and handles both offline batch processing and real-time streaming data natively:

* **Batch Mode:** Pass a 2D feature matrix and a dictionary of prediction arrays (as shown above) to process everything at vectorized speeds.
* **Streaming Mode:** Pass a single 1D feature array and a dictionary of individual model outputs. `router.predict()` will automatically detect it and return a single final prediction instead of an array.

If you want to handle the ensembling math yourself or inspect the raw dynamic routing weights, use `predict_weights`:

```python
# Returns a dictionary of weights (or array of weights for a batch)
# e.g., {"rf": 0.7, "xgb": 0.2, "mlp": 0.1}
weights = router.predict_weights(X_test, temperature=0.1)
```

---

### Hyperparameter tuning

`deskit` supports hyperparameter tuning of KNN-based methods using Leave One Out (LOO) tuning.
This technique excludes the closest data point with a negligent distance from the one being used in predict(), allowing for `deskit` methods to be tuned from the same DSEL set it was fit on and removing the need for cross-validation as it sees the entire set throughout the tuning process.

```python
results = router.predict(X_val, val_preds, loo=True)
```

---

## Why deskit?

Most DES libraries are tied to scikit-learn. deskit only ever sees a numpy
feature matrix and a dict of prediction arrays, so the models themselves are
never touched after training. This allows for more flexibility and a lighter library.

Furthermore, deskit works with both classification and regression, while the majority of DES
libraries and literature is focused only on classification tasks.

```python
# PyTorch example 
with torch.no_grad():
    val_preds  = {name: m(X_val_t).cpu().numpy()  for name, m in models.items()}
    test_preds = {name: m(X_test_t).cpu().numpy() for name, m in models.items()}

router = KNORAU(task="classification", metric="accuracy", mode="max", k=20)
router.fit(X_val, y_val, val_preds)

# Get final blended predictions
X_test_np = X_test_t.cpu().numpy()
predictions = router.predict(X_test_np, test_preds, temperature=0.1)
```

---

## Algorithms

Full explanation of the algorithms, syntax, and parameters is available in the [documentation](https://TikaaVo.github.io/deskit/).
If you're struggling to decide on which algorithm to use, see the [algorithm selection guide](https://TikaaVo.github.io/deskit/selection).

| Method     | Best for       | Notes                                                                                                  |
|------------|----------------|--------------------------------------------------------------------------------------------------------|
| `DEWS-U`   | Regression     | Softmax over neighborhood-averaged scores. Temperature controls sharpness.                             |
| `DEWS-I`   | Regression     | Like DEWS-U but scores are inverse-distance weighted.                                                  |
| `DEWS-T`   | Both           | Like DEWS-I but fits a weighted trend line over neighbor scores.                                       |
| `DEWS-V`   | Regression     | Like DEWS-U but scores are variance-penalized.                                                                             |
| `DEWS-IV`  | Regression     | Like DEWS-V but scores are also inverse-distance weighted.                                             |
| `LWSE-U`   | Both           | Per-sample NNLS weight estimation over the local neighbourhood.                                        |
| `LWSE-I`   | Both           | Like LWSE-U but rows are inverse-distance weighted.                                                    |
| `KNORA-U`  | Classification | Each model earns one vote per neighbor it correctly classifies.                  |
| `KNORA-E`  | Classification | Only models correct on all neighbors survive; falls back to smaller neighborhoods. |
| `KNORA-IU` | Classification | Like KNORA-U but votes are inverse-distance weighted.                                                  |
| `OLA`      | Both           | Hard selection: only the single best model in the neighborhood contributes.                            |

---

## ANN backends

deskit supports three Approximate Nearest Neighbour backends plus exact search:

| Preset | Backend | Install | Notes |
|---|---|---|---|
| `exact` | sklearn KNN |  `scikit-learn` | Exact, no extra deps |
| `balanced` | FAISS IVF | `faiss-cpu` | ~98% recall, good default |
| `fast` | FAISS IVF | `faiss-cpu` | ~95% recall, faster queries |
| `turbo` | FAISS flat | `faiss-cpu` | Exact via FAISS, GPU-friendly |
| `high_dim_balanced` | HNSW | `hnswlib` | Best for >100 features, balanced |
| `high_dim_fast` | HNSW | `hnswlib` | Best for >100 features, faster |

Annoy is also available as a custom backend — memory-efficient and simple,
good for datasets that need to be persisted to disk.

```python
# Exact search (no extra deps)
router = KNORAU(..., preset="exact")

# High-dimensional data
router = KNORAU(..., preset="high_dim_balanced")

# Custom FAISS config
router = KNORAU(..., preset="custom", finder="faiss",
                index_type="ivf", n_probes=50)

# Annoy
router = KNORAU(..., preset="custom", finder="annoy",
                n_trees=100, search_k=-1)
```

---

## Custom metrics

Any callable `(y_true, y_pred) -> float` works:

```python
def pinball(y_true, y_pred, alpha=0.9):
    e = y_true - y_pred
    return alpha * e if e >= 0 else (alpha - 1) * e

router = DEWSU(task="regression", metric=pinball, mode="min", k=20)
```

Built-in metric strings: `accuracy`, `mae`, `mse`, `rmse`, `log_loss`, `prob_correct`.

Alternatively, pass `metric=None` and populate the dictionary passed to fit() with your own per-sample computed scores instead of predictions.

---

## Data types

deskit can be used with non-tabular data types like images, time series, and more. However, when used, the
passed features either need to be run through a feature extractor beforehand, such as a CNN backbone for images.

## Benchmark results

20-seed benchmark (seeds 0–19) on standard sklearn and OpenML datasets. "Best Single" is the best
individual model selected on the validation set. "Simple Average" is uniform
equal-weight blending, included as a baseline.

It is important to consider that these experiments were run with the default hyperparameters, meaning that
they could vary greatly with different values, and results could improve with tuning.
For a more detailed benchmark breakdown, see the [benchmark in the documentation](https://TikaaVo.github.io/deskit/benchmark).
To see the full results, see `results.txt` in the `tests` folder.

This pool was selected for having variability in architectures while avoiding a single dominant model.

deskit algorithms tested: OLA, DEWS-U, DEWS-I, DEWS-T, DEWS-V, DEWS-IV, LWSE-U, LWSE-I, KNORA-U, KNORA-E, KNORA-IU.

### Regression (MAE, lower is better)

Pool: KNN, Decision Tree, SVR, Ridge, Bayesian Ridge.

% shown as delta vs Best Single. 20-seed mean.

| Dataset                      | Best Single | Simple Avg | deskit best               |
|------------------------------|-------------|------------|---------------------------|
| California Housing (sklearn) | 0.3956      | +7.99%     | **−2.54%** (DEWS-I)       |
| Bike Sharing (OpenML)        | 51.678      | +47.77%    | **−6.86%** (DEWS-I)       |
| Abalone (OpenML)             | **1.4981**  | +1.14%     | +1.47% (KNORA-U/KNORA-IU) |
| Diabetes (sklearn)           | **44.504**  | +3.18%     | +0.86% (DEWS-IV)          |
| Concrete Strength (OpenML)   | 5.2686      | +23.66%    | **−5.41%** (LWSE-I)       |

deskit beats best single and simple averaging on 3/5 regression datasets. This shows how DES can provide a
strong boost if used on the right dataset, but it might be counterproductive if used blindly.

KNORA variants are designed for classification, which explains the poor performance
on regression datasets; However, some exceptions can occur in certain datasets when the target is discrete
and classification-like (like in Abalone).

DEWS-I and LWSE-I show the largest improvements on their respective datasets.

### Classification (Accuracy, higher is better)

Pool: KNN, Decision Tree, Gaussian NB, SVM-RBF, Logistic Regression.

% shown as delta vs Best Single. 20-seed mean.

| Dataset                | Best Single | Simple Avg | deskit best                    |
|------------------------|-------------|------------|--------------------------------|
| HAR (OpenML)           | 98.24%      | −0.33%     | **+0.16%** (DEWS-T)            |
| Yeast (OpenML)         | 58.87%      | +0.77%     | **+1.66%** (KNORA-IU)          |
| Image Segment (OpenML) | 93.70%      | +1.40%     | **+2.25%** (DEWS-T / DEWS-IV)  |
| Waveform (OpenML)      | **85.91%**  | −0.98%     | −0.39% (DEWS-T)                |
| Vowel (OpenML)         | 89.95%      | −2.05%     | **+2.95%** (LWSE-I)            |

deskit beats or matches best single and simple averaging on 4/5 classification datasets.

### Speed (mean ms fit + predict, 20 seeds, all tested algorithms combined)

Consider that usually it is recommended to only use one algorithm at a time, this benchmark ran eleven of them at the
same time, so with a single one runtime is expected to be about 11x faster. For this benchmark, `preset='balanced'` was used,
so the backend was an ANN algorithm with FAISS IVF.

| Dataset            | deskit (11 algorithms) |
|--------------------|------------------------|
| California Housing | 351.0 ms               |
| Bike Sharing       | 283.5 ms               |
| Abalone            | 72.9 ms                |
| Diabetes           | 14.0 ms                |
| Concrete Strength  | 22.5 ms                |
| HAR                | 693.1 ms               |
| Yeast              | 44.7 ms                |
| Image Segment      | 69.9 ms                |
| Waveform           | 124.5 ms               |
| Vowel              | 39.0 ms                |

deskit caches all model predictions on the validation set at fit time and reads
from that matrix at inference.

---

## Contributing

Issues and PRs welcome.