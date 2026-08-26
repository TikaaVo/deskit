# Usage Guide

This guide is for those who want to get started with the library. It will go through all the important details and methods.

## Step 1: Install and Import

In order to install deskit, run the following command:

```bash
pip install deskit
```

These are the dependencies for deskit to work:

- Python (>= 3.8)
- NumPy (>= 1.21)

Because deskit is based around K-Nearest Neighbors (KNN), at least one of the following must be installed, in order to use KNN or ANN (See the backends section of the [main page](index.md)):

```bash
# The library runs with Nearest Neighbors from sklearn for exact KNN
pip install scikit-learn

# Alternatively, ANN can be used for faster runtimes at the cost of
# slightly lower accuracy. The following three are supported;
# Install the one you want to use.
pip install faiss-cpu   # FAISS (good default for most datasets)
pip install annoy       # Annoy (memory-efficient, simple)
pip install hnswlib     # HNSW (best for high-dimensional data)
```

Once the library is installed, it can be used. Every DES method offered by deskit is contained within a separate class, therefore the class must be imported. The class names are made up of the letters in the algorithm name in all caps. The following example shows three alternative ways to import DEWS-U:

```python
from deskit import KNORAU # Cleanest
from deskit.des import KNORAU
from deskit.des.knorau import KNORAU
```

If you struggle with selecting the algorithm, check out the [Algorithm Selection Guide](selection.md).

## Step 2: Initialize the router

Once the library is imported, it can be initalized:

```python
router = KNORAU(task="classification", metric="accuracy", mode="max", k=20)
```

The initialization function accepts the following parameters:

---

## Parameters

| Parameter     | Type | Default                               | Description                                                                                                                                                     |
|---------------|---|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `task`        | str | —                                     | `"classification"` or `"regression"`                                                                                                                            |
| `metric`      | str or callable | —                                     | Scoring function per sample. Built-ins: `accuracy`, `mae`, `mse`, `rmse`, `log_loss`, `prob_correct`. Custom callables `(y_true, y_pred) -> float` are accepted |
| `mode`        | str | —                                     | `"max"` if higher is better, `"min"` if lower                                                                                                                   |
| `k`           | int | 10                                    | Number of neighbours                                                                                                                                            |
| `threshold`   | float | 0.5                                   | Competence cutoff                                                                                                                                               |
| `temperature` | float | 0.5/1.0 for regression/classification | Defines how smooth the model blend is               
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)                                                                                                    |
| `preset`      | str | `"balanced"`                          | ANN backend preset. Options:    `"exact"`,`"balanced"`, `"fast"`, `"turbo"`, `"high_dim_balanced"`, `"high_dim_fast"`                                                                                                                                    |
| `finder`      | str | —, optional                           | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)        
| `loo`      | bool | false                          | Enables Leave One Out (LOO) tuning; ignores the closest neighbor with negligible distance when selecting the K neighbors.       
| `normalization`      | str or callable | bestrel                          | Normalization function per sample. Built-ins: `minmax`, `zscore`, `bestrel`, `softmax`, `rank_norm`                                                                      |
---

It is important to note that not all of these are used for every algorithm. `temperature` is only used by DEWS methods, while LWSE methods also don't use `metric`, `mode`, or `threshold`. Consider their respective documentation pages from the [menu](index.md) to find out more.

## Step 3: Fit the router

Once the router is defined, it can be fit as follows:

```python
router.fit(X_val, y_val, val_preds)
```

The `fit` function accepts three parameters in every case, being:

- X_val: Validation features
- y_val: Validation labels
- val_preds: A dictionary of model predictions, structured in {model_name: [prediction array]} pairs. For each prediction, raw prediction values are accepted, but for classification tasks, deskit also accepts confidence score arrays (for example, [0.1, 0.7, 0.2]) with the use of log_loss. For classification tasks utilizing probability arrays, the arrays must be 2D with the shape (n_samples, n_classes). For hard predictions or regression values, they should be 1D with the shape (n_samples,).

Ensure that all the arrays used are ordered the same way, i.e. when considering the point `X_val[i]`, the correct label was `y_val[i]`, and model `model_name` predicted `val_preds[model_name][i]`.

## Step 4: Get the predictions

deskit allows two variants of prediction, either automatic ensembling or returning weights.

### predict

The `predict` function is used as follows:

```python
predictions = router.predict(X_test, test_preds)
```

It accepts the following parameters:

- X_test: Testing data point(s)
- test_preds: Model predictions. These are structured similarly to val_preds in the `fit` function
- temperature: in case you want to override the initialization temperature for a specific case

The function accepts both a single data point at a time or an array/NumPy array of data points for batch processing.

This function calculates the weights per model and then ensembles your model predictions based on these weights and returns the final results. For classification tasks, it ensembles the models using weighted voting, while for regression tasks, it ensembles them using weighted averaging.

The return type is as follows:

- Single point: Scalar
- Batch: 1D array of shape (n_samples,)

### predict_weights

The `predict_weights` function is used as follows:

```python
weights = router.predict_weights(X_test)
```

It accepts the following parameters:

- X_test: Testing data point(s)
- temperature: in case you want to override the initialization temperature for a specific case

The function accepts both a single data point at a time or an array/NumPy array of data points for batch processing.

This function calculates the weights per model and returns the weights in dictionary form in {model_name:weight} pairs (for example, {"rf": 0.7, "xgb": 0.2, "mlp": 0.1}).

The return type is as follows:

- Single point: Single dictionary of {model_name: weight} pairs
- Batch: List of dictionaries