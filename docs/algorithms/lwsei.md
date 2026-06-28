# LWSE-I

LWSE-U is a dynamic, per-sample stacking algorithm, designed for both classification and regression tasks.
It uses NNLS to stack models locally for every test case and uses those weights to ensemble the models. 
The stacking is distance-weighted, so closer neighbors pull it more than farther ones. 

For classification, only confidence scores can be used.

---

## When to use

- LWSE-I was designed to be an algorithm that can perform very well in certain cases, improving effectiveness
by a large margin, but it can be inconsistent for general use
- LWSE-I performs best with heterogeneous model pools and datasets with strong local structure.
- LWSE-I performs worst when datasets have one uniformly dominant model, sparse class distributions
with small k or in high dimensional datasets
- LWSE-I is also heavier than most other DES algorithms, and while it usually isn't an issue, it scales with
k and the amount of models, so if both of those are large it can add computational costs

## How it works

When `fit` is called, LWSE-I fits a KNN algorithm on the validation data and builds a prediction matrix.

When `predict` is called, it finds the K nearest neighbors from the test point and WLS weights are computed
based on the inverse-distance. Afterwards, the local system is built and NNLS is solved on the system. Finally,
the coefficients are divided by the sum to normalize them into weights. If the solver returns all zeros, weights fall 
back to being uniform.

These weights can then be provided as output or combined with predictions to make a final ensembled answer.

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | str | — | `"classification"` or `"regression"` |
| `k` | int | 10 | Number of neighbours. Higher k gives more stable fits but reduces locality. |
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)   
| `preset` | str | `"balanced"` | ANN backend preset. Options:    `"exact"`,`"balanced"`, `"fast"`, `"turbo"`, `"high_dim_balanced"`, `"high_dim_fast"` |
| `finder`      | str | —, optional                           | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |
| `loo`      | bool | false                          | Enables Leave One Out (LOO) tuning; ignores the closest neighbor with negligible distance when selecting the K neighbors.                                                                             |

LWSE-I has no `metric`, `mode`, `threshold`, or `temperature` parameters. The objective is
always local squared error, non-negativity is enforced by the solver, and sparsity emerges
naturally from the NNLS solution.

---

## Example

```python
# Regression
from deskit.des.lwsei import LWSEI

router = LWSEI(task="regression", k=20)
router.fit(X_val, y_val, val_preds)
answers = router.predict(X_test, test_preds)
```

```python
# Classification
from deskit.des.lwsei import LWSEI

router = LWSEI(task="classification", k=10)
router.fit(X_val, y_val, val_preds)
answers = router.predict(X_test, test_preds)
```

---

## Notes

LWSE-I does not use a metric or a gate threshold. Whether a model contributes at a given test
point is decided entirely by whether it reduces local squared error.