# OLA

This algorithm is a DCS algorithm (Dynamic Classifier Selection), meaning that it selects a single model to have the
entire weight instead of ensembling multiple of them.

---

## When to use

- OLA is often used as a diagnostic tool or when one only wants to run the inference of a single model per test case to
cut down computation
- It performs best when a single model is dramatically better than the others in a competence regions
- It performs worst when throwing away most models and only keeping one loses accuracy

---

## How it works

When `fit` is called, OLA fits a KNN algorithm on the validation data and builds a criterion score matrix, then normalizes
them globally with min-max normalization. 

When `predict` is called, it finds the K nearest neighbors from the test point and uses the score matrix to average
every models' scores over the K neighbors. Afterwards, it uses `argmax` and gives full weight to the model that had the highest
accuracy.

These weights can then be provided as output or combined with predictions to make a final ensembled answer.

---

## Parameters

| Parameter     | Type | Default      | Description                                                                                                                                                     |
|---------------|---|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `task`        | str | —            | `"classification"` or `"regression"`                                                                                                                            |
| `metric`      | str or callable | —            | Scoring function per sample. Built-ins: `accuracy`, `mae`, `mse`, `rmse`, `log_loss`, `prob_correct`. Custom callables `(y_true, y_pred) -> float` are accepted. None is accepted if the fit dictionary is pre-computed per-sample scores instead of predictions. |
| `mode`        | str | —            | `"max"` if higher is better, `"min"` if lower                                                                                                                   |
| `k`           | int | 10           | Number of neighbours                                                                                                                                            |
| `threshold`   | float | 0.5          | Accepted for internal API consistency but not used                                                                                                                                               |
| `temperature` | float | 1.0          | Accepted for internal API consistency but not used                 
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)                                                                                                          |
| `preset`      | str | `"balanced"` | ANN backend preset. Options:    `"exact"`,`"balanced"`, `"fast"`, `"turbo"`, `"high_dim_balanced"`, `"high_dim_fast"`                                                                                                                                              |
| `finder`      | str | —, optional  | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |
| `loo`      | bool | false                          | Enables Leave One Out (LOO) tuning; ignores the closest neighbor with negligible distance when selecting the K neighbors.                                                                             |

---

## Example
```python
from deskit.des.ola import OLA

router = OLA(task="regression", metric="mae", mode="min", k=20)
router.fit(X_val, y_val, val_preds)
answers = router.predict(X_test, test_preds)
```

---