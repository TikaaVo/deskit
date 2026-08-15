# KNORA-E

This algorithm comes from the KNORA family and is designed to be a more aggressive variation for datasets
with models that dominate their competence regions. It checks the K nearest neighbors and keeps reducing K 
until one model has perfect predictions on all neighbors.

---

## When to use

- KNORA-E is an aggressive algorithm designed for classification tasks to be used when certain models dominate their region, 
to be used a discrete metric, usually `accuracy`
- It performs best when the pool has clear regional specialists
- It performs worst on noisy or heavily overlapping datasets, and particularly on regression tasks

---

## How it works

When `fit` is called, KNORA-E fits a KNN algorithm on the validation data and builds a criterion score matrix. 

When `predict` is called, it finds the K nearest neighbors from the test point and for normalizes the model scores
for every neighbor with min-max normalization. If all models scored identically, they all get `1.0`. If there is are models that 
exceeds the threshold for every neighbor case (for classification, this would mean that the model gets all of the neighbors correct),
then those models survive and get equal weights, otherwise, k is decreased by 1 and the process is repeated. If the algorithm reaches
`k=1` and still no model passes, it assigns uniform weights to all the models.

These weights can then be provided as output or combined with predictions to make a final ensembled answer.

---

## Parameters

| Parameter     | Type | Default      | Description                                                                                                                                                     |
|---------------|---|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `task`        | str | —            | `"classification"` or `"regression"`                                                                                                                            |
| `metric`      | str or callable | —            | Scoring function per sample. Built-ins: `accuracy`, `mae`, `mse`, `rmse`, `log_loss`, `prob_correct`. Custom callables `(y_true, y_pred) -> float` are accepted |
| `mode`        | str | —            | `"max"` if higher is better, `"min"` if lower                                                                                                                   |
| `k`           | int | 10           | Number of neighbours                                                                                                                                            |
| `threshold`   | float | 0.5          | Competence cutoff                                                                                                                                               |
| `temperature` | float | 1.0          | Accepted for internal API consistency but not used      
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)                                                                                                           |
| `preset`      | str | `"balanced"` | ANN backend preset. Options:    `"exact"`,`"balanced"`, `"fast"`, `"turbo"`, `"high_dim_balanced"`, `"high_dim_fast"`                                                                                                                                              |
| `finder`      | str | —, optional  | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |
| `loo`      | bool | false                          | Enables Leave One Out (LOO) tuning; ignores the closest neighbor with negligible distance when selecting the K neighbors.      
| `norm`      | str or callable | bestrel                          | Normalization function per sample. Built-ins: `minmax`, `zscore`, `bestrel`, `softmax`, `rank_norm`                                                                           |

---

## Example
```python
from deskit.des.knorae import KNORAE

router = KNORAE(task="classification", metric="accuracy", mode="max", k=20)
router.fit(X_val, y_val, val_preds)
answers = router.predict(X_test, test_preds)
```

---

## Notes

The threshold parameter is used only for regression; For classification with a discrete metrics, any threshold `(0,1]` behaves identically.
However, this algorithm is not designed to be used for regression, so using it for regression is not recommended.