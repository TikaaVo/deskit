# KNORA-U

This algorithm comes from the KNORA family and is designed to be safe variation that works for
most classification tasks. It checks the K nearest neighbors and has the models perform weighted vote based on how
many correct predictions were made.

---

## When to use

- KNORA-U is an algorithm defined for general classification tasks, to be used a discrete metric, usually `accuracy`
- It performs best when the dataset has genuine class overlap
- It performs worst on datasets where one model heavily dominates a competence region

---

## How it works

When `fit` is called, KNORA-U fits a KNN algorithm on the validation data and builds a criterion score matrix. 

When `predict` is called, it finds the K nearest neighbors from the test point and for normalizes the model scores
for every neighbor with min-max normalization. If all models scored identically, they all get `1.0`. For each model, the amount of neighbors where its score
exceeded the threshold (in classification, this would mean that the predictions were correct) is that model's vote total.
The vote counts are then normalized by dividing them by the total amount of votes and turned into weights. If no model earned any votes at all, 
uniform weights are assigned.

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
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)                                                                                                                         |
| `preset`      | str | `"balanced"` | ANN backend preset. Options:    `"exact"`,`"balanced"`, `"fast"`, `"turbo"`, `"high_dim_balanced"`, `"high_dim_fast"`                                                                                                                                             |
| `finder`      | str | —, optional  | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |
| `loo`      | bool | false                          | Enables Leave One Out (LOO) tuning; ignores the closest neighbor with negligible distance when selecting the K neighbors.                                                                             |

---

## Example
```python
from deskit.des.knorau import KNORAU

router = KNORAU(task="classification", metric="accuracy", mode="max", k=20)
router.fit(X_val, y_val, val_preds)
answers = router.predict(X_test, test_preds)
```

---

## Notes

The threshold parameter is used only for regression; For classification with a discrete metrics, any threshold `(0,1]` behaves identically.
However, this algorithm is not designed to be used for regression, so using it for regression is not recommended.