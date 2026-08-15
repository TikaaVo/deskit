# DEWS-V

This algorithm is designed for datasets and models where consistency among neighbors is important. 
It is a variation of DEWS-U that takes variability into consideration.
It uses soft blending between the top experts in a certain competence region to compute a set of weights for the models.

---

## When to use

- DEWS-V is not generally recommended, as empirically it has shown to perform worse than other algorithms
like DEWS-I or DEWS-U. 
- It performs best when a dataset has regions where models should be consistent throughout. However, for this purpose,
DEWS-IV is recommended over DEWS-V. 
- It performs worst for small values of k, since few data points can make variability measures noisy. It also struggles when
datasets don't have clear region boundaries, so a strong model may have variability.

---

## How it works

When `fit` is called, DEWS-V fits a KNN algorithm on the validation data and builds a criterion score matrix.

When `predict` is called, it finds the K nearest neighbors from the test point and uses the score matrix to combine
every models' scores over the K neighbors and subtracts a penalty term proportional to the variability from the scores.
Afterwards, it normalizes the average scores using min-max normalization and removes the models under a threshold. 
Finally, it takes the remaining models and creates weights with their scores using softmax with temperature.

These weights can then be provided as output or combined with predictions to make a final ensembled answer.

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
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)                                                                                                                      |
| `preset`      | str | `"balanced"`                          | ANN backend preset. Options:    `"exact"`,`"balanced"`, `"fast"`, `"turbo"`, `"high_dim_balanced"`, `"high_dim_fast"`                                                                                                                                             |
| `finder`      | str | —, optional                           | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |
| `loo`      | bool | false                          | Enables Leave One Out (LOO) tuning; ignores the closest neighbor with negligible distance when selecting the K neighbors.       
| `norm`      | str or callable | bestrel                          | Normalization function per sample. Built-ins: `minmax`, `zscore`, `bestrel`, `softmax`, `rank_norm`                                                                          |

---

## Example
```python
# Regression
from deskit.des.dewsv import DEWSV

router = DEWSV(task="regression", metric="mae", mode="min", k=20)
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

```python
# Classification
from deskit.des.dewsv import DEWSV

router = DEWSV(task="classification", metric="log_loss", mode="min", k=20)
router.fit(X_val, y_val, val_preds)
answers = router.predict(X_test, test_preds)
```

---

## Notes

A lower temperature is recommended for regression because regression metrics tend to produce scores on a 
continuous scale where differences can be large, so a low temperature sharpens the softmax to reflect that. 
In contrast, classification metrics tend to produce scores that are closer together, so a higher temperature
keeps the blend soft.