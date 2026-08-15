# DEWS-I

This algorithm is designed to be consistent and flexible for both classification and
regression tasks. It is a variation of DEWS-U that takes neighbor distance into consideration.
It uses soft blending between the top experts in a certain competence region to compute a set of weights for the models.

---

## When to use

- DEWS-I is currently the general recommendation for regression tasks. DEWS-I works best with soft metrics, 
so it also works for classification with confidence scores, but not as well with hard predictions
- It performs best when the competence regions and pool are smooth and heterogeneous
- It performs worst for homogeneous datasets and for classification with hard predictions

---

## How it works

When `fit` is called, DEWS-I fits a KNN algorithm on the validation data and builds a criterion score matrix.

When `predict` is called, it finds the K nearest neighbors from the test point and uses the score matrix to combine
every models' scores over the K neighbors with inverse-distance weights. Afterwards, it normalizes the average scores 
using min-max normalization and removes the models under a threshold. Finally, it takes the remaining models 
and creates weights with their scores using softmax with temperature.

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
| `distance_metric` | str | "euclidean" |Distance metric  used for KNN/ANN. See [distance metrics](../backends/distance_metrics.md)                                                                                                 |
| `preset`      | str | `"balanced"`                          | ANN backend preset. Options:    `"exact"`,`"balanced"`, `"fast"`, `"turbo"`, `"high_dim_balanced"`, `"high_dim_fast"`                                                                                                                                               |
| `finder`      | str | —, optional                           | Only if the preset is `"custom"`; Options: `"knn"`, `"faiss"`, `"annoy"`, `"hnsw"`                                                                              |
| `loo`      | bool | false                          | Enables Leave One Out (LOO) tuning; ignores the closest neighbor with negligible distance when selecting the K neighbors.                                                                             |
| `norm`      | str or callable | bestrel                          | Normalization function per sample. Built-ins: `minmax`, `zscore`, `bestrel`, `softmax`, `rank_norm`    

---

## Example
```python
# Regression
from deskit.des.dewsi import DEWSI

router = DEWSI(task="regression", metric="mae", mode="min", k=20)
router.fit(X_val, y_val, val_preds)
weights = router.predict(x)
```

```python
# Classification
from deskit.des.dewsi import DEWSI

router = DEWSI(task="classification", metric="log_loss", mode="min", k=20)
router.fit(X_val, y_val, val_preds)
answers = router.predict(X_test, test_preds)
```

---

## Notes

A lower temperature is recommended for regression because regression metrics tend to produce scores on a 
continuous scale where differences can be large, so a low temperature sharpens the softmax to reflect that. 
In contrast, classification metrics tend to produce scores that are closer together, so a higher temperature
keeps the blend soft.