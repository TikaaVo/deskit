# Pool Architecture Breakdown

All results are from a 3-seed × 10-fold benchmark (seeds 0–2) over the OpenML-CTR23
suite, filtered to datasets ≥ 250 samples and large enough that the smallest router
`k` (k=2) fits inside the 25% comparison split. Each (seed, fold) produces an
independent train/comparison/test split. Every cell below is aggregated over all
2100 completed (dataset, seed, fold) evaluations.

The point of this breakdown is a specific prior: **DES is expected to have a harder
time adding value over a well-tuned, diverse heterogeneous pool than over a
homogeneous one.** These five pools let that prior be checked directly.

---

## Methodology

**Split (per dataset × seed × fold):** a 10-fold split first peels off the test fold.
From the remaining data, 25% (min 10 rows) is held out as the **comparison set**, and
the rest is the training set. The comparison set is used only for DES/stacking tuning
and final ensemble fitting — never for pruning or base-model training.

**Pool pruning:** each pool trains `POOL_SIZE = 25` candidates on a sub-train slice
(80/20 split of train), scores them on the held-out 20% validation slice, keeps every
model whose validation error is within `1.2×` of the best (minimum 2 kept), then
retrains the survivors on the full training set. Pruning is done **once per error
metric**, so the MAE-pruned and RMSE-pruned pools are genuinely different pools, not
one pool viewed two ways.

**Tuning:** every method — DES, stacking, and the heterogeneous base learners — is
tuned with the same Random Search budget (50 evaluations; 20 for heterogeneous base
learners, which use halving). DES/OLA are tuned by leave-one-out scoring on the
comparison set; stacking baselines use 3-fold CV.

**Metrics:** MAE (reported here, lower is better), tuned and pruned
independently.

---

## Pool architectures

Each pool has a different source of diversity, which is what should decide whether
DES has room to add value over the best member.

| Pool | Members | Diversity source |
|---|---|---|
| Bagging-KNN | 25 × KNN (k=5) | Bootstrap resampling of rows |
| Bagging-DT | 25 × DecisionTree | Bootstrap resampling of rows |
| Subspace-KNN | 25 × KNN (k=5) | Random feature subspace (half the features per member) |
| Subspace-DT | 25 × DecisionTree | Random feature subspace |
| Heterogeneous | Ridge, ElasticNet, KNN, DT, RF, XGBoost, MLP | Different inductive biases, all tuned |

---

## Methods compared

- **Oracle** — per-point best model, an upper bound, not deployable.
- **SingleBest** — best pool member chosen on the comparison set. The no-ensembling baseline.
- **SimpleAverage** — uniform blend of all surviving members.
- **ValWeightedAverage** — inverse validation-error weighting.
- **RidgeStacking / StackingRF / StackingGB** — meta-learners over pool predictions.
- **OLA** — overall local accuracy, selects the single best local model.
- **DEWS-U / DEWS-I / DEWS-V / DEWS-IV / DEWS-T** — dynamic ensemble weighting schemes.

`% vs SingleBest` = improvement over `SingleBest`. `% HC` = oracle headroom closed.
`Rank` is 1 = best. `Oracle` is excluded from ranking commentary but shown for headroom.

---

## Results — Bagging-DT

| Method | Mean | Std | % vs SingleBest | % HC | Rank |
|---|---:|---:|---:|---:|---:|
| Oracle | 1763.41 | 5820.69 | +84.68% | 100.00% | 1 |
| DEWS-I | 7798.74 | 23432.01 | +32.23% | 38.07% | 2 |
| DEWS-V | 7799.63 | 23404.75 | +32.23% | 38.06% | 3 |
| ValWeightedAverage | 7803.17 | 23337.30 | +32.20% | 38.02% | 4 |
| SimpleAverage | 7806.86 | 23347.90 | +32.16% | 37.98% | 5 |
| DEWS-U | 7809.42 | 23458.64 | +32.14% | 37.96% | 6 |
| DEWS-T | 7818.09 | 23494.37 | +32.07% | 37.87% | 7 |
| DEWS-IV | 7818.42 | 23494.86 | +32.06% | 37.86% | 8 |
| StackingRF | 7915.95 | 23664.24 | +31.22% | 36.86% | 9 |
| RidgeStacking | 7950.29 | 23908.71 | +30.92% | 36.51% | 10 |
| StackingGB | 8158.66 | 24373.26 | +29.11% | 34.37% | 11 |
| OLA | 11331.51 | 33908.10 | +1.54% | 1.81% | 12 |
| SingleBest | 11508.26 | 34260.04 | 0.00% | 0.00% | 13 |

## Results — Bagging-KNN

| Method | Mean | Std | % vs SingleBest | % HC | Rank |
|---|---:|---:|---:|---:|---:|
| Oracle | 4287.75 | 13908.81 | +58.39% | 100.00% | 1 |
| DEWS-I | 9341.37 | 27994.55 | +9.34% | 15.99% | 2 |
| DEWS-U | 9352.37 | 28020.40 | +9.23% | 15.81% | 3 |
| DEWS-T | 9352.86 | 28023.98 | +9.23% | 15.80% | 4 |
| DEWS-IV | 9357.88 | 28049.12 | +9.18% | 15.72% | 5 |
| DEWS-V | 9360.45 | 28022.65 | +9.15% | 15.68% | 6 |
| ValWeightedAverage | 9394.79 | 28101.27 | +8.82% | 15.11% | 7 |
| SimpleAverage | 9395.08 | 28101.55 | +8.82% | 15.10% | 8 |
| StackingRF | 9446.39 | 28271.10 | +8.32% | 14.25% | 9 |
| RidgeStacking | 9508.57 | 28487.51 | +7.72% | 13.21% | 10 |
| StackingGB | 9812.41 | 29709.78 | +4.77% | 8.16% | 11 |
| OLA | 10110.17 | 30244.48 | +1.88% | 3.21% | 12 |
| SingleBest | 10303.52 | 30766.08 | 0.00% | 0.00% | 13 |

## Results — Subspace-DT

| Method | Mean | Std | % vs SingleBest | % HC | Rank |
|---|---:|---:|---:|---:|---:|
| Oracle | 3157.77 | 10307.06 | +72.57% | 100.00% | 1 |
| RidgeStacking | 8255.53 | 24593.48 | +28.29% | 38.98% | 2 |
| ValWeightedAverage | 8298.71 | 24681.26 | +27.92% | 38.47% | 3 |
| SimpleAverage | 8316.16 | 24720.09 | +27.77% | 38.26% | 4 |
| DEWS-I | 8354.85 | 24916.49 | +27.43% | 37.80% | 5 |
| DEWS-U | 8367.60 | 24959.99 | +27.32% | 37.64% | 6 |
| StackingRF | 8375.23 | 25016.05 | +27.25% | 37.55% | 7 |
| DEWS-IV | 8378.61 | 24978.00 | +27.22% | 37.51% | 8 |
| DEWS-V | 8380.19 | 25013.67 | +27.21% | 37.49% | 9 |
| DEWS-T | 8395.35 | 25056.80 | +27.08% | 37.31% | 10 |
| StackingGB | 8633.06 | 26098.24 | +25.01% | 34.47% | 11 |
| SingleBest | 11512.63 | 34278.40 | 0.00% | 0.00% | 12 |
| OLA | 11737.91 | 35179.77 | −1.96% | −2.70% | 13 |

## Results — Subspace-KNN

| Method | Mean | Std | % vs SingleBest | % HC | Rank |
|---|---:|---:|---:|---:|---:|
| Oracle | 3889.14 | 12291.13 | +60.97% | 100.00% | 1 |
| StackingRF | 8505.85 | 25410.76 | +14.64% | 24.01% | 2 |
| RidgeStacking | 8644.44 | 25336.81 | +13.25% | 21.73% | 3 |
| StackingGB | 8703.05 | 26180.23 | +12.66% | 20.76% | 4 |
| DEWS-I | 8704.69 | 25886.80 | +12.64% | 20.74% | 5 |
| DEWS-IV | 8719.70 | 25916.09 | +12.49% | 20.49% | 6 |
| DEWS-V | 8735.17 | 25945.31 | +12.34% | 20.23% | 7 |
| DEWS-U | 8738.56 | 25996.26 | +12.30% | 20.18% | 8 |
| DEWS-T | 8738.58 | 25980.58 | +12.30% | 20.18% | 9 |
| ValWeightedAverage | 9022.31 | 26675.17 | +9.46% | 15.51% | 10 |
| SimpleAverage | 9046.49 | 26739.44 | +9.21% | 15.11% | 11 |
| OLA | 9808.66 | 29284.05 | +1.56% | 2.57% | 12 |
| SingleBest | 9964.52 | 29567.24 | 0.00% | 0.00% | 13 |

---

## Cross-pool summary — best DEWS vs best non-oracle

`ΔSB` = % improvement over `SingleBest` (positive = better). `Gap` = DEWS ΔSB − best
non-oracle ΔSB; a large negative gap means DEWS is losing meaningful ground to a
non-DES ensemble on that pool.

| Pool | Best DEWS | DEWS mean | DEWS ΔSB | Best non-oracle | Best ΔSB | Gap |
|---|---|---:|---:|---|---:|---:|
| Bagging-DT | DEWS-I | 7798.74 | +32.23% | DEWS-I | +32.23% | 0.00 pp |
| Bagging-KNN | DEWS-I | 9341.37 | +9.34% | DEWS-I | +9.34% | 0.00 pp |
| Heterogeneous | DEWS-V | 6769.99 | −1.48% | RidgeStacking | +3.03% | −4.51 pp |
| Subspace-DT | DEWS-I | 8354.85 | +27.43% | RidgeStacking | +28.29% | −0.86 pp |
| Subspace-KNN | DEWS-I | 8704.69 | +12.64% | StackingRF | +14.64% | −2.00 pp |

---