# Algorithm Selection Guide

Selecting the best algorithm to use for your specific case can be difficult, which is why this
guide was assembled based on theory and empirical data to help you decide which one to use.

Algorithm list:

- [DEWS-U](algorithms/dewsu.md) — soft blending via distance-weighted softmax. 
- [DEWS-I](algorithms/dewsi.md) — Like DEWS-U but scores are inverse-distance weighted. General recommendation for regression.
- [DEWS-T](algorithms/dewst.md) — Like DEWS-I but fits a weighted trend line over neighbor scores.
- [DEWS-V](algorithms/dewsv.md) — Like DEWS-U but scores are variance-penalized.
- [DEWS-IV](algorithms/dewsiv.md) — Like DEWS-V but scores are also inverse-distance weighted.   
- [LWSE-U](algorithms/lwseu.md) — Per-sample NNLS weight estimation over the local neighbourhood.
- [LWSE-I](algorithms/lwsei.md) — Like LWSE-U but rows are inverse-distance weighted.   
- [KNORA-U](algorithms/knorau.md) — vote-count weighting. Safe default for classification.
- [KNORA-E](algorithms/knorae.md) — intersection-based. Best when models have clear regional dominance.
- [KNORA-IU](algorithms/knoraiu.md) — like KNORA-U with inverse-distance weighted votes.
- [OLA](algorithms/ola.md) — single model selection. Useful as a diagnostic tool.

## Regression

The entire DEWS family can be used for regression, as well as the LWSE family. KNORA algorithms can be used for 
regression if the target value is classification-like, so it's usually contained within a certain range and it's discrete, 
but it is not usually recommended.

For general-purpose, plug-and-play regression tasks, use one of the following algorithms:

- [DEWS-I](algorithms/dewsi.md): simple and consistent. Best on smooth and heterogeneous datasets and pools.
- [DEWS-T](algorithms/dewst.md): it can get more performance than DEWS-I, but is less
consistent. Tune the r2 threshold parameter to match DEWS-I more or less closely. Best on regions with directional structure,
so a models performance linearly increases or decreases when approaching certain regions.
- [OLA](algorithms/ola.md): it selects a single model, so it tends to perform worse than the other algorithms unless a single 
model has very strong local dominance, but this means you only have to run one model's inference per test point, which can cut
computation.

If you want to get potentially better results and don't mind trying multiple algorithms, analyzing your dataset more in depth,
and/or tuning hyperparameters, consider the following alongside the previous ones:

- [DEWS-IV](algorithms/dewsiv.md): performs worse than the other algorithms in most cases, but can very rarely perform well
in noisy datasets where consistency is important. 
- [LWSE-I](algorithms/lwsei.md): has the potential to improve performance, but is less consistent than
the previous ones. Best when a dataset has clear local structure and with larger values of k. It is more computationally
expensive than the other algorithms and is more inconsistent and k-dependent.

## Classification

Most regression algorithms can work for classification tasks, although many require you to pass confidence scores instead of
raw predictions. On top of the algorithms mentioned in the regression section, KNORA variants are better for classification.

For general-purpose, plug-and-play classification tasks, use one of the following algorithms:

- [DEWS-T](algorithms/dewst.md): for classification, this algorithm is safe and more consistent than for regression, usually
matching or beating DEWS-I. However, the threshold parameter should still be held high. Best on regions with directional structure,
so a models performance linearly increases or decreases when approaching certain regions.
- [DEWS-I](algorithms/dewsi.md): simple and consistent. Best on smooth and heterogeneous datasets and pools.
- [KNORA-IU](algorithms/knoraiu.md): safe and consistent. Best when the dataset has class overlaps or imbalanced datasets.
- [OLA](algorithms/ola.md): it selects a single model, so it tends to perform worse than the other algorithms unless a single 
model has very strong local dominance, but this means you only have to run one model's inference per test point, which can cut
computation.

If you want to get potentially better results and don't mind trying multiple algorithms, analyzing your dataset more in depth,
and/or tuning hyperparameters, consider the following alongside the previous ones:

- [DEWS-IV](algorithms/dewsiv.md): performs worse than the other algorithms in most cases, but can very rarely perform well
in noisy datasets where consistency is important. 
- [LWSE-I](algorithms/lwsei.md): has the potential to improve performance drastically, but is less consistent than
the previous ones. Best when a dataset has clear local structure and with larger values of k. It is more computationally
expensive than the other algorithms and is more inconsistent and k-dependent.
- [KNORA-E](algorithms/knorae.md): similar situation to DEWS-IV but even more pronounced, its failures are more extreme and
its successes are rarer, so it is generally not recommended to be considered, as it is a very aggressive algorithm. 
However, it can rarely work when other algorithms struggle, especially when there are clear regional specialist.

## Not recommended for use

These algorithms are not recommended, mostly because there is a different algorithm that does the same thing but better.

- [KNORA-U](algorithms/knorau.md): not a bad algorithm, simple and consistent, but KNORA-IU is better in basically every scenario
- [DEWS-U](algorithms/dewsu.md): once again, simple and consistent, but DEWS-I is better in almost every case
- [LWSE-U](algorithms/lwseu.md): basically a downgrade from LWSE-I, performs well in the same areas but always worse than LWSE-I
- [DEWS-V](algorithms/dewsv.md): weak and inconsistent, almost never performs well, and when it does, DEWS-IV is a better choice