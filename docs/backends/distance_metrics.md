# Distance metrics

Distance metrics are different formulas used by KNN/ANN to measure distance and select the nearest neighbors. 
The compatible ones vary depending on the backend.

Use the table below to select a distance metric(s) supported by the backend being used:

### Legend  
- ✅ Native support 
- 🔁 Fallback to exact `sklearn` KNN
- ❌ Not supported  

| Metric | Why it matters for DES | KNN (exact) | FAISS flat/HNSW | FAISS IVF | Annoy | HNSW (hnswlib) | HNSW (nmslib) |
|--------|-------------------------|-------------|-----------------|-----------|-------|----------------|---------------|
| `euclidean` | Standard L2 norm; best default for continuous feature spaces. Defines spherical local regions for selecting competent classifiers. | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `manhattan` | L1 norm; robust to outliers and often preferable in high‑dimensional feature spaces (mitigates the curse of dimensionality for DES). | ✅ | ✅ | 🔁 | ✅ | ❌ | ✅ |
| `chebyshev` | L∞ norm; focuses on the worst‑case feature deviation. Useful when a single dominant feature dictates similarity (e.g., bounded grids). | ✅ | ✅ | 🔁 | ❌ | ❌ | ✅ |
| `minkowski` | Generalises L1/L2 via `p`; lets you tune the shape of the local neighbourhood between Manhattan and Euclidean. | ✅ | ✅ | 🔁 | ❌ | ❌ | ❌ |
| `cosine` | Angle between vectors, ignoring magnitude. Ideal for normalised embeddings (deep features, TF‑IDF) where direction, not scale, matters for classifier competence. | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `dot` | Inner product (not a metric). Used for max‑inner‑product search (e.g., recommender systems). Generally `not recommended` for standard DES unless you specifically need relevance‑based ranking. | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `canberra` | Weighted L1 that emphasises relative changes; sensitive to values near zero. Useful for sparse count data or features where zeros are meaningful. | ✅ | ✅ | 🔁 | ❌ | ❌ | ❌ |
| `braycurtis` | Normalised L1 bounded to [0,1]; common for compositional or ecological data (e.g., abundance vectors). | ✅ | ✅ | 🔁 | ❌ | ❌ | ❌ |
| `jensenshannon` | Symmetric KL divergence; requires non‑negative inputs. Ideal for measuring similarity between `classifier output probability distributions` or bag‑of‑word features. | ❌ | ✅ | 🔁 | ❌ | ❌ | ❌ |
| `correlation` | Pearson correlation distance (`1‑ρ`). Good for time‑series or features with linear trends – helps select classifiers that generalise across similar patterns. | ✅ | 🔁 | 🔁 | ❌ | ❌ | ❌ |
| `hamming` | Fraction of differing binary/categorical components. Directly applicable to binary feature spaces or binarised classifier predictions. | ✅ | 🔁 | 🔁 | ✅ | ❌ | ❌ |
---