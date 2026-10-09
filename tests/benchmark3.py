"""
benchmark.py
============
Full benchmark pipeline for "Extending DES to Regression".

Pools are built with a global pruning step: split training 80/20 into
sub-train/validation, tune (heterogeneous only) or fit each candidate on
sub-train, prune anything with validation error > 1.2x the best (keeping
>= 2), then retrain survivors on the full training set and predict on
X_comp/X_test. X_comp is used only for DES/stacking tuning and final
ensemble fitting, never for pruning or base-model training.

Hyperparameters (DES, stacking, base models) are all tuned via Random
Search (Bergstra & Bengio 2012) with the same RS budget of 50 evaluations.
"""

import os
import sys
import time
import hashlib
import warnings
import logging
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, uniform as sp_uniform

# ──────────────────────────────────────────────────────────────────────────────
# Silence Treelite warnings (from cuML RandomForest)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TREELITE_LOG_LEVEL"] = "0"

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None
    _XGB_AVAILABLE = False

# DISABLE_GPU: at this benchmark's scale, cuML's per-call CUDA overhead
# outweighs what it saves, and many `loky` workers sharing one GPU with no
# cross-process RMM coordination caused repeated cudaErrorIllegalAddress
# crashes. Flip to False for a future run with larger per-fit data.
DISABLE_GPU = True

if DISABLE_GPU:
    _CUML_AVAILABLE = False
else:
    try:
        import cuml
        from cuml import Ridge as RidgeGPU
        from cuml.neighbors import KNeighborsRegressor as KNN_GPU
        from cuml.ensemble import RandomForestRegressor as RF_GPU
        _CUML_AVAILABLE = True
    except ImportError:
        _CUML_AVAILABLE = False

# ── scikit-learn imports ──────────────────────────────────────────────────────
from sklearn.datasets import fetch_openml
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (required to unlock HalvingRandomSearchCV)
from sklearn.model_selection import KFold, RandomizedSearchCV, HalvingRandomSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor as KNN_SK
from sklearn.tree import DecisionTreeRegressor as DT_SK
from sklearn.linear_model import Ridge as Ridge_SK, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor as RF_SK
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin, clone
import openml

# ──────────────────────────────────────────────────────────────────────────────
# Logging: per-task worker log files + a main.log; terminal only gets ETA
# lines and summary tables (see _console below).
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes the stream after every record."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_worker_logger(dataset_id: int, seed: int, fold: int) -> logging.Logger:
    """Logger writing to logs/worker_{did}_{seed}_{fold}.log (file-only,
    propagate=False so per-task records don't also land in main.log)."""
    logger = logging.getLogger(f"worker_{dataset_id}_{seed}_{fold}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    log_file = LOG_DIR / f"worker_{dataset_id}_{seed}_{fold}.log"
    fh = FlushFileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Thread pinning
# ──────────────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
# Main process logger: file-only (logs/main.log). Terminal output is limited
# to the ETA line and per-dataset summary tables (see _console below).
MAIN_LOG_FILE = LOG_DIR / "main.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    handlers=[FlushFileHandler(MAIN_LOG_FILE, mode="a", encoding="utf-8")],
)
main_logger = logging.getLogger(__name__)


def _console(msg: str = "") -> None:
    """Print curated, low-noise output directly to the terminal (ETA lines,
    summary tables); everything else goes to log files."""
    print(msg, flush=True)

# ── Suppress stderr for C++ logs ────────────────────────────────────────────
@contextmanager
def suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
        os.close(devnull)

# ── Choose implementations ──────────────────────────────────────────────────
if _CUML_AVAILABLE:
    main_logger.info("cuML detected – using GPU‑accelerated Ridge, KNN, and RandomForest.")
    Ridge = RidgeGPU
    KNeighborsRegressor = KNN_GPU
    RandomForestRegressor = RF_GPU
    DecisionTreeRegressor = DT_SK  # keep on CPU (cuML version missing)
else:
    main_logger.info("cuML not found – falling back to scikit‑learn CPU versions.")
    Ridge = Ridge_SK
    KNeighborsRegressor = KNN_SK
    DecisionTreeRegressor = DT_SK
    RandomForestRegressor = RF_SK


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM IMPUTERS
# ──────────────────────────────────────────────────────────────────────────────

class MedianOrZeroImputer(BaseEstimator, TransformerMixin):
    """Impute numeric columns with median if available, else 0 (no warnings)."""
    def fit(self, X, y=None):
        n_cols = X.shape[1]
        medians = []
        for col in range(n_cols):
            col_data = X[:, col]
            if np.any(~np.isnan(col_data)):
                medians.append(np.nanmedian(col_data))
            else:
                medians.append(np.nan)
        self.medians_ = np.array(medians)
        return self

    def transform(self, X):
        X = X.copy()
        for col in range(X.shape[1]):
            if np.isnan(self.medians_[col]):
                X[:, col] = 0.0
            else:
                mask = np.isnan(X[:, col])
                X[mask, col] = self.medians_[col]
        return X


class MostFrequentOrMinusOneImputer(BaseEstimator, TransformerMixin):
    """Impute categorical (coded as -1 for missing) with mode if available, else -1."""
    def fit(self, X, y=None):
        self.modes_ = []
        for col in range(X.shape[1]):
            non_missing = X[:, col][X[:, col] != -1]
            if len(non_missing) > 0:
                vals, counts = np.unique(non_missing, return_counts=True)
                self.modes_.append(vals[np.argmax(counts)])
            else:
                self.modes_.append(-1)
        return self

    def transform(self, X):
        X = X.copy()
        for col in range(X.shape[1]):
            mask = (X[:, col] == -1)
            X[mask, col] = self.modes_[col]
        return X


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SEEDS               = [0, 1, 2]
N_FOLDS             = 10
POOL_SIZE           = 25
MIN_DATASET_SAMPLES = 250
RANDOM_STATE_BASE   = 42
RESULTS_DIR         = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RS_BUDGET          = 50          # for DES and stacking baselines
BASE_TUNE_BUDGET   = 20          # for base model tuning (heterogeneous only)
PRUNE_RATIO        = 1.2         # keep models with MAE ≤ 1.2 × best
VAL_FRACTION       = 0.2         # fraction of training used for validation (pruning)

# Inner-CV fold counts for hyperparameter search (reduced from 5/3 as a
# compute/variance trade-off). Only affects candidate-scoring for stacking
# baselines and heterogeneous base learners — DES/OLA always use LOO scoring
# (see tune_des_loo), which has no k-fold knob to reduce.
STACKING_CV        = 3           # was 5; used by RidgeStacking/StackingRF/StackingGB
BASE_TUNE_CV        = 2          # was 3; used by Heterogeneous pool's RS/Halving base-model tuning

MEMORY_SAFETY_FRACTION = 0.80
N_JOBS = -1

# FAISS HNSW per-vector memory: (d*4 + M*2*4) bytes/vector (d=post-preprocessing
# n_features, M=graph links/node, default 32). Source: faiss wiki "Guidelines
# to choose an index". Override if your deskit build uses a different M.
FAISS_HNSW_M = 32

# Rough fixed VRAM overhead (CUDA context + RMM pool + cuML/XGBoost init) per
# GPU-active worker, independent of data size — a floor, not a precise figure.
GPU_CONTEXT_OVERHEAD_BYTES = 297 * 1024 ** 2

# Fraction of *free* VRAM we're willing to use in total across all workers.
GPU_MEMORY_SAFETY_FRACTION = 0.75


def load_suite_datasets(suite_id: int = 353) -> list[int]:
    suite = openml.study.get_suite(suite_id)
    return list(suite.data)


ALL_DATASET_IDS = load_suite_datasets()


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACES (continuous)
# ──────────────────────────────────────────────────────────────────────────────
K_MAX = 50
MIN_K_ROUTER = 2   # smallest k passed to tune_des_loo's randint(2, ...)
TEMP_MIN, TEMP_MAX = 0.01, 10.0
THRESH_MIN, THRESH_MAX = 0.0, 0.8
R2_MIN, R2_MAX = 0.5, 1.0

DISTANCE_METRICS    = ["euclidean", "manhattan", "chebyshev", "cosine"]
ERROR_METRICS       = ["mae", "rmse"]

DEWS_METHODS = ["DEWS-U", "DEWS-I", "DEWS-V", "DEWS-IV", "DEWS-T"]
OLA_METHOD   = "OLA"
ALL_METHODS  = DEWS_METHODS + [OLA_METHOD]


# ──────────────────────────────────────────────────────────────────────────────
# GPU DETECTION for XGBoost
# ──────────────────────────────────────────────────────────────────────────────
_XGB_GPU_AVAILABLE: bool | None = None

def _check_xgb_gpu() -> bool:
    global _XGB_GPU_AVAILABLE
    if _XGB_GPU_AVAILABLE is not None:
        return _XGB_GPU_AVAILABLE
    if DISABLE_GPU or not _XGB_AVAILABLE:
        # Same reasoning as cuML above: small per-fit data + many concurrent
        # loky workers on one GPU risks CUDA context issues for little gain.
        _XGB_GPU_AVAILABLE = False
        return False
    try:
        probe = xgb.XGBRegressor(device="cuda", n_estimators=1, max_depth=1, verbosity=0)
        probe.fit(np.array([[1.0], [2.0]], dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32))
        _XGB_GPU_AVAILABLE = True
        main_logger.info("XGBoost CUDA GPU detected — StackingGB will run on GPU.")
    except Exception as exc:
        _XGB_GPU_AVAILABLE = False
        main_logger.info(f"XGBoost CUDA not available ({exc}); StackingGB falls back to CPU.")
    return _XGB_GPU_AVAILABLE


# ──────────────────────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _stable_hash(s: str) -> int:
    """Deterministic string -> int hash, stable across processes and runs.

    Python's built-in hash() is salted with a random seed per-process
    (PYTHONHASHSEED) unless explicitly disabled, so two `loky` worker
    processes (or two separate runs) will produce *different* hash values
    for the same string. That silently broke reproducibility of DES tuning
    seeds. sha256 has no such randomization.
    """
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_id: int):
    main_logger.info(f"Loading dataset {dataset_id} from OpenML...")
    ds = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    dataset_name = ds.details.get("name", str(dataset_id))
    df = ds.data.copy()

    target_names_list = list(ds.target_names) if ds.target_names is not None else []
    target_name = target_names_list[0] if target_names_list else None
    target_raw = ds.target
    if isinstance(target_raw, pd.DataFrame):
        target_series = target_raw.iloc[:, 0]
    elif isinstance(target_raw, pd.Series):
        target_series = target_raw
    else:
        target_series = pd.Series(target_raw)
    if target_name and target_name in df.columns:
        df = df.drop(columns=[target_name])

    y = pd.to_numeric(target_series, errors="coerce").values.astype(np.float64)
    valid = ~np.isnan(y)
    df_valid = df[valid].reset_index(drop=True)
    y = y[valid]

    numeric_col_names = df_valid.select_dtypes(include=[np.number]).columns.tolist()
    categorical_col_names = df_valid.select_dtypes(exclude=[np.number]).columns.tolist()
    parts = []
    if numeric_col_names:
        parts.append(df_valid[numeric_col_names].values.astype(np.float64))
    if categorical_col_names:
        cat_df = df_valid[categorical_col_names].astype("category")
        parts.append(cat_df.apply(lambda s: s.cat.codes.astype(np.float64)).values)
    X_raw = np.hstack(parts) if parts else np.empty((len(y), 0), dtype=np.float64)
    n_num = len(numeric_col_names)
    n_cat = len(categorical_col_names)
    main_logger.info(
        f"  '{dataset_name}': {X_raw.shape[0]} samples × {X_raw.shape[1]} features "
        f"({n_num} numeric, {n_cat} categorical) [raw, unscaled]"
    )
    return X_raw, y, dataset_name, n_num, n_cat


def _make_fold_preprocessor(n_num: int, n_cat: int):
    numeric_pipe = Pipeline([
        ("impute", MedianOrZeroImputer()),
        ("scale",  StandardScaler()),
    ])
    transformers = [("num", numeric_pipe, list(range(n_num)))]
    if n_cat > 0:
        categorical_pipe = Pipeline([
            ("impute", MostFrequentOrMinusOneImputer()),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipe, list(range(n_num, n_num + n_cat))))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ──────────────────────────────────────────────────────────────────────────────
# POOL BUILDING WITH TUNING & PRUNING
# ──────────────────────────────────────────────────────────────────────────────

POOL_NAMES = ["Bagging-KNN", "Bagging-DT", "Subspace-KNN", "Subspace-DT", "Heterogeneous"]

def build_pruned_pool(
    pool_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_comp: np.ndarray,
    X_test: np.ndarray,
    n_features: int,
    random_state: int,
    logger: logging.Logger,
    error_metric: str = "mae",
    prune_ratio: float = PRUNE_RATIO,
) -> tuple[dict, dict, float, float, float]:
    """Returns (comp_preds, test_preds, fit_time_ms, predict_time_ms, tune_time_ms).

    error_metric controls which validation error pruning ranks/cuts on
    (call once per error_metric so each metric's pool is pruned on that
    same metric). prune_ratio is exposed (not read from PRUNE_RATIO
    directly) so callers like the pruning ablation can sweep it. tune_time_ms
    isolates the Heterogeneous pool's RS/Halving base-model search cost from
    fit_time_ms (0.0 for the four homogeneous pools, which do no tuning).
    """
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRACTION, random_state=random_state
    )

    models = []
    tune_time_ms = 0.0
    t_fit0 = time.perf_counter()

    if pool_name in ("Bagging-KNN", "Bagging-DT", "Subspace-KNN", "Subspace-DT"):
        logger.info(f"    [BUILD] Homogeneous pool {pool_name}: training {POOL_SIZE} models on sub‑train...")
        is_bagging = pool_name.startswith("Bagging")
        EstimatorClass = KNeighborsRegressor if pool_name.endswith("KNN") else DecisionTreeRegressor
        rng = np.random.RandomState(random_state)
        n_sub_feat = max(1, n_features // 2)
        for i in range(POOL_SIZE):
            name = f"m{i:02d}"
            seed = random_state + i
            m = (EstimatorClass(n_neighbors=5) if EstimatorClass is KNeighborsRegressor
                 else EstimatorClass(random_state=seed))
            if is_bagging:
                # resample_seed is carried through pruning so the SAME
                # per-member bootstrap can be reapplied when retraining on
                # the full training set below. Without this, every retrained
                # KNN member ends up fit on identical data and (since KNN
                # has no internal randomness) the pool collapses to copies.
                idx = resample(np.arange(len(X_sub)), random_state=seed)
                with suppress_stderr():
                    m.fit(X_sub[idx], y_sub[idx])
                models.append((name, m, None, seed))
            else:
                # Diversity here comes from feat_idx, reapplied at retrain
                # time regardless — no row-resampling needed.
                feat_idx = rng.choice(n_features, size=n_sub_feat, replace=False)
                with suppress_stderr():
                    m.fit(X_sub[:, feat_idx], y_sub)
                models.append((name, m, feat_idx, None))

    elif pool_name == "Heterogeneous":
        logger.info(f"    [BUILD] Heterogeneous pool: tuning base models via RS({BASE_TUNE_BUDGET}) on sub‑train...")
        # KNeighborsRegressor has no `random_state` param, so each entry
        # carries an explicit accepts_random_state flag rather than assuming
        # a uniform constructor signature. RF/XGB use HalvingRandomSearchCV
        # (same nominal n_iter candidates, but bad ones are screened out
        # early on a data subset) with tightened n_estimators ranges, since
        # on sub-train-sized data large forests/boosters essentially never win.
        base_models = [
            ("ridge", Ridge, {"alpha": loguniform(1e-3, 1e3)}, True, False),
            ("elasticnet", ElasticNet, {"alpha": loguniform(1e-3, 1e3), "l1_ratio": sp_uniform(0, 1)}, True, False),
            ("knn", KNeighborsRegressor, {"n_neighbors": randint(3, 25)}, False, False),
            ("dt", DecisionTreeRegressor, {"max_depth": [5, 10, 20, None], "min_samples_leaf": randint(1, 10)}, True, False),
            ("rf", RandomForestRegressor, {"n_estimators": randint(50, 150), "max_depth": [5, 10, 20, None]}, True, True),
            ("xgb", xgb.XGBRegressor, {"n_estimators": randint(50, 150), "max_depth": randint(2, 9), "learning_rate": loguniform(5e-3, 0.3)}, True, True),
            ("mlp", MLPRegressor, {"hidden_layer_sizes": [(50,), (100,), (50,50)], "alpha": loguniform(1e-4, 1e-1)}, True, False),
        ]
        for name, ModelClass, param_dist, accepts_random_state, use_halving in base_models:
            estimator = ModelClass(random_state=random_state) if accepts_random_state else ModelClass()
            search_cls = HalvingRandomSearchCV if use_halving else RandomizedSearchCV
            search_kwargs = dict(
                estimator=estimator,
                param_distributions=param_dist,
                cv=BASE_TUNE_CV,
                scoring=("neg_mean_absolute_error" if error_metric == "mae"
                         else "neg_root_mean_squared_error"),
                random_state=random_state,
                n_jobs=1,
            )
            if use_halving:
                search_kwargs["n_candidates"] = BASE_TUNE_BUDGET  # analogue of n_iter for halving
            else:
                search_kwargs["n_iter"] = BASE_TUNE_BUDGET
            rs = search_cls(**search_kwargs)
            t_tune0 = time.perf_counter()
            with suppress_stderr():
                rs.fit(X_sub, y_sub)
            tune_time_ms += (time.perf_counter() - t_tune0) * 1000
            best_m = rs.best_estimator_
            models.append((name, best_m, None, None))

    else:
        raise ValueError(f"Unknown pool: {pool_name}")

    fit_time_ms = (time.perf_counter() - t_fit0) * 1000

    logger.info(f"    [BUILD] Evaluating {len(models)} models on validation set...")
    val_preds = {}
    for name, m, feat_idx, _resample_seed in models:
        if feat_idx is None:
            with suppress_stderr():
                val_preds[name] = m.predict(X_val)
        else:
            with suppress_stderr():
                val_preds[name] = m.predict(X_val[:, feat_idx])

    val_metric_fn = mae if error_metric == "mae" else rmse
    val_errors = {name: val_metric_fn(y_val, val_preds[name]) for name in val_preds}
    best_err = min(val_errors.values())
    keep_names = [name for name, e in val_errors.items() if e <= prune_ratio * best_err]
    if len(keep_names) < 2:
        keep_names = sorted(val_errors.items(), key=lambda x: x[1])[:2]
        keep_names = [name for name, _ in keep_names]

    logger.info(
        f"    [BUILD] Pruned from {len(models)} to {len(keep_names)} models "
        f"on {error_metric.upper()} (kept: {keep_names})"
    )

    kept_models = [
        (name, m, feat_idx, resample_seed)
        for name, m, feat_idx, resample_seed in models
        if name in keep_names
    ]
    logger.info(f"    [BUILD] Retraining {len(kept_models)} models on full training set...")

    comp_preds = {}
    test_preds = {}
    t_predict0 = time.perf_counter()

    for name, m_old, feat_idx, resample_seed in kept_models:
        new_m = clone(m_old)

        # Reapply this member's own bootstrap resample of the FULL training
        # set (not sub-train) to preserve Bagging diversity after retraining.
        if resample_seed is not None:
            boot_idx = resample(np.arange(len(X_train)), random_state=resample_seed)
            X_fit, y_fit = X_train[boot_idx], y_train[boot_idx]
        else:
            X_fit, y_fit = X_train, y_train

        if feat_idx is not None:
            with suppress_stderr():
                new_m.fit(X_fit[:, feat_idx], y_fit)
                comp_preds[name] = new_m.predict(X_comp[:, feat_idx])
                test_preds[name] = new_m.predict(X_test[:, feat_idx])
        else:
            with suppress_stderr():
                new_m.fit(X_fit, y_fit)
                comp_preds[name] = new_m.predict(X_comp)
                test_preds[name] = new_m.predict(X_test)

    predict_time_ms = (time.perf_counter() - t_predict0) * 1000
    logger.info(f"    [BUILD] Pool {pool_name} ready.")
    return comp_preds, test_preds, fit_time_ms, predict_time_ms, tune_time_ms


# ──────────────────────────────────────────────────────────────────────────────
# DESKIT ROUTER CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def make_router(method: str, distance_metric: str, error_metric: str,
                k: int = 10, temperature: float = 1.0,
                threshold: float = 0.0, r2_threshold: float = None,
                ann_overrides: dict | None = None):
    """ann_overrides, when given, REPLACES (not merges into) the ANN backend
    kwargs (preset/finder/index_type) — e.g. dict(preset="exact") for the
    exact-search ablation, which needs no finder/index_type at all. Leave as
    None for the normal custom/faiss/hnsw behavior used everywhere else.
    """
    ann_kwargs = ann_overrides if ann_overrides is not None else dict(
        preset="custom", finder="faiss", index_type="hnsw",
    )
    dews_common = dict(
        task="regression",
        mode="min",
        metric=error_metric,
        k=k,
        threshold=threshold,
        temperature=temperature,
        distance_metric=distance_metric,
        **ann_kwargs,
    )
    ola_common = dict(
        task="regression",
        mode="min",
        metric=error_metric,
        k=k,
        distance_metric=distance_metric,
        **ann_kwargs,
    )
    if method == "DEWS-U":
        from deskit import DEWSU
        return DEWSU(**dews_common)
    elif method == "DEWS-I":
        from deskit import DEWSI
        return DEWSI(**dews_common)
    elif method == "DEWS-V":
        from deskit import DEWSV
        return DEWSV(**dews_common)
    elif method == "DEWS-IV":
        from deskit import DEWSIV
        return DEWSIV(**dews_common)
    elif method == "DEWS-T":
        from deskit import DEWST
        return DEWST(**dews_common, r2_threshold=r2_threshold)
    elif method == "OLA":
        from deskit import OLA
        return OLA(**ola_common)
    else:
        raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────────────────────────────────────
# DES HYPERPARAMETER TUNING — LOO Random Search (continuous)
# ──────────────────────────────────────────────────────────────────────────────

def tune_des_loo(
    method: str,
    error_metric: str,
    X_comp: np.ndarray,
    y_comp: np.ndarray,
    comp_preds: dict,
    random_state: int,
    n_iter: int = RS_BUDGET,
    ann_overrides: dict | None = None,
) -> tuple[dict | None, float, list[dict]]:
    """Returns (best_cfg, tune_time_ms, trials).

    ann_overrides is forwarded as-is to make_router on every RS candidate
    (see make_router's docstring) — used by the ANN ablation to run the LOO
    search itself under exact search, not just the final fit.

    tune_time_ms covers the whole n_iter-candidate LOO loop — the DES/OLA
    analogue of RandomizedSearchCV/HalvingRandomSearchCV's search time for
    baselines — recorded separately from evaluate_des's fit/predict times
    (which only cover the single final fit on best_cfg).

    trials holds every RS candidate (config + LOO score, NaN on failure),
    not just the winner — the raw material for hyperparameter sensitivity
    analysis (see analyze_des_sensitivity).
    """
    t_tune0 = time.perf_counter()
    rng = np.random.RandomState(random_state)
    n_pool = len(comp_preds)
    trials: list[dict] = []
    if n_pool < 2:
        return None, (time.perf_counter() - t_tune0) * 1000, trials

    best_score = np.inf
    best_cfg: dict | None = None

    for trial_idx in range(n_iter):
        dm = rng.choice(DISTANCE_METRICS)
        k = int(rng.randint(2, min(K_MAX, n_pool) + 1))

        if method == OLA_METHOD:
            temperature, threshold, r2_threshold = 1.0, 0.0, None
            predict_kwargs: dict = dict(k=k)
        elif method == "DEWS-T":
            temperature = float(10 ** rng.uniform(np.log10(TEMP_MIN), np.log10(TEMP_MAX)))
            threshold   = float(rng.uniform(THRESH_MIN, THRESH_MAX))
            r2_threshold = float(rng.uniform(R2_MIN, R2_MAX))
            predict_kwargs = dict(
                k=k, temperature=temperature,
                threshold=threshold, r2_threshold=r2_threshold,
            )
        else:
            temperature = float(10 ** rng.uniform(np.log10(TEMP_MIN), np.log10(TEMP_MAX)))
            threshold   = float(rng.uniform(THRESH_MIN, THRESH_MAX))
            r2_threshold = None
            predict_kwargs = dict(k=k, temperature=temperature, threshold=threshold)

        score = np.nan
        try:
            router = make_router(
                method, dm, error_metric, k, temperature, threshold, r2_threshold,
                ann_overrides=ann_overrides,
            )
            router.fit(X_comp, y_comp, comp_preds)
            loo_preds = router.predict(X_comp, comp_preds, loo=True, **predict_kwargs)
            score = mae(y_comp, loo_preds) if error_metric == "mae" else rmse(y_comp, loo_preds)
            if score < best_score:
                best_score = score
                best_cfg = dict(
                    distance_metric=dm,
                    error_metric=error_metric,
                    k=k,
                    temperature=temperature,
                    threshold=threshold,
                    r2_threshold=r2_threshold,
                )
        except Exception:
            pass

        trials.append(dict(
            trial=trial_idx,
            distance_metric=dm,
            k=k,
            temperature=temperature,
            threshold=threshold,
            r2_threshold=r2_threshold,
            loo_score=score,
        ))

    tune_time_ms = (time.perf_counter() - t_tune0) * 1000
    return best_cfg, tune_time_ms, trials


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE DES EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_des(
    method: str,
    k: int,
    temperature: float,
    threshold: float,
    distance_metric: str,
    error_metric: str,
    r2_threshold: float | None,
    X_comp: np.ndarray,
    y_comp: np.ndarray,
    comp_preds: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_preds: dict,
    ann_overrides: dict | None = None,
) -> dict:
    row = dict(
        method=method, k=k, temperature=temperature, threshold=threshold,
        distance_metric=distance_metric, error_metric=error_metric,
        r2_threshold=(r2_threshold if method == "DEWS-T" else None),
    )
    if k > len(comp_preds):
        row.update(dict(
            mae=np.nan, rmse=np.nan,
            fit_time_ms=np.nan, predict_time_ms=np.nan, weight_time_ms=np.nan,
            weight_entropy=np.nan, weight_mean_max=np.nan, n_active_mean=np.nan,
            error=f"k={k} exceeds pool size {len(comp_preds)}",
        ))
        return row

    try:
        router = make_router(
            method=method,
            distance_metric=distance_metric,
            error_metric=error_metric,
            k=k,
            temperature=temperature if method != OLA_METHOD else 1.0,
            threshold=threshold     if method != OLA_METHOD else 0.0,
            r2_threshold=(r2_threshold if method == "DEWS-T" else None),
            ann_overrides=ann_overrides,
        )
        t0 = time.perf_counter()
        router.fit(X_comp, y_comp, comp_preds)
        fit_time_ms = (time.perf_counter() - t0) * 1000

        if method == OLA_METHOD:
            predict_kwargs: dict = dict(k=k)
        else:
            predict_kwargs = dict(k=k, temperature=temperature, threshold=threshold)
            if method == "DEWS-T":
                predict_kwargs["r2_threshold"] = r2_threshold

        t0 = time.perf_counter()
        preds = router.predict(X_test, test_preds, **predict_kwargs)
        predict_time_ms = (time.perf_counter() - t0) * 1000

        if method != OLA_METHOD:
            t0 = time.perf_counter()
            weights_out = router.predict_weights(X_test, **predict_kwargs)
            weight_time_ms = (time.perf_counter() - t0) * 1000
            model_names = list(comp_preds.keys())
            if isinstance(weights_out, list):
                weight_arr = np.array(
                    [[w.get(m, 0.0) for m in model_names] for w in weights_out]
                )
            else:
                weight_arr = np.asarray(weights_out)
            eps = 1e-10
            log_w = np.log(np.clip(weight_arr, eps, 1.0))
            weight_entropy  = float(np.mean(-np.sum(weight_arr * log_w, axis=1)))
            weight_mean_max = float(np.mean(weight_arr.max(axis=1)))
            n_active_mean   = float(np.mean((weight_arr > 1e-4).sum(axis=1)))
        else:
            weight_time_ms = np.nan
            weight_entropy = weight_mean_max = n_active_mean = np.nan

        row.update(dict(
            mae=mae(y_test, preds), rmse=rmse(y_test, preds),
            fit_time_ms=fit_time_ms, predict_time_ms=predict_time_ms,
            weight_time_ms=weight_time_ms, weight_entropy=weight_entropy,
            weight_mean_max=weight_mean_max, n_active_mean=n_active_mean,
            error=None,
        ))
    except Exception as exc:
        # We don't have a logger here; the caller can log if needed.
        row.update(dict(
            mae=np.nan, rmse=np.nan,
            fit_time_ms=np.nan, predict_time_ms=np.nan, weight_time_ms=np.nan,
            weight_entropy=np.nan, weight_mean_max=np.nan, n_active_mean=np.nan,
            error=str(exc),
        ))
    return row


# ──────────────────────────────────────────────────────────────────────────────
# BASELINE EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def _fit_predict_timed(estimator, param_dist, search_cls, n_iter_key,
                        comp_mat, y_comp, test_mat,
                        cv_k, cv_scoring, random_state, n_iter, **extra):
    """Fits search_cls(estimator, param_dist) on (comp_mat, y_comp), predicts
    on test_mat. Returns (preds, fit_time_ms, predict_start_time) —
    predict_start_time is captured immediately before .predict(), so
    time.perf_counter() - predict_start_time gives the actual predict
    duration when computed right after (see _record in evaluate_baselines)."""
    kwargs = {n_iter_key: n_iter, "cv": cv_k, "scoring": cv_scoring,
              "random_state": random_state, "n_jobs": 1, **extra}
    rs = search_cls(estimator, param_dist, **kwargs)
    t_fit = time.perf_counter()
    with suppress_stderr():
        rs.fit(comp_mat, y_comp)
    fit_ms = (time.perf_counter() - t_fit) * 1000
    t_pred = time.perf_counter()
    with suppress_stderr():
        preds = rs.best_estimator_.predict(test_mat)
    return preds, fit_ms, t_pred


def evaluate_baselines(
    comp_preds: dict,
    test_preds: dict,
    y_comp: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    error_metric: str = "mae",
    n_iter: int = RS_BUDGET,
) -> dict:
    """error_metric drives both which competence-set error ranks/weights
    SingleBest & ValWeightedAverage, and the CV scoring for the stacking
    models below. Oracle's per-point argmin is metric-invariant (minimizing
    |error| and error^2 pick the same model at any single point)."""
    results: dict = {}
    model_names = list(comp_preds.keys())
    comp_mat = np.stack([comp_preds[m] for m in model_names], axis=1)
    test_mat = np.stack([test_preds[m] for m in model_names], axis=1)
    comp_metric_fn = mae if error_metric == "mae" else rmse
    comp_errs = np.array([comp_metric_fn(y_comp, comp_preds[m]) for m in model_names])
    n_comp = len(y_comp)
    cv_k = max(2, min(STACKING_CV, n_comp // 5))
    cv_scoring = "neg_mean_absolute_error" if error_metric == "mae" else "neg_root_mean_squared_error"

    def _record(name: str, preds: np.ndarray, t_pred: float,
                fit_time_ms: float | None = None) -> None:
        results[name] = dict(
            mae=mae(y_test, preds),
            rmse=rmse(y_test, preds),
            predict_time_ms=(time.perf_counter() - t_pred) * 1000,
            fit_time_ms=fit_time_ms if fit_time_ms is not None else np.nan,
        )

    t0 = time.perf_counter()
    _record("SingleBest", test_mat[:, int(comp_errs.argmin())], t0)

    t0 = time.perf_counter()
    _record("SimpleAverage", test_mat.mean(axis=1), t0)

    t0 = time.perf_counter()
    w = 1.0 / np.maximum(comp_errs, 1e-8)
    w /= w.sum()
    _record("ValWeightedAverage", test_mat @ w, t0)

    t0 = time.perf_counter()
    per_pt = np.abs(test_mat - y_test[:, None])
    oracle_preds = test_mat[np.arange(len(y_test)), per_pt.argmin(axis=1)]
    _record("Oracle", oracle_preds, t0)

    if n_comp < 10:
        for name in ("RidgeStacking", "StackingRF", "StackingGB"):
            t0 = time.perf_counter()
            _record(name, test_mat.mean(axis=1), t0, np.nan)
        return results

    # Ridge Stacking — left on plain RandomizedSearchCV: a single linear
    # solve per candidate is already cheap, so halving buys little here.
    preds, fit_ms, t_pred = _fit_predict_timed(
        Ridge(fit_intercept=True), {"alpha": loguniform(1e-3, 1e3)},
        RandomizedSearchCV, "n_iter",
        comp_mat, y_comp, test_mat, cv_k, cv_scoring, random_state, n_iter,
    )
    _record("RidgeStacking", preds, t_pred, fit_ms)

    # RF Stacking — HalvingRandomSearchCV: bad candidates get screened out
    # early on a data subset. n_estimators tightened (400->150): large
    # forests are essentially never the winner at this competence-set scale.
    rf_param_dist = {
        "n_estimators":    randint(50, 150),
        "max_depth":       [3, 5, 7, 10, None],
        "min_samples_leaf": randint(1, 10),
        "max_features":    ["sqrt", "log2", 0.5, None],
    }
    preds, fit_ms, t_pred = _fit_predict_timed(
        RandomForestRegressor(random_state=random_state, max_depth=None), rf_param_dist,
        HalvingRandomSearchCV, "n_candidates",
        comp_mat, y_comp, test_mat, cv_k, cv_scoring, random_state, n_iter,
    )
    _record("StackingRF", preds, t_pred, fit_ms)

    # XGBoost Stacking — same halving + tightened-range treatment as RF.
    if not _XGB_AVAILABLE:
        main_logger.warning("xgboost not installed — StackingGB skipped.")
        t0 = time.perf_counter()
        _record("StackingGB", test_mat.mean(axis=1), t0, np.nan)
    else:
        use_gpu = _check_xgb_gpu()
        device = "cuda" if use_gpu else "cpu"

        gb_param_dist = {
            "n_estimators":     randint(50, 150),
            "max_depth":        randint(2, 9),
            "learning_rate":    loguniform(5e-3, 0.3),
            "subsample":        sp_uniform(0.5, 0.5),
            "colsample_bytree": sp_uniform(0.5, 0.5),
            "min_child_weight": randint(1, 10),
            "reg_alpha":        loguniform(1e-4, 10.0),
            "reg_lambda":       loguniform(1e-4, 10.0),
        }

        def _make_xgb(dev: str):
            return xgb.XGBRegressor(
                device=dev,
                tree_method="hist",
                objective="reg:squarederror",
                verbosity=0,
                nthread=1,
                random_state=random_state,
            )

        def _run_gb_rs(dev: str) -> tuple[np.ndarray, float]:
            preds, fm, _ = _fit_predict_timed(
                _make_xgb(dev), gb_param_dist, HalvingRandomSearchCV, "n_candidates",
                comp_mat, y_comp, test_mat, cv_k, cv_scoring, random_state, n_iter,
                error_score="raise",
            )
            return preds, fm

        try:
            preds_gb, fit_ms = _run_gb_rs(device)
            t_pred = time.perf_counter()
            _record("StackingGB", preds_gb, t_pred, fit_ms)
        except Exception as exc:
            main_logger.warning(
                f"StackingGB {'GPU' if use_gpu else 'CPU'} failed ({exc})"
                + ("; retrying on CPU" if use_gpu else "; skipping")
            )
            if use_gpu:
                try:
                    preds_gb, fit_ms = _run_gb_rs("cpu")
                    t_pred = time.perf_counter()
                    _record("StackingGB", preds_gb, t_pred, fit_ms)
                except Exception as exc2:
                    main_logger.warning(f"StackingGB CPU fallback also failed ({exc2}); skipping.")
                    t0 = time.perf_counter()
                    _record("StackingGB", test_mat.mean(axis=1), t0, np.nan)
            else:
                t0 = time.perf_counter()
                _record("StackingGB", test_mat.mean(axis=1), t0, np.nan)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _save(rows: list, path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_parquet(path, index=False)


def _load_checkpoints() -> tuple[list, list, list, list, set]:
    meta_path = RESULTS_DIR / "meta_features.parquet"
    base_path = RESULTS_DIR / "baseline_results.parquet"
    des_path  = RESULTS_DIR / "des_results.parquet"
    sens_path = RESULTS_DIR / "sensitivity_results.parquet"

    meta_rows: list        = []
    baseline_rows: list     = []
    des_rows: list          = []
    sensitivity_rows: list  = []
    completed_folds: set = set()

    def _try_load(path: Path, store: list, label: str) -> None:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                store.extend(df.to_dict("records"))
                main_logger.info(f"Loaded {len(store)} {label} rows from checkpoint.")
            except Exception as e:
                main_logger.warning(f"Could not read {label} checkpoint: {e}")

    _try_load(meta_path, meta_rows, "meta")
    _try_load(base_path, baseline_rows, "baseline")
    _try_load(des_path,  des_rows,  "DES")
    _try_load(sens_path, sensitivity_rows, "sensitivity")

    for row in meta_rows:
        try:
            completed_folds.add(
                (int(row["dataset_id"]), int(row["seed"]), int(row["fold"]))
            )
        except (KeyError, TypeError):
            pass

    main_logger.info(f"Resuming with {len(completed_folds)} completed fold tasks.")
    return meta_rows, baseline_rows, des_rows, sensitivity_rows, completed_folds


# ──────────────────────────────────────────────────────────────────────────────
# META-FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_meta_features(comp_preds, test_preds, y_comp, y_test,
                           pool_name, n_samples, n_features):
    model_names = list(comp_preds.keys())
    n_models = len(model_names)
    comp_maes = np.array([mae(y_comp, comp_preds[m]) for m in model_names])
    pred_mat = np.stack([comp_preds[m] for m in model_names], axis=1)
    corr_mat = np.corrcoef(pred_mat.T) if n_models > 1 else np.array([[1.0]])
    upper_tri = corr_mat[np.triu_indices(n_models, k=1)]
    mean_pairwise_corr = float(np.mean(upper_tri)) if len(upper_tri) else 0.0
    sm = np.sort(comp_maes)
    n = len(sm)
    gini_num = 2 * np.dot(np.arange(1, n + 1), sm) - (n + 1) * sm.sum()
    pool_error_gini = float(gini_num / (n * sm.sum())) if sm.sum() > 0 else 0.0
    pool_error_cv = float(comp_maes.std() / comp_maes.mean()) if comp_maes.mean() > 0 else 0.0
    best_mae_comp = float(comp_maes.min())
    worst_mae_comp = float(comp_maes.max())
    second_best_mae = float(np.sort(comp_maes)[1]) if n_models > 1 else best_mae_comp
    test_mat = np.stack([test_preds[m] for m in model_names], axis=1)
    simple_avg_preds = test_mat.mean(axis=1)
    simple_avg_mae_test = mae(y_test, simple_avg_preds)
    val_w_raw = 1.0 / np.maximum(comp_maes, 1e-8)
    val_w = val_w_raw / val_w_raw.sum()
    val_wt_mae_test = mae(y_test, test_mat @ val_w)
    single_best_mae_test = mae(y_test, test_mat[:, int(comp_maes.argmin())])
    per_pt_errors = np.abs(test_mat - y_test[:, None])
    oracle_preds = test_mat[np.arange(len(y_test)), per_pt_errors.argmin(axis=1)]
    oracle_mae_test = mae(y_test, oracle_preds)
    test_maes = np.array([mae(y_test, test_preds[m]) for m in model_names])

    return {
        "best_model_comp_mae":        best_mae_comp,
        "worst_model_comp_mae":       worst_mae_comp,
        "worst_to_best_ratio":        worst_mae_comp / max(best_mae_comp, 1e-10),
        "best_to_worst_gap_pct":      (worst_mae_comp - best_mae_comp) / max(worst_mae_comp, 1e-10),
        "best_to_2nd_gap_pct":        (second_best_mae - best_mae_comp) / max(second_best_mae, 1e-10),
        "pool_error_gini":            pool_error_gini,
        "pool_error_cv":              pool_error_cv,
        "mean_pairwise_pred_corr":    mean_pairwise_corr,
        "models_near_best_5pct":      float(np.mean(comp_maes <= best_mae_comp * 1.05)),
        "models_near_best_10pct":     float(np.mean(comp_maes <= best_mae_comp * 1.10)),
        "models_near_best_20pct":     float(np.mean(comp_maes <= best_mae_comp * 1.20)),
        "single_best_mae_test":       single_best_mae_test,
        "simple_avg_mae_test":        simple_avg_mae_test,
        "val_weighted_mae_test":      val_wt_mae_test,
        "oracle_mae_test":            oracle_mae_test,
        "simple_avg_gain_over_best":  (single_best_mae_test - simple_avg_mae_test) / max(single_best_mae_test, 1e-10),
        "val_weighted_gain_over_simple": (simple_avg_mae_test - val_wt_mae_test) / max(simple_avg_mae_test, 1e-10),
        "oracle_headroom_over_simple": (simple_avg_mae_test - oracle_mae_test) / max(simple_avg_mae_test, 1e-10),
        "frac_beats_simple_avg":      float(np.mean(test_maes < simple_avg_mae_test)),
        "frac_beats_val_weighted":    float(np.mean(test_maes < val_wt_mae_test)),
        "n_models":                   float(n_models),
        "is_bagging":                 float("Bagging" in pool_name),
        "is_subspace":                float("Subspace" in pool_name),
        "is_heterogeneous":           float(pool_name == "Heterogeneous"),
        "is_knn":                     float("KNN" in pool_name),
        "num_instances":              float(n_samples),
        "num_features":               float(n_features),
        "instances_per_feature":      float(n_samples / max(n_features, 1)),
        "comp_set_size":              float(len(y_comp)),
        "target_mean":                float(y_comp.mean()),
        "target_std":                 float(y_comp.std()),
        "target_skew":                float(pd.Series(y_comp).skew()),
        "target_kurt":                float(pd.Series(y_comp).kurt()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_fold(
    dataset_id: int,
    seed: int,
    fold_idx: int,
    X_full: np.ndarray,
    y_full: np.ndarray,
    dataset_name: str,
    n_num: int,
    n_cat: int,
    logger: logging.Logger,
):
    """Shared train/comparison/test split + preprocessing for one
    (dataset, seed, fold) task, used by both _worker_main and the ablation
    worker. Returns None if the fold should be skipped (pool too small for
    the minimum router k), else a dict with X_train/y_train, X_comp/y_comp,
    X_test/y_test, n_features, n_comp, and pool_rs."""
    n_samples = len(y_full)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_BASE + seed)
    splits = list(kf.split(X_full))
    traincomp_idx, test_idx = splits[fold_idx]
    X_tc_raw, X_test_raw = X_full[traincomp_idx], X_full[test_idx]
    y_tc, y_test = y_full[traincomp_idx], y_full[test_idx]

    rng = np.random.RandomState(RANDOM_STATE_BASE + seed * 10_000 + fold_idx)
    n_comp = max(10, int(len(X_tc_raw) * 0.25))
    comp_idx = rng.choice(len(X_tc_raw), size=n_comp, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_tc_raw)), comp_idx)

    X_train_raw, y_train = X_tc_raw[train_idx], y_tc[train_idx]
    X_comp_raw,  y_comp  = X_tc_raw[comp_idx],  y_tc[comp_idx]

    preprocessor = _make_fold_preprocessor(n_num, n_cat)
    X_train = preprocessor.fit_transform(X_train_raw).astype(np.float64)
    X_comp  = preprocessor.transform(X_comp_raw).astype(np.float64)
    X_test  = preprocessor.transform(X_test_raw).astype(np.float64)
    n_features = X_train.shape[1]

    min_required_comp = MIN_K_ROUTER
    if min_required_comp >= n_comp:
        logger.warning(
            f"[SKIP] {dataset_name} seed={seed} fold={fold_idx}: "
            f"n_comp={n_comp} <= {min_required_comp}"
        )
        return None

    pool_rs = RANDOM_STATE_BASE + seed * 1_000 + fold_idx

    return dict(
        X_train=X_train, y_train=y_train,
        X_comp=X_comp, y_comp=y_comp,
        X_test=X_test, y_test=y_test,
        n_features=n_features, n_comp=n_comp,
        n_samples=n_samples, pool_rs=pool_rs,
    )


def _tune_and_evaluate_des(
    method: str, error_metric: str,
    X_comp: np.ndarray, y_comp: np.ndarray, comp_preds: dict,
    X_test: np.ndarray, y_test: np.ndarray, test_preds: dict,
    tune_seed: int, ann_overrides: dict | None = None,
    tag_tuned_params: bool = False,
) -> tuple[dict | None, list[dict], float]:
    """tune_des_loo then evaluate_des on its winner — the glue shared by
    _worker_main's DES loop and the ablation worker. Returns
    (row_or_None, trials, tune_time_ms); row is None if tuning found no
    usable config (e.g. pool too small for any k). tag_tuned_params adds
    tuned_k/tuned_dm/tuned_temp/tuned_thresh/tuned_r2 to the row (used by
    the main benchmark's des_results.parquet, not the ablation's)."""
    best_cfg, tune_time_ms, trials = tune_des_loo(
        method=method, error_metric=error_metric, X_comp=X_comp, y_comp=y_comp,
        comp_preds=comp_preds, random_state=tune_seed, n_iter=RS_BUDGET,
        ann_overrides=ann_overrides,
    )
    if best_cfg is None:
        return None, trials, tune_time_ms

    row = evaluate_des(
        method=method, X_comp=X_comp, y_comp=y_comp, comp_preds=comp_preds,
        X_test=X_test, y_test=y_test, test_preds=test_preds,
        ann_overrides=ann_overrides, **best_cfg,
    )
    row["tune_time_ms"] = tune_time_ms
    if tag_tuned_params:
        row.update(
            tuned_k=best_cfg["k"], tuned_dm=best_cfg["distance_metric"],
            tuned_temp=best_cfg.get("temperature"),
            tuned_thresh=best_cfg.get("threshold"),
            tuned_r2=best_cfg.get("r2_threshold"),
        )
    return row, trials, tune_time_ms


def _worker_main(
    dataset_id: int,
    seed: int,
    fold_idx: int,
    X_full: np.ndarray,
    y_full: np.ndarray,
    dataset_name: str,
    n_num: int,
    n_cat: int,
) -> dict:
    logger = setup_worker_logger(dataset_id, seed, fold_idx)
    logger.info(f"[START] {dataset_name} seed={seed} fold={fold_idx} ({len(y_full)} samples)")
    out: dict = dict(meta=[], baseline=[], des=[], sensitivity=[])

    fold_data = _prepare_fold(
        dataset_id, seed, fold_idx, X_full, y_full, dataset_name, n_num, n_cat, logger
    )
    if fold_data is None:
        return out

    X_train, y_train = fold_data["X_train"], fold_data["y_train"]
    X_comp,  y_comp  = fold_data["X_comp"],  fold_data["y_comp"]
    X_test,  y_test  = fold_data["X_test"],  fold_data["y_test"]
    n_features = fold_data["n_features"]
    n_samples  = fold_data["n_samples"]
    pool_rs    = fold_data["pool_rs"]

    for pool_name in POOL_NAMES:
        for error_metric in ERROR_METRICS:
            logger.info(
                f"[WORK] {dataset_name} seed={seed} fold={fold_idx} | "
                f"pool={pool_name} | pruned/tuned on {error_metric}"
            )
            try:
                comp_preds, test_preds, pool_fit_ms, pool_predict_ms, pool_tune_ms = build_pruned_pool(
                    pool_name, X_train, y_train, X_comp, X_test,
                    n_features, pool_rs, logger=logger,
                    error_metric=error_metric,
                )
            except Exception as exc:
                logger.error(f"Pool building failed [{pool_name}, {error_metric}]: {exc}")
                continue

            # Meta-features. Tagged with tuned_error_metric since the pool
            # itself (which models survived pruning) can now differ between
            # the mae and rmse passes — these are two distinct pools, not
            # one pool viewed two ways.
            try:
                meta = extract_meta_features(
                    comp_preds, test_preds, y_comp, y_test,
                    pool_name, n_samples, n_features,
                )
            except Exception as exc:
                logger.error(f"Meta-feature extraction failed [{pool_name}, {error_metric}]: {exc}")
                continue

            meta.update(dict(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                seed=seed,
                fold=fold_idx,
                pool=pool_name,
                tuned_error_metric=error_metric,
                pool_fit_time_ms=pool_fit_ms,
                pool_predict_time_ms=pool_predict_ms,
                pool_tune_time_ms=pool_tune_ms,
            ))
            out["meta"].append(meta)

            # Baselines — now using the metric-matched pool, and CV-scored
            # on this same metric (RidgeStacking/StackingRF/StackingGB),
            # rather than always being MAE-pruned and MAE-scored.
            logger.info(
                f"[BASELINES] {dataset_name} seed={seed} fold={fold_idx} | "
                f"pool={pool_name} | metric={error_metric}"
            )
            try:
                bl = evaluate_baselines(
                    comp_preds, test_preds, y_comp, y_test, pool_rs,
                    error_metric=error_metric,
                )
            except Exception as exc:
                logger.error(f"Baseline evaluation failed [{pool_name}, {error_metric}]: {exc}")
                bl = {}

            for bname, bvals in bl.items():
                out["baseline"].append(dict(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    seed=seed,
                    fold=fold_idx,
                    pool=pool_name,
                    tuned_error_metric=error_metric,
                    baseline=bname,
                    **bvals,
                ))

            # DES tuning + evaluation on this same metric-matched pool.
            for method in ALL_METHODS:
                tune_seed = pool_rs + _stable_hash(f"{method}|{error_metric}") % 100_000
                logger.info(
                    f"[DES-TUNE] {dataset_name} seed={seed} fold={fold_idx} | "
                    f"pool={pool_name} | method={method} | metric={error_metric}"
                )

                row, trials, _ = _tune_and_evaluate_des(
                    method, error_metric, X_comp, y_comp, comp_preds,
                    X_test, y_test, test_preds, tune_seed, tag_tuned_params=True,
                )

                for tr in trials:
                    out["sensitivity"].append(dict(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name,
                        seed=seed,
                        fold=fold_idx,
                        pool=pool_name,
                        method=method,
                        tuned_error_metric=error_metric,
                        **tr,
                    ))

                if row is None:
                    logger.warning(
                        f"tune_des_loo returned None for "
                        f"({method}, {pool_name}, {error_metric}) — skipping"
                    )
                    continue

                row.update(dict(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    seed=seed,
                    fold=fold_idx,
                    pool=pool_name,
                    tuned_error_metric=error_metric,
                ))
                out["des"].append(row)

    logger.info(
        f"[DONE] {dataset_name} seed={seed} fold={fold_idx} "
        f"| meta={len(out['meta'])} baseline={len(out['baseline'])} "
        f"des={len(out['des'])} sensitivity={len(out['sensitivity'])}"
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# WORKER COUNT & MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
# N_JOBS = min(CPU cap, RAM cap, VRAM cap). RAM covers predictions,
# raw/preprocessed data, and the per-worker FAISS HNSW index. VRAM only
# matters when cuML/GPU-XGBoost are active: cuML workers each open their own
# CUDA context on the SAME physical GPU with no cross-process coordination,
# so N_JOBS multiplies VRAM use directly — the binding constraint past 2-3
# GPU-active workers on a typical consumer GPU.

def _estimate_categorical_ohe_width(X: np.ndarray, n_num: int, n_cat: int) -> int:
    """Upper-bound the one-hot-encoded categorical width: X's categorical
    columns (n_num:n_num+n_cat) hold integer codes (see load_dataset), and
    OneHotEncoder expands each into n_unique_codes columns."""
    if n_cat == 0:
        return 0
    cat_block = X[:, n_num:n_num + n_cat]
    return sum(max(1, len(np.unique(cat_block[:, col]))) for col in range(n_cat))


def _estimate_worker_bytes(datasets: dict) -> int:
    """Peak system-RAM footprint of a single worker, in bytes: raw +
    preprocessed data, prediction matrices for all pools/models, one live
    FAISS HNSW index (routers run sequentially within a worker, so peak is
    one index, not n_methods x n_metrics), plus a blanket safety multiplier
    for RandomizedSearchCV/joblib/pandas transient copies."""
    if not datasets:
        return 1 * 1024 ** 3
    worst = 0
    for X, y, name, n_num, n_cat in datasets.values():
        n = len(y)
        n_comp = max(10, int(n * 0.9 * 0.25))
        n_test = max(1, int(n * 0.1))

        pred_bytes = len(POOL_NAMES) * POOL_SIZE * (n_comp + n_test) * 8
        data_bytes = X.nbytes + y.nbytes

        ohe_width = _estimate_categorical_ohe_width(X, n_num, n_cat)
        n_features_post = n_num + ohe_width

        # FAISS HNSW index over X_comp, in the post-preprocessing feature
        # space (float32 inside FAISS regardless of our float64 arrays).
        faiss_bytes_per_vec = n_features_post * 4 + FAISS_HNSW_M * 2 * 4
        faiss_index_bytes = n_comp * faiss_bytes_per_vec

        total = (pred_bytes + data_bytes) * 4 + faiss_index_bytes
        if total > worst:
            worst = total
        main_logger.debug(
            f"  Worker footprint for '{name}': n={n} n_comp={n_comp} "
            f"n_features_post_ohe={n_features_post} "
            f"pred={pred_bytes/1e6:.0f} MB data={data_bytes/1e6:.0f} MB "
            f"faiss_hnsw={faiss_index_bytes/1e6:.0f} MB "
            f"-> est {total/1e6:.0f} MB/worker"
        )
    return max(worst, 256 * 1024 ** 2)


def _gpu_active() -> bool:
    """True if any worker-side code path will touch the GPU."""
    return _CUML_AVAILABLE or _check_xgb_gpu()


def _query_gpu_free_bytes() -> int | None:
    """Free VRAM on GPU 0, via pynvml. Returns None if unavailable."""
    if not _PYNVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(meminfo.free)
    except Exception as exc:
        main_logger.warning(f"pynvml query failed ({exc}); skipping VRAM cap.")
        return None


def _estimate_worker_vram_bytes(datasets: dict) -> int:
    """Peak per-worker VRAM footprint: fixed CUDA/RMM context overhead plus
    the largest single GPU model fit a worker holds at once. Training-data
    device copy dominates over cuML/XGBoost's compact model representations."""
    if not datasets:
        return GPU_CONTEXT_OVERHEAD_BYTES
    worst_train_bytes = 0
    for X, y, name, n_num, n_cat in datasets.values():
        n = len(y)
        n_train = max(1, int(n * 0.9 * 0.75))
        ohe_width = _estimate_categorical_ohe_width(X, n_num, n_cat)
        n_features_post = n_num + ohe_width
        # device copy of X_train (float32) + y_train, plus ~2x headroom for
        # cuML's internal staging buffers during fit (RF/KNN build scratch).
        train_bytes = n_train * n_features_post * 4 * 2
        if train_bytes > worst_train_bytes:
            worst_train_bytes = train_bytes
    return GPU_CONTEXT_OVERHEAD_BYTES + worst_train_bytes


def _choose_n_jobs(datasets: dict | None = None,
                   max_workers: int | None = None) -> int:
    n_logical = os.cpu_count() or 1
    cap = min(max_workers, n_logical) if max_workers else n_logical

    if not _PSUTIL_AVAILABLE:
        fallback = max(1, n_logical // 3)
        main_logger.warning(
            f"psutil not found — defaulting to N_JOBS={fallback}. "
            "Install psutil for memory-safe auto-scaling."
        )
        cpu_ram_chosen = min(fallback, cap)
    else:
        avail_bytes = psutil.virtual_memory().available
        safe_bytes  = avail_bytes * MEMORY_SAFETY_FRACTION
        per_worker  = _estimate_worker_bytes(datasets or {})
        n_by_mem    = max(1, int(safe_bytes // per_worker))
        cpu_ram_chosen = min(cap, n_by_mem)
        main_logger.info(
            f"RAM-safe worker selection: "
            f"{avail_bytes/1e9:.1f} GB avail × {MEMORY_SAFETY_FRACTION:.0%} safe, "
            f"est {per_worker/1e6:.0f} MB/worker (incl. FAISS HNSW) → "
            f"up to {n_by_mem} by RAM, {cap} by CPU"
        )

    if not _gpu_active():
        main_logger.info(f"No GPU backend active (cuML/XGBoost-GPU) → using N_JOBS={cpu_ram_chosen}")
        return cpu_ram_chosen

    free_vram = _query_gpu_free_bytes()
    if free_vram is None:
        # Can't measure VRAM but GPU is active — defaulting to the RAM cap
        # risks over-launching GPU contexts, so fall back to a fixed cap.
        gpu_fallback_cap = 2
        chosen = min(cpu_ram_chosen, gpu_fallback_cap)
        main_logger.warning(
            "GPU backend active (cuML/XGBoost-GPU) but pynvml is unavailable "
            "— cannot measure free VRAM. Capping N_JOBS at "
            f"{gpu_fallback_cap} as a safety fallback. Install pynvml "
            "(`pip install nvidia-ml-py`) for an accurate VRAM-aware cap. "
            f"Using N_JOBS={chosen}."
        )
        return chosen

    per_worker_vram = _estimate_worker_vram_bytes(datasets or {})
    safe_vram = free_vram * GPU_MEMORY_SAFETY_FRACTION
    n_by_vram = max(1, int(safe_vram // per_worker_vram))
    chosen = min(cpu_ram_chosen, n_by_vram)

    main_logger.info(
        f"GPU-safe worker selection: "
        f"{free_vram/1e9:.2f} GB free VRAM × {GPU_MEMORY_SAFETY_FRACTION:.0%} safe, "
        f"est {per_worker_vram/1e6:.0f} MB/worker (context + train data) → "
        f"up to {n_by_vram} by VRAM, {cpu_ram_chosen} by RAM/CPU → using {chosen}"
    )
    if chosen < cpu_ram_chosen:
        main_logger.info(
            "NOTE: VRAM is the binding constraint, not RAM/CPU — expected "
            "when multiple loky workers share one GPU (cuML/RMM doesn't "
            "coordinate GPU memory across processes)."
        )
    return chosen


# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SENSITIVITY ANALYSIS (DEWS / OLA)
# ──────────────────────────────────────────────────────────────────────────────
# Uses the raw RS trial log (every candidate tune_des_loo evaluated, not
# just the winner) to ask which hyperparameters actually move the LOO score
# per method. Fits a small Random Forest surrogate (hyperparameters ->
# loo_score) and reports permutation importance on a held-out split, rather
# than a correlation coefficient — DEWS's k in particular tends toward a
# soft-parabola relationship with score (good in the middle, worse at both
# ends), which a monotonic statistic like Spearman is structurally blind to
# but a tree-based surrogate picks up regardless of shape.
#
# distance_metric is ordinal-encoded (not one-hot) so permuting its one
# column shuffles the whole categorical assignment at once. Trials with
# loo_score = NaN (router raised an exception) are dropped before fitting.

from sklearn.inspection import permutation_importance

# Hyperparameters to test per method. r2_threshold only exists for DEWS-T;
# OLA has no temperature/threshold/r2_threshold at all (see make_router).
_CONTINUOUS_HPARAMS = ["k", "temperature", "threshold", "r2_threshold"]

# Minimum trials required before we trust a surrogate fit + permutation
# importance for a given (method[, metric]) group. Below this, the RF has
# too few points to learn a reliable score surface, and permutation
# importance on a small held-out split becomes mostly noise.
_MIN_TRIALS_FOR_SURROGATE = 50


def _applicable_hparams(method: str) -> list[str]:
    """Which continuous hyperparameters actually exist for this method (see
    make_router / tune_des_loo — OLA has none of these; only DEWS-T has
    r2_threshold)."""
    if method == OLA_METHOD:
        return []
    if method == "DEWS-T":
        return _CONTINUOUS_HPARAMS
    return [h for h in _CONTINUOUS_HPARAMS if h != "r2_threshold"]


def analyze_des_sensitivity(
    sensitivity_rows: list,
    group_by_error_metric: bool = True,
    n_repeats: int = 20,
    test_size: float = 0.25,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compute per-method hyperparameter sensitivity from the RS trial log
    via a Random Forest surrogate + permutation importance.

    For each (method[, tuned_error_metric]) group: fits RandomForestRegressor
    on that method's applicable hyperparameters -> loo_score, holds out
    test_size of the trials, and computes permutation_importance on the
    held-out split (not training — permutation importance on training data
    is inflated by overfitting).

    Returns one row per (method[, tuned_error_metric], hyperparameter) with
    importance_mean/std, n_trials, and the surrogate's held-out R² (a low R²
    means the surrogate hasn't learned much, so treat importances loosely).
    Also prints a table per method, sorted by importance_mean descending.

    Aggregates trials across every dataset/seed/fold, since the question is
    "which hyperparameter matters for method X in general", not per-fold.
    """
    if not sensitivity_rows:
        _console("No sensitivity rows to analyze (sensitivity_results.parquet is empty).")
        return pd.DataFrame()

    df = pd.DataFrame(sensitivity_rows)
    df = df[df["loo_score"].notna()].copy()
    if df.empty:
        _console("All RS trials failed (loo_score is NaN everywhere) — nothing to analyze.")
        return pd.DataFrame()

    group_cols = ["method"] + (["tuned_error_metric"] if group_by_error_metric else [])
    records = []

    for group_key, gdf in df.groupby(group_cols, dropna=False):
        group_vals = group_key if isinstance(group_key, tuple) else (group_key,)
        group_dict = dict(zip(group_cols, group_vals))
        method = group_dict["method"]

        hparams = _applicable_hparams(method)
        feature_cols: list[str] = []
        X_cols = {}

        for hparam in hparams:
            if hparam not in gdf.columns:
                continue
            vals = gdf[hparam].astype(float)
            if hparam == "temperature":
                # sampled log-uniformly — feed the surrogate the space it
                # was actually drawn in, not raw linear space (a linear-space
                # column would make most of the sampled range crowd into a
                # sliver near zero, which a tree splits on poorly).
                vals = np.log10(vals.replace(0, np.nan))
                label = "log10(temperature)"
            else:
                label = hparam
            X_cols[label] = vals
            feature_cols.append(label)

        if "distance_metric" in gdf.columns:
            # Ordinal-encode so one shuffle permutes the whole categorical
            # assignment at once (see module-level note above).
            codes, _ = pd.factorize(gdf["distance_metric"])
            X_cols["distance_metric"] = pd.Series(codes, index=gdf.index).astype(float)
            feature_cols.append("distance_metric")

        if not feature_cols:
            continue

        Xg = pd.DataFrame(X_cols)
        yg = gdf["loo_score"].astype(float)
        valid = Xg.notna().all(axis=1) & yg.notna()
        Xg, yg = Xg.loc[valid], yg.loc[valid]

        if len(yg) < _MIN_TRIALS_FOR_SURROGATE:
            _console(
                f"[{method}] skipping surrogate fit — only {len(yg)} usable trials "
                f"(need >= {_MIN_TRIALS_FOR_SURROGATE})."
            )
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            Xg, yg, test_size=test_size, random_state=random_state
        )

        rf = RF_SK(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=N_JOBS,
        )
        rf.fit(X_tr, y_tr)
        surrogate_r2 = float(rf.score(X_te, y_te))

        perm = permutation_importance(
            rf, X_te, y_te,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=N_JOBS,
            scoring="r2",
        )

        for i, feat in enumerate(feature_cols):
            records.append(dict(
                **group_dict,
                hyperparameter=feat,
                importance_mean=float(perm.importances_mean[i]),
                importance_std=float(perm.importances_std[i]),
                surrogate_r2=surrogate_r2,
                n_trials=int(len(yg)),
            ))

    result = pd.DataFrame(records)
    if result.empty:
        _console("No method group had enough trials for a surrogate fit.")
        return result

    _console("\n" + "=" * 78)
    _console("HYPERPARAMETER SENSITIVITY ANALYSIS (DEWS / OLA)")
    _console("RF surrogate (loo_score ~ hyperparameters) + permutation importance")
    _console("=" * 78)
    for group_key, gdf in result.groupby(group_cols, dropna=False):
        group_vals = group_key if isinstance(group_key, tuple) else (group_key,)
        label = ", ".join(f"{c}={v}" for c, v in zip(group_cols, group_vals))
        r2 = gdf["surrogate_r2"].iloc[0]
        n_trials = gdf["n_trials"].iloc[0]
        _console(f"\n[{label}]  (surrogate held-out R²={r2:.3f}, n_trials={n_trials})")
        show = gdf.sort_values("importance_mean", ascending=False)[
            ["hyperparameter", "importance_mean", "importance_std"]
        ].copy()
        show["importance_mean"] = show["importance_mean"].map(lambda x: f"{x:+.4f}")
        show["importance_std"] = show["importance_std"].map(lambda x: f"{x:.4f}")
        _console(show.to_string(index=False))
    _console("=" * 78)
    _console(
        "Note: importance_mean is the drop in held-out R² when that column is "
        "shuffled — higher means more sensitive. A low surrogate_r2 means the "
        "RF hasn't captured much of the score surface, so treat that method's "
        "importances as low-confidence."
    )

    out_path = RESULTS_DIR / "des_sensitivity_summary.csv"
    result.to_csv(out_path, index=False)
    main_logger.info(f"Sensitivity summary written to {out_path}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# ABLATION STUDY — pool pruning and ANN backend
# ──────────────────────────────────────────────────────────────────────────────
# Two variants vs. a "control" (normal PRUNE_RATIO + faiss/hnsw), run on a
# few datasets spanning small/medium/large rather than the whole benchmark:
#   - "no_pruning": prune_ratio=+inf (build_pruned_pool keeps everyone with
#     error <= ratio*best, and nothing exceeds +inf) — isolates what pruning
#     itself buys DES/OLA.
#   - "exact_ann": ann_overrides=dict(preset="exact"), applied to BOTH the
#     LOO tuning search and the final fit/predict, so it's "exact throughout"
#     vs. "approximate throughout", not just exact at the very end.
# Only "mae" is used (not both ERROR_METRICS) to halve the cost — the
# question here doesn't need both metrics for a clear answer.

ABLATION_DATASET_COUNT = 3       # small / medium / large, by samples×features
ABLATION_SEEDS          = [0]    # one seed — comparing pipeline variants, not estimating seed variance
ABLATION_N_FOLDS        = 3      # first 3 of N_FOLDS, not all 10
ABLATION_ERROR_METRIC   = "mae"

ABLATION_VARIANTS = [
    # (variant_name, prune_ratio, ann_overrides)
    ("control",    PRUNE_RATIO,   None),
    ("no_pruning", float("inf"),  None),
    ("exact_ann",  PRUNE_RATIO,   dict(preset="exact")),
]


def _select_ablation_datasets(
    all_datasets: dict[int, tuple],
    dataset_work: dict[int, float],
    n: int = ABLATION_DATASET_COUNT,
) -> list[int]:
    """Pick n datasets evenly spaced across the sorted dataset_work range
    (samples x features), so the ablation covers small/medium/large problems
    instead of n arbitrary ones."""
    ids_sorted = sorted(dataset_work, key=lambda d: dataset_work[d])
    if len(ids_sorted) <= n:
        return ids_sorted
    idxs = sorted(set(np.linspace(0, len(ids_sorted) - 1, n).round().astype(int)))
    return [ids_sorted[i] for i in idxs]


def _ablation_worker_main(
    dataset_id: int,
    seed: int,
    fold_idx: int,
    X_full: np.ndarray,
    y_full: np.ndarray,
    dataset_name: str,
    n_num: int,
    n_cat: int,
) -> list[dict]:
    logger = setup_worker_logger(dataset_id, seed, fold_idx)
    logger.info(f"[ABLATION-START] {dataset_name} seed={seed} fold={fold_idx}")
    rows: list[dict] = []

    fold_data = _prepare_fold(
        dataset_id, seed, fold_idx, X_full, y_full, dataset_name, n_num, n_cat, logger
    )
    if fold_data is None:
        return rows

    X_train, y_train = fold_data["X_train"], fold_data["y_train"]
    X_comp,  y_comp  = fold_data["X_comp"],  fold_data["y_comp"]
    X_test,  y_test  = fold_data["X_test"],  fold_data["y_test"]
    n_features = fold_data["n_features"]
    pool_rs    = fold_data["pool_rs"]

    for pool_name in POOL_NAMES:
        for variant_name, prune_ratio, ann_overrides in ABLATION_VARIANTS:
            logger.info(
                f"[ABLATION] {dataset_name} seed={seed} fold={fold_idx} | "
                f"pool={pool_name} | variant={variant_name}"
            )
            try:
                comp_preds, test_preds, pool_fit_ms, pool_predict_ms, pool_tune_ms = build_pruned_pool(
                    pool_name, X_train, y_train, X_comp, X_test,
                    n_features, pool_rs, logger=logger,
                    error_metric=ABLATION_ERROR_METRIC,
                    prune_ratio=prune_ratio,
                )
            except Exception as exc:
                logger.error(
                    f"Ablation pool build failed [{pool_name}, {variant_name}]: {exc}"
                )
                continue

            n_pool_models = len(comp_preds)

            for method in ALL_METHODS:
                tune_seed = (
                    pool_rs
                    + _stable_hash(f"{method}|{ABLATION_ERROR_METRIC}|{variant_name}") % 100_000
                )
                try:
                    row, _trials, _ = _tune_and_evaluate_des(
                        method, ABLATION_ERROR_METRIC, X_comp, y_comp, comp_preds,
                        X_test, y_test, test_preds, tune_seed, ann_overrides=ann_overrides,
                    )
                except Exception as exc:
                    logger.error(
                        f"Ablation tuning/eval failed [{pool_name}, {variant_name}, {method}]: {exc}"
                    )
                    continue

                if row is None:
                    continue

                row.update(dict(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    seed=seed,
                    fold=fold_idx,
                    pool=pool_name,
                    variant=variant_name,
                    n_pool_models=n_pool_models,
                    pool_fit_time_ms=pool_fit_ms,
                    pool_predict_time_ms=pool_predict_ms,
                    pool_tune_time_ms=pool_tune_ms,
                ))
                rows.append(row)

    logger.info(f"[ABLATION-DONE] {dataset_name} seed={seed} fold={fold_idx} | rows={len(rows)}")
    return rows


def analyze_ablation(ablation_rows: list) -> pd.DataFrame:
    """Summarize the pruning / ANN ablation: mean MAE and timings per (pool,
    method, variant), plus each non-control variant's % change vs. control.
    Prints one table per pool and writes results/ablation_summary.csv."""
    if not ablation_rows:
        _console("No ablation rows to analyze (ablation_results.parquet is empty).")
        return pd.DataFrame()

    df = pd.DataFrame(ablation_rows)
    df = df[df["error"].isna()].copy() if "error" in df.columns else df
    if df.empty:
        _console("Every ablation row errored out — nothing to summarize.")
        return pd.DataFrame()

    agg = df.groupby(["pool", "method", "variant"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        predict_time_ms_mean=("predict_time_ms", "mean"),
        fit_time_ms_mean=("fit_time_ms", "mean"),
        tune_time_ms_mean=("tune_time_ms", "mean"),
        n_pool_models_mean=("n_pool_models", "mean"),
        n=("mae", "count"),
    ).reset_index()

    control = agg[agg["variant"] == "control"].set_index(["pool", "method"])

    def _pct_vs_control(row, col):
        key = (row["pool"], row["method"])
        if row["variant"] == "control" or key not in control.index:
            return np.nan
        base = control.loc[key, col]
        if pd.isna(base) or base == 0:
            return np.nan
        return (row[col] - base) / base * 100

    agg["mae_pct_vs_control"] = agg.apply(lambda r: _pct_vs_control(r, "mae_mean"), axis=1)
    agg["predict_time_pct_vs_control"] = agg.apply(
        lambda r: _pct_vs_control(r, "predict_time_ms_mean"), axis=1
    )

    _console("\n" + "=" * 78)
    _console("ABLATION STUDY — pool pruning and ANN backend")
    _console(
        f"({df['dataset_name'].nunique()} datasets, seeds={sorted(df['seed'].unique())}, "
        f"folds={sorted(df['fold'].unique())}, metric={ABLATION_ERROR_METRIC})"
    )
    _console("=" * 78)

    for pool in POOL_NAMES:
        sub = agg[agg["pool"] == pool]
        if sub.empty:
            continue
        _console(f"\n[{pool}]")
        show = sub.sort_values(["method", "variant"])[
            ["method", "variant", "mae_mean", "mae_pct_vs_control",
             "n_pool_models_mean", "predict_time_ms_mean", "predict_time_pct_vs_control"]
        ].copy()
        show["mae_mean"] = show["mae_mean"].map(_fmt_metric)
        show["mae_pct_vs_control"] = show["mae_pct_vs_control"].map(_fmt_pct)
        show["n_pool_models_mean"] = show["n_pool_models_mean"].map(
            lambda x: "—" if pd.isna(x) else f"{x:.1f}"
        )
        show["predict_time_ms_mean"] = show["predict_time_ms_mean"].map(
            lambda x: "—" if pd.isna(x) else f"{x:.2f}"
        )
        show["predict_time_pct_vs_control"] = show["predict_time_pct_vs_control"].map(_fmt_pct)
        _console(show.to_string(index=False))

    _console("=" * 78)
    _console(
        "Note: mae_pct_vs_control / predict_time_pct_vs_control are +worse/-better "
        "relative to the control variant (normal PRUNE_RATIO + faiss/hnsw) for "
        "that same (pool, method)."
    )

    out_path = RESULTS_DIR / "ablation_summary.csv"
    agg.to_csv(out_path, index=False)
    main_logger.info(f"Ablation summary written to {out_path}")

    return agg


def run_ablation_study() -> None:
    main_logger.info("=" * 60)
    main_logger.info("Ablation study: pool pruning and ANN backend")
    main_logger.info("=" * 60)

    all_datasets, dataset_work = _load_valid_datasets()
    ablation_dataset_ids = _select_ablation_datasets(all_datasets, dataset_work)
    main_logger.info(
        f"Ablation datasets ({len(ablation_dataset_ids)}): "
        f"{[all_datasets[d][2] for d in ablation_dataset_ids]}"
    )

    ablation_path = RESULTS_DIR / "ablation_results.parquet"
    ablation_rows: list = []
    if ablation_path.exists():
        try:
            ablation_rows = pd.read_parquet(ablation_path).to_dict("records")
            main_logger.info(f"Loaded {len(ablation_rows)} ablation rows from checkpoint.")
        except Exception as e:
            main_logger.warning(f"Could not read ablation checkpoint: {e}")

    completed = {(r["dataset_id"], r["seed"], r["fold"]) for r in ablation_rows}

    tasks: list[tuple] = []
    for did in ablation_dataset_ids:
        for seed in ABLATION_SEEDS:
            for fold in range(ABLATION_N_FOLDS):
                if (did, seed, fold) in completed:
                    continue
                tasks.append((did, seed, fold, *all_datasets[did]))

    main_logger.info(f"{len(tasks)} ablation tasks remaining ({len(completed)} already done).")

    if not tasks:
        main_logger.info("All ablation tasks already completed.")
        analyze_ablation(ablation_rows)
        return

    n_jobs = _choose_n_jobs(datasets={did: all_datasets[did] for did in ablation_dataset_ids})
    main_logger.info(f"Dispatching {len(tasks)} ablation tasks across N_JOBS={n_jobs} workers...")

    gen = Parallel(
        n_jobs=n_jobs, backend="loky", verbose=0, return_as="generator",
    )(
        delayed(_ablation_worker_main)(did, seed, fold, X, y, name, n_num, n_cat)
        for did, seed, fold, X, y, name, n_num, n_cat in tasks
    )

    try:
        for i, bundle in enumerate(gen):
            if bundle:
                ablation_rows.extend(bundle)
            if (i + 1) % 5 == 0:
                _save(ablation_rows, ablation_path)
                main_logger.info(f"Ablation progress: {i + 1}/{len(tasks)} tasks done.")
    except KeyboardInterrupt:
        main_logger.warning("\nCtrl+C received — flushing ablation checkpoint before exiting…")
        _save(ablation_rows, ablation_path)
        raise

    _save(ablation_rows, ablation_path)
    main_logger.info(f"Ablation study complete. {len(ablation_rows)} rows saved to {ablation_path}")
    analyze_ablation(ablation_rows)


# This section only formats and prints already-computed rows; it does not
# change what is computed or what gets saved to the parquet checkpoints.

def _fmt_metric(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.4f}"


def _fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.2f}%"


def _fmt_rank(x: float) -> str:
    return "—" if pd.isna(x) else str(int(x))


def print_dataset_summary(
    dataset_id: int,
    dataset_name: str,
    baseline_rows: list,
    des_rows: list,
) -> None:
    """Print 10 tables (POOL_NAMES × ERROR_METRICS) of per-method results for
    one fully-completed dataset, averaged over all seeds and folds.

    Each table shows, for the baselines plus the DES/OLA methods tuned on
    that table's error metric: the mean ± std of that metric, % improvement
    over the SingleBest baseline, and rank (1 = best). Console-only — does
    not touch the saved results or per-worker log files.
    """
    df_base_all = pd.DataFrame([r for r in baseline_rows if r["dataset_id"] == dataset_id])
    df_des_all = pd.DataFrame([r for r in des_rows if r["dataset_id"] == dataset_id])

    header = f" RESULTS — {dataset_name} (id={dataset_id}) "
    _console("\n" + header.center(78, "="))

    for pool in POOL_NAMES:
        for metric in ERROR_METRICS:
            sub_base_pool = (
                df_base_all[
                    (df_base_all["pool"] == pool) & (df_base_all["tuned_error_metric"] == metric)
                ]
                if not df_base_all.empty else df_base_all
            )
            sub_des = (
                df_des_all[
                    (df_des_all["pool"] == pool) & (df_des_all["tuned_error_metric"] == metric)
                ]
                if not df_des_all.empty else df_des_all
            )

            rows = []
            if not sub_base_pool.empty:
                stats = sub_base_pool.groupby("baseline")[metric].agg(["mean", "std"])
                for name, r in stats.iterrows():
                    rows.append({"Method": name, "mean": r["mean"], "std": r["std"]})
            if not sub_des.empty:
                stats = sub_des.groupby("method")[metric].agg(["mean", "std"])
                for name, r in stats.iterrows():
                    rows.append({"Method": name, "mean": r["mean"], "std": r["std"]})

            _console(f"\n[{pool}] — tuned on {metric.upper()}")
            if not rows:
                _console("  (no completed rows for this pool/metric)")
                continue

            tbl = pd.DataFrame(rows)
            sb_match = tbl.loc[tbl["Method"] == "SingleBest", "mean"]
            sb = float(sb_match.iloc[0]) if len(sb_match) and pd.notna(sb_match.iloc[0]) else np.nan
            tbl["pct_vs_best"] = (sb - tbl["mean"]) / sb * 100 if pd.notna(sb) and sb != 0 else np.nan
            tbl["rank"] = tbl["mean"].rank(method="min")
            tbl = tbl.sort_values("rank", na_position="last")

            display = pd.DataFrame({
                "Method": tbl["Method"],
                metric.upper(): tbl["mean"].map(_fmt_metric),
                "Std": tbl["std"].map(_fmt_metric),
                "% vs SingleBest": tbl["pct_vs_best"].map(_fmt_pct),
                "Rank": tbl["rank"].map(_fmt_rank),
            })
            _console(display.to_string(index=False))

    _console("=" * 78)


def _load_valid_datasets() -> tuple[dict[int, tuple], dict[int, float]]:
    """Loads every dataset in ALL_DATASET_IDS, filters out ones too small for
    MIN_DATASET_SAMPLES or too small for the minimum router k, and returns
    (all_datasets, dataset_work) — dataset_work is the samples×features
    heuristic used both for ETA estimation and for picking "representative"
    datasets for the ablation study.
    """
    all_datasets: dict[int, tuple] = {}
    dataset_work: dict[int, float] = {}

    for dataset_id in ALL_DATASET_IDS:
        try:
            result = load_dataset(dataset_id)
            X_ds, y_ds, _name, _n_num, _n_cat = result
            if len(y_ds) < MIN_DATASET_SAMPLES:
                main_logger.warning(
                    f"Skipping dataset {dataset_id}: "
                    f"only {len(y_ds)} samples (min {MIN_DATASET_SAMPLES})"
                )
                continue
            expected_n_comp = max(10, int(len(y_ds) * 0.9 * 0.25))
            if MIN_K_ROUTER >= expected_n_comp:
                main_logger.warning(
                    f"Skipping dataset {dataset_id}: "
                    f"pool size too small for min k={MIN_K_ROUTER}"
                )
                continue
            all_datasets[dataset_id] = result
            dataset_work[dataset_id] = len(y_ds) * (_n_num + _n_cat)
        except Exception as exc:
            main_logger.error(f"Failed to load dataset {dataset_id}: {exc}")

    main_logger.info(f"Loaded {len(all_datasets)} valid datasets.")
    return all_datasets, dataset_work


def run_benchmark() -> None:
    main_logger.info("=" * 60)
    main_logger.info("Benchmark: DES for Regression — single-phase evaluation")
    main_logger.info(f"RS_BUDGET={RS_BUDGET}  |  N_FOLDS={N_FOLDS}  |  SEEDS={SEEDS}")
    main_logger.info("=" * 60)

    all_datasets, dataset_work = _load_valid_datasets()

    meta_rows, baseline_rows, des_rows, sensitivity_rows, completed_folds = _load_checkpoints()

    tasks: list[tuple] = []
    total_work   = 0.0
    skipped_work = 0.0

    for did in all_datasets:
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                key = (did, seed, fold)
                w = dataset_work[did]
                total_work += w
                if key in completed_folds:
                    skipped_work += w
                    continue
                tasks.append((did, seed, fold, *all_datasets[did]))

    remaining_work = total_work - skipped_work
    main_logger.info(
        f"{len(tasks)} tasks remaining "
        f"({len(completed_folds)} already done via checkpoints)"
    )
    main_logger.info(f"Remaining work (samples × features): {remaining_work:.1e}")

    # Per-dataset fold-completion tracking, used only to know when a dataset
    # is fully done so its 10 summary tables can be printed. Seeded from the
    # checkpoint so datasets finished in a previous run are accounted for.
    EXPECTED_FOLDS_PER_DATASET = len(SEEDS) * N_FOLDS
    dataset_names = {did: all_datasets[did][2] for did in all_datasets}
    dataset_done_count: dict[int, int] = {
        did: sum(1 for s in SEEDS for f in range(N_FOLDS) if (did, s, f) in completed_folds)
        for did in all_datasets
    }
    datasets_printed: set[int] = set()

    for did, done in dataset_done_count.items():
        if done >= EXPECTED_FOLDS_PER_DATASET:
            print_dataset_summary(did, dataset_names[did], baseline_rows, des_rows)
            datasets_printed.add(did)

    if not tasks:
        main_logger.info("All tasks already completed. Exiting.")
        analyze_des_sensitivity(sensitivity_rows)
        return

    n_jobs = _choose_n_jobs(datasets=all_datasets)
    main_logger.info(f"Dispatching {len(tasks)} tasks across N_JOBS={n_jobs} workers...")

    CHECKPOINT_EVERY = 10

    def _flush(force: bool = False) -> None:
        _save(meta_rows,     RESULTS_DIR / "meta_features.parquet")
        _save(baseline_rows, RESULTS_DIR / "baseline_results.parquet")
        _save(des_rows,      RESULTS_DIR / "des_results.parquet")
        _save(sensitivity_rows, RESULTS_DIR / "sensitivity_results.parquet")
        if force:
            main_logger.info("Checkpoints flushed.")

    start_time          = time.time()
    completed           = 0
    completed_work_new  = 0.0
    rate                = 0.0
    alpha               = 0.9

    gen = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=0,
        return_as="generator",
    )(
        delayed(_worker_main)(did, seed, fold, X, y, name, n_num, n_cat)
        for did, seed, fold, X, y, name, n_num, n_cat in tasks
    )

    try:
        for i, bundle in enumerate(gen):
            completed += 1
            task_did = tasks[i][0]

            if bundle is None:
                main_logger.warning(f"Worker bundle {completed} returned None — skipping.")
                continue

            meta_rows.extend(bundle["meta"])
            baseline_rows.extend(bundle["baseline"])
            des_rows.extend(bundle["des"])
            sensitivity_rows.extend(bundle["sensitivity"])

            src = bundle["meta"] or bundle["baseline"] or bundle["des"]
            if src:
                did = src[0]["dataset_id"]
                w   = dataset_work.get(did, 0.0)
                completed_work_new += w

            elapsed = time.time() - start_time
            if completed_work_new > 0 and elapsed > 0:
                current_rate = elapsed / completed_work_new
                rate = current_rate if completed == 1 \
                       else alpha * rate + (1 - alpha) * current_rate
                eta_s = rate * (remaining_work - completed_work_new)
                d = int(eta_s // 86400)
                h = int((eta_s % 86400) // 3600)
                m = int((eta_s % 3600) // 60)
                s = int(eta_s % 60)
                eta_msg = (
                    f"ETA: {d}d {h:02d}:{m:02d}:{s:02d}  |  "
                    f"rate: {rate*1e6:.2f} s/M-work  "
                    f"({completed}/{len(tasks)} new tasks)"
                )
                main_logger.info(eta_msg)
                _console(eta_msg)

            if completed % CHECKPOINT_EVERY == 0:
                _flush()

            dataset_done_count[task_did] = dataset_done_count.get(task_did, 0) + 1
            if (
                dataset_done_count[task_did] >= EXPECTED_FOLDS_PER_DATASET
                and task_did not in datasets_printed
            ):
                print_dataset_summary(task_did, dataset_names[task_did], baseline_rows, des_rows)
                datasets_printed.add(task_did)

    except KeyboardInterrupt:
        warn_msg = "\nCtrl+C received — flushing completed results before exiting…"
        main_logger.warning(warn_msg)
        _console(warn_msg)
        _flush(force=True)
        main_logger.warning(
            f"Saved {len(meta_rows)} meta, {len(baseline_rows)} baseline, "
            f"{len(des_rows)} DES rows from {completed} completed tasks.\n"
            "Re-run to resume from checkpoints."
        )
        raise SystemExit(1)

    _flush(force=True)

    main_logger.info("=" * 60)
    main_logger.info(
        f"Benchmark complete.  Results saved to {RESULTS_DIR}/\n"
        f"  meta_features.parquet    — {len(meta_rows)} rows\n"
        f"  baseline_results.parquet — {len(baseline_rows)} rows\n"
        f"  des_results.parquet      — {len(des_rows)} rows\n"
        f"  sensitivity_results.parquet — {len(sensitivity_rows)} rows"
    )
    main_logger.info("=" * 60)
    _console(f"\nBenchmark complete. Results saved to {RESULTS_DIR}/")

    analyze_des_sensitivity(sensitivity_rows)


if __name__ == "__main__":
    if "--sensitivity-only" in sys.argv:
        # Re-run just the hyperparameter sensitivity analysis against
        # whatever's already in results/sensitivity_results.parquet, without
        # touching or re-running the rest of the benchmark.
        _, _, _, _sensitivity_rows, _ = _load_checkpoints()
        analyze_des_sensitivity(_sensitivity_rows)
    elif "--ablation" in sys.argv:
        # Runs ONLY the pool-pruning / ANN-backend ablation (a handful of
        # representative datasets, see ABLATION_DATASET_COUNT/SEEDS/N_FOLDS)
        # — separate from, and much cheaper than, the full benchmark.
        run_ablation_study()
    else:
        run_benchmark()