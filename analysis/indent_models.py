"""
indent_models.py
================
Regression models for the 3D random-indentation data (Table S1, S2 and Fig. 5).

Inputs : dL1..dL5 (per-indentation median over the hold window, uH)
Targets: location model -> x, y, z (mm)      force model -> F (g; reported in N)

Models (scikit-learn), each wrapped in TransformedTargetRegressor(StandardScaler) so
that every target is standardised:
    "DT"  : DecisionTreeRegressor
    "RF"  : RandomForestRegressor (300 trees)
    "MLP" : StandardScaler on dL + MLPRegressor (ReLU, Adam, lr 1e-3, early stopping on
            15 % of the training data, 50 epochs patience, at most 3000 epochs)

Protocol (identical for every model and target set):
    1) split the indentations once into train (80 %) and test (20 %), fixed seed
    2) GridSearchCV over DEFAULT_GRIDS with 5-fold GroupKFold on the training set,
       scored by the normalized MAE (MAE / std of each target, averaged over targets)
    3) refit the selected hyperparameters on the whole training set and evaluate
       once on the test set
"""

import json

import numpy as np
from joblib import dump
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

FEATURE_COLS = ["dL1", "dL2", "dL3", "dL4", "dL5"]
TARGET_SETS = {"xyz": ["x", "y", "z"], "F": ["F"]}
UNITS = {"x": "mm", "y": "mm", "z": "mm", "F": "g"}
MODEL_NAMES = ("DT", "RF", "MLP")
G_TO_N = 9.80665e-3

DEFAULT_GRIDS = {
    "DT": {"regressor__max_depth": [6, 8, 10, 12, 16, None],
           "regressor__min_samples_leaf": [1, 2, 5, 10]},
    "RF": {"regressor__max_depth": [None, 20],
           "regressor__min_samples_leaf": [1, 2, 5],
           "regressor__max_features": [1.0, "sqrt"]},
    "MLP": {"regressor__mlp__hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 128),
                                                   (256, 256), (128, 256)],
            "regressor__mlp__alpha": [1e-4, 1e-3]},
}


def build_estimator(name, seed=42):
    """Model wrapped so the targets are standardised (inverse-transformed on predict)."""
    if name == "DT":
        reg = DecisionTreeRegressor(random_state=seed)
    elif name == "RF":
        reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
    elif name == "MLP":
        reg = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(solver="adam", learning_rate_init=1e-3, max_iter=3000,
                                 early_stopping=True, validation_fraction=0.15,
                                 n_iter_no_change=50, random_state=seed)),
        ])
    else:
        raise ValueError(f"unknown model {name}")
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())


def normalized_mae(y_true, y_pred):
    """Mean over targets of MAE / std(target), so every target counts equally."""
    yt = np.asarray(y_true, float).reshape(len(y_true), -1)
    yp = np.asarray(y_pred, float).reshape(len(y_pred), -1)
    std = yt.std(axis=0)
    std[std == 0] = 1.0
    return float(np.mean(np.abs(yt - yp).mean(axis=0) / std))


NMAE_SCORER = make_scorer(normalized_mae, greater_is_better=False)


def _fit_y(Y):
    return Y.ravel() if Y.shape[1] == 1 else Y


def predict(model, X):
    return np.asarray(model.predict(X)).reshape(len(X), -1)


def make_dataset(pts, targets):
    """pts: one row per indentation (extract_indent_data.load_dataset).
    Returns X (N, 5), Y (N, T), point ids (N,) and depth = -z (N,)."""
    return (pts[FEATURE_COLS].to_numpy(float), pts[targets].to_numpy(float),
            pts["point_id"].to_numpy(), -pts["z"].to_numpy(float))


def split_points(point_ids, test_size=0.2, seed=42):
    """Fixed-seed split of the indentations. Returns (train_ids, test_ids)."""
    ids = np.unique(point_ids)
    perm = np.random.default_rng(seed).permutation(ids)
    n_test = int(round(test_size * len(ids)))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def tune_and_fit(name, X, Y, groups, cv=5, seed=42, n_jobs=-1):
    """Grid search with GroupKFold on (X, Y), then refit the best setting on all of it.
    Returns (fitted model, selected hyperparameters)."""
    search = GridSearchCV(build_estimator(name, seed), DEFAULT_GRIDS[name],
                          scoring=NMAE_SCORER, cv=GroupKFold(n_splits=cv), n_jobs=n_jobs,
                          refit=False)
    search.fit(X, _fit_y(Y), groups=groups)
    model = build_estimator(name, seed).set_params(**search.best_params_)
    model.fit(X, _fit_y(Y))
    return model, search.best_params_


def save_model(model, path, name, set_name, params, test_mae):
    """joblib model + <path>_meta.json with the settings and test MAE."""
    dump(model, path)
    meta = {"model": name, "target_set": set_name, "targets": TARGET_SETS[set_name],
            "units": {t: UNITS[t] for t in TARGET_SETS[set_name]},
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
            "features": FEATURE_COLS,
            "input": "per-point median of the hold window (20 samples at 10 Hz)",
            "dL_definition": "L - median of the hover baseline window of the same point",
            "F_definition": "force_g - median hover baseline force of the same point (g)",
            "z_definition": "z_rep - z_surface (negative = into the material)",
            "test_mae": test_mae}
    with open(str(path).rsplit(".", 1)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
