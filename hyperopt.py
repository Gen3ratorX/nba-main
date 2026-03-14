"""
hyperopt.py - Optuna hyperparameter tuning for NBA prediction models.
Tunes XGBoost, LightGBM, and ensemble weights via Bayesian optimization.
Results cached to disk so tuning doesn't re-run every pipeline execution.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nbautils import log_info, log_warning

# Suppress Optuna's verbose logging
logging.getLogger("optuna").setLevel(logging.WARNING)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


CACHE_PATH = Path("models/optuna_best_params.json")


def _load_cache() -> Dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(data: Dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2))


def _cv_brier(pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Mean Brier score across TimeSeriesSplit folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        p = pipeline.predict_proba(X_val)[:, 1]
        scores.append(brier_score_loss(y_val, p))
    return float(np.mean(scores))


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 80,
    force: bool = False,
) -> Dict:
    """Tune XGBoost hyperparameters with Optuna. Returns best params dict."""
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        log_warning("Optuna or XGBoost not available; returning defaults.")
        return _default_xgb_params()

    cache = _load_cache()
    if "xgboost" in cache and not force:
        log_info("Using cached XGBoost hyperparameters from previous tuning run.")
        return cache["xgboost"]

    log_info(f"⏳ Tuning XGBoost with {n_trials} Optuna trials...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        }
        clf = XGBClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
        return _cv_brier(pipe, X, y)

    study = optuna.create_study(direction="minimize", study_name="xgboost_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    log_info(f"✅ XGBoost tuning complete. Best Brier: {study.best_value:.4f}")
    log_info(f"   Best params: {best}")

    cache["xgboost"] = best
    _save_cache(cache)
    return best


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 80,
    force: bool = False,
) -> Dict:
    """Tune LightGBM hyperparameters with Optuna. Returns best params dict."""
    if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
        log_warning("Optuna or LightGBM not available; returning defaults.")
        return _default_lgbm_params()

    cache = _load_cache()
    if "lightgbm" in cache and not force:
        log_info("Using cached LightGBM hyperparameters from previous tuning run.")
        return cache["lightgbm"]

    log_info(f"⏳ Tuning LightGBM with {n_trials} Optuna trials...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        }
        clf = LGBMClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
        return _cv_brier(pipe, X, y)

    study = optuna.create_study(direction="minimize", study_name="lightgbm_tune")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    log_info(f"✅ LightGBM tuning complete. Best Brier: {study.best_value:.4f}")
    log_info(f"   Best params: {best}")

    cache["lightgbm"] = best
    _save_cache(cache)
    return best


def tune_all(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 80,
    force: bool = False,
) -> Dict:
    """Run all tuning and return combined results."""
    results = {}
    results["xgboost"] = tune_xgboost(X, y, n_trials=n_trials, force=force)
    results["lightgbm"] = tune_lightgbm(X, y, n_trials=n_trials, force=force)
    return results


def get_best_params(model_name: str) -> Optional[Dict]:
    """Retrieve cached best params for a model (xgboost or lightgbm)."""
    cache = _load_cache()
    return cache.get(model_name)


def _default_xgb_params() -> Dict:
    return {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "gamma": 0.3,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
    }


def _default_lgbm_params() -> Dict:
    return {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
    }


if __name__ == "__main__":
    import argparse
    from data_processor import NBADataProcessor

    parser = argparse.ArgumentParser(description="Tune model hyperparameters with Optuna")
    parser.add_argument("--trials", type=int, default=80, help="Number of Optuna trials per model")
    parser.add_argument("--force", action="store_true", help="Force re-tune even if cache exists")
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "all"], default="all")
    args = parser.parse_args()

    from strict_backtest import CANONICAL_FEATURES

    processor = NBADataProcessor()
    df = processor.process_all_data()
    df = df.dropna(subset=["home_won"]).iloc[15:].copy()

    for f in CANONICAL_FEATURES:
        if f not in df.columns:
            df[f] = 0

    X = df[CANONICAL_FEATURES].copy()
    y = df["home_won"].astype(int).copy()

    log_info(f"Tuning on {len(X)} games with {len(CANONICAL_FEATURES)} features")

    if args.model == "xgboost":
        tune_xgboost(X, y, n_trials=args.trials, force=args.force)
    elif args.model == "lightgbm":
        tune_lightgbm(X, y, n_trials=args.trials, force=args.force)
    else:
        tune_all(X, y, n_trials=args.trials, force=args.force)
