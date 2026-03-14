"""
ensemble.py - Stacking ensemble for NBA moneyline prediction.
Level-0: XGBoost + LightGBM + LogisticRegression (diverse learners)
Level-1: LogisticRegression meta-learner trained on out-of-fold predictions
Compatible with sklearn's CalibratedClassifierCV and Pipeline conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nbautils import log_info, log_warning

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

from hyperopt import get_best_params, _default_xgb_params, _default_lgbm_params


class StackedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Two-level stacking ensemble.

    Level-0 base learners produce out-of-fold (OOF) predictions via
    TimeSeriesSplit — no data leakage. The level-1 meta-learner is trained
    on these OOF predictions.

    At inference time, each base learner predicts on new data, and the
    meta-learner combines them.

    Exposes fit / predict / predict_proba so it works seamlessly with
    CalibratedClassifierCV.
    """

    def __init__(
        self,
        xgb_params: dict | None = None,
        lgbm_params: dict | None = None,
        n_cv_folds: int = 3,
    ):
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params
        self.n_cv_folds = n_cv_folds

        # Will be set during fit
        self.base_learners_: list = []
        self.meta_learner_ = None
        self.classes_ = np.array([0, 1])

    def _build_base_learners(self) -> list:
        """Build list of (name, pipeline) for each base learner."""
        learners = []

        # XGBoost
        if XGBOOST_AVAILABLE:
            params = self.xgb_params or get_best_params("xgboost") or _default_xgb_params()
            xgb = XGBClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
            learners.append(("xgb", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", xgb),
            ])))

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            params = self.lgbm_params or get_best_params("lightgbm") or _default_lgbm_params()
            lgbm = LGBMClassifier(
                **params,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            learners.append(("lgbm", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", lgbm),
            ])))

        # Logistic Regression (always available, provides diversity)
        lr = LogisticRegression(
            C=0.5,
            max_iter=2000,
            random_state=42,
        )
        learners.append(("lr", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", lr),
        ])))

        if not learners:
            raise RuntimeError("No base learners available. Install xgboost or lightgbm.")

        return learners

    def fit(self, X, y):
        """Fit the stacking ensemble using OOF predictions."""
        X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
        y_arr = np.asarray(y).ravel()

        learner_specs = self._build_base_learners()
        n_learners = len(learner_specs)
        n_samples = len(y_arr)

        # Generate OOF predictions for each base learner
        oof_preds = np.zeros((n_samples, n_learners))
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)

        names = []
        for j, (name, pipeline) in enumerate(learner_specs):
            names.append(name)
            for train_idx, val_idx in tscv.split(X_arr):
                X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                y_tr = y_arr[train_idx]
                pipeline.fit(X_tr, y_tr)
                oof_preds[val_idx, j] = pipeline.predict_proba(X_val)[:, 1]

        # Train each base learner on full data for inference
        self.base_learners_ = []
        for name, pipeline in learner_specs:
            pipeline.fit(X_arr, y_arr)
            self.base_learners_.append((name, pipeline))

        # Only use rows that have OOF predictions (skip first fold's training rows)
        has_oof = np.any(oof_preds != 0, axis=1)
        oof_X = oof_preds[has_oof]
        oof_y = y_arr[has_oof]

        # Train meta-learner on OOF predictions
        self.meta_learner_ = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )
        self.meta_learner_.fit(oof_X, oof_y)

        learner_names = ", ".join(names)
        log_info(f"🏗️  Stacking ensemble fitted: [{learner_names}] → meta-LR")
        log_info(f"   OOF training rows for meta-learner: {len(oof_y)}")

        return self

    def predict_proba(self, X):
        """Get probability predictions from the stacking ensemble."""
        X_arr = np.asarray(X) if not isinstance(X, np.ndarray) else X

        # Get base learner predictions
        base_preds = np.column_stack([
            pipeline.predict_proba(X_arr)[:, 1]
            for _, pipeline in self.base_learners_
        ])

        # Meta-learner combines them
        meta_probs = self.meta_learner_.predict_proba(base_preds)
        return meta_probs

    def predict(self, X):
        """Get class predictions."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def get_base_learner_names(self) -> list:
        """Return names of fitted base learners."""
        return [name for name, _ in self.base_learners_]


def build_ensemble_pipeline(
    xgb_params: dict | None = None,
    lgbm_params: dict | None = None,
) -> StackedEnsemble:
    """
    Convenience factory. Returns a StackedEnsemble ready to fit.
    The ensemble handles its own imputation and scaling internally,
    so callers should pass raw feature DataFrames.
    """
    return StackedEnsemble(
        xgb_params=xgb_params,
        lgbm_params=lgbm_params,
    )
