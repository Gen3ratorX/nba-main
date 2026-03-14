# Plan: Ensemble Model + Hyperparameter Tuning

## Current State
- Single XGBoost model with sigmoid calibration + reliability alpha mapping
- Hardcoded hyperparams: `n_estimators=100, max_depth=3, lr=0.1, min_child_weight=10, subsample=0.7, colsample_bytree=0.7, gamma=0.3, reg_alpha=0.5, reg_lambda=3.0`
- 33 features, all backtest gates passing
- `model_utils.py` has an unused `VotingClassifier` ensemble (RF+GB+XGB+LR) — never wired into `main.py` or `strict_backtest.py`
- Neither `lightgbm` nor `optuna` are installed

## What We'll Build

### 1. Install Dependencies
- `pip install lightgbm optuna` and add to `requirements.txt`

### 2. Optuna Hyperparameter Tuning (`hyperopt.py` — new file)
- `tune_xgboost(X, y, n_trials=100)` — Optuna study optimizing XGBoost params via TimeSeriesSplit CV
  - Search space: `n_estimators [50-400]`, `max_depth [2-8]`, `learning_rate [0.01-0.3]`, `min_child_weight [3-20]`, `subsample [0.5-0.9]`, `colsample_bytree [0.5-0.9]`, `gamma [0-1]`, `reg_alpha [0-2]`, `reg_lambda [1-5]`
  - Objective: minimize mean Brier score across 5 TimeSeriesSplit folds (better than accuracy for calibrated models)
  - Returns best params dict
- `tune_lightgbm(X, y, n_trials=100)` — same approach for LightGBM
  - Search space: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- `tune_ensemble_weights(X, y, models, n_trials=50)` — optimize soft-voting weights
- Results cached to `models/optuna_best_params.json` so tuning doesn't re-run every pipeline execution

### 3. Stacking Ensemble (`ensemble.py` — new file)
- `StackedEnsemble` class wrapping:
  - **Level-0 (base learners):** XGBoost, LightGBM, LogisticRegression (with StandardScaler)
  - **Level-1 (meta-learner):** LogisticRegression trained on out-of-fold predictions from level-0
  - Uses `cross_val_predict` with TimeSeriesSplit to generate OOF predictions (no leakage)
- `fit(X_train, y_train)` — fits all base learners + meta-learner
- `predict_proba(X)` — returns meta-learner probabilities
- Compatible with `CalibratedClassifierCV` wrapping (has `predict_proba`, `predict`, `fit`)
- Falls back gracefully if LightGBM unavailable (XGB + LR only)

### 4. Wire Into `main.py` `_train_moneyline_model()`
- Load cached Optuna params (or use defaults if no tuning has been run)
- Replace single `XGBClassifier` pipeline with `StackedEnsemble`
- Keep existing calibration flow (CalibratedClassifierCV + reliability alpha)
- Add `--tune` CLI flag to trigger Optuna tuning (slow, ~5-10 min)

### 5. Wire Into `strict_backtest.py`
- Replace `_build_base_pipeline()` to use `StackedEnsemble`
- Same calibration and reliability alpha flow — no changes needed there

### 6. Run Backtest & Validate
- Run `python main.py --strict-backtest --days-back 180`
- All acceptance gates must still pass

## Files Changed
| File | Action |
|---|---|
| `requirements.txt` | Add `lightgbm`, `optuna` |
| `hyperopt.py` | **New** — Optuna tuning functions |
| `ensemble.py` | **New** — StackedEnsemble class |
| `main.py` | Update `_train_moneyline_model()` to use ensemble + optional tuning |
| `strict_backtest.py` | Update `_build_base_pipeline()` to use ensemble |
| `config.py` | Add `ENSEMBLE_CONFIG` and `OPTUNA_CONFIG` sections |

## Risk Mitigation
- If ensemble performs worse than single XGB in backtest, we keep XGB as the sole base learner and still benefit from Optuna-tuned hyperparams
- Fallback: if LightGBM fails to install, ensemble degrades to XGB + LR (still better than single XGB)
- Optuna results are cached — tuning only runs on explicit `--tune` flag, not every pipeline run
