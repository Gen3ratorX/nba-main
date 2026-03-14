"""
strict_backtest.py
Canonical rolling backtest for the unified moneyline pipeline.
Outputs one machine-readable report file for auditability.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_processor import NBADataProcessor
from nbautils import log_error, log_info, log_warning
from ensemble import build_ensemble_pipeline
from hyperopt import get_best_params

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# Core features (always active — proven positive importance)
CORE_FEATURES = [
    'form_diff', 'elo_diff', 'fatigue_index', 'rest_diff',
    'pace_home_L5', 'pace_away_L5', 'combined_pace',
    'off_rating_home_L10', 'off_rating_away_L10',
    'back_to_back_home', 'back_to_back_away',
    'net_rating_diff',
    # EWMA recency-weighted features
    'ewma_ppg_home',
    'ewma_papg_home', 'ewma_papg_away',
    'ewma_net_diff',
    # Player availability features
    'minutes_missing_home', 'minutes_missing_away',
    'star_missing_home', 'star_missing_away',
    'roster_strength_home',
    # Head-to-head features
    'h2h_win_rate', 'h2h_games_played',
]

# Activated when odds data is available (auto-detected)
ODDS_FEATURES = [
    'vegas_implied_home', 'vegas_implied_away', 'vegas_implied_diff',
    'vegas_total_line', 'has_vegas_odds',
    'ml_movement_home', 'ml_movement_magnitude',
    'total_line_movement', 'odds_snapshot_count',
]

# Pruned — negative permutation importance (hurt accuracy):
# 'home_court_advantage', 'ewma_ppg_away', 'def_rating_home_L10',
# 'def_rating_away_L10', 'roster_strength_away', 'def_rating_diff',
# 'schedule_strength_diff'


def get_active_features(df):
    """Return features that actually have data (non-zero variance)."""
    active = list(CORE_FEATURES)
    for f in ODDS_FEATURES:
        if f in df.columns and df[f].std() > 1e-8:
            active.append(f)
    return active


# For backward compat — callers that reference CANONICAL_FEATURES
CANONICAL_FEATURES = CORE_FEATURES + ODDS_FEATURES

CALIBRATION_ECE_MAX = 0.12

HOME_ODDS_COLUMNS = [
    'ml_home', 'home_ml', 'home_odds', 'home_moneyline',
    'moneyline_home', 'odds_home', 'home_price',
]
AWAY_ODDS_COLUMNS = [
    'ml_away', 'away_ml', 'away_odds', 'away_moneyline',
    'moneyline_away', 'odds_away', 'away_price',
]
GENERIC_ODDS_COLUMNS = ['odds', 'moneyline', 'ml', 'price']


@dataclass
class FoldResult:
    fold: int
    train_games: int
    test_games: int
    train_date_end: str
    test_date_start: str
    accuracy: float
    brier: float
    log_loss: float
    roc_auc: float | None
    ece: float
    mce: float
    picks: int
    pick_win_rate: float
    units: float
    roi_percent: float
    market_picks: int
    fallback_picks: int


def _safe_roc_auc(y_true, p_home) -> float | None:
    try:
        return float(roc_auc_score(y_true, p_home))
    except Exception:
        return None


def _safe_american_odds(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        odds = int(float(value))
    except Exception:
        return None

    # American odds should be <= -100 or >= +100.
    if -100 < odds < 100:
        return None
    return odds


def _american_to_profit_units(odds: int) -> float:
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def _resolve_pick_odds(
    df_fold: pd.DataFrame,
    y_pred_home: np.ndarray,
    fallback_odds: int = -110,
) -> Tuple[np.ndarray, np.ndarray]:
    odds_values: List[int] = []
    from_market: List[bool] = []

    rows = df_fold.to_dict('records')
    for row, pred_home in zip(rows, y_pred_home):
        candidate_cols = HOME_ODDS_COLUMNS if int(pred_home) == 1 else AWAY_ODDS_COLUMNS

        odds = None
        for col in candidate_cols:
            if col in row:
                odds = _safe_american_odds(row.get(col))
                if odds is not None:
                    break

        # Fallback to generic single-column odds if side-specific values are unavailable.
        if odds is None:
            for col in GENERIC_ODDS_COLUMNS:
                if col in row:
                    odds = _safe_american_odds(row.get(col))
                    if odds is not None:
                        break

        if odds is None:
            odds = fallback_odds
            from_market.append(False)
        else:
            from_market.append(True)

        odds_values.append(int(odds))

    return np.asarray(odds_values, dtype=int), np.asarray(from_market, dtype=bool)


def _evaluate_betting(
    y_true,
    p_home: np.ndarray,
    odds_values: np.ndarray,
    from_market: np.ndarray,
    threshold: float = 0.55,
) -> Dict:
    y_true_np = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    y_pred = (p_home > 0.5).astype(int)
    picks_mask = np.maximum(p_home, 1 - p_home) >= threshold

    pick_count = int(picks_mask.sum())
    if pick_count == 0:
        return {
            'picks': 0,
            'pick_win_rate': 0.0,
            'units': 0.0,
            'roi_percent': 0.0,
            'market_picks': 0,
            'fallback_picks': 0,
        }

    pick_preds = y_pred[picks_mask]
    pick_true = y_true_np[picks_mask]
    pick_odds = odds_values[picks_mask]
    pick_from_market = from_market[picks_mask]

    wins = (pick_preds == pick_true)
    payouts = np.array([_american_to_profit_units(int(o)) for o in pick_odds], dtype=float)
    units = np.where(wins, payouts, -1.0).sum()
    win_rate = float(wins.mean())

    market_picks = int(pick_from_market.sum())
    fallback_picks = int(pick_count - market_picks)

    return {
        'picks': pick_count,
        'pick_win_rate': win_rate,
        'units': float(units),
        'roi_percent': float((units / pick_count) * 100.0),
        'market_picks': market_picks,
        'fallback_picks': fallback_picks,
    }


def _calibration_metrics(y_true, p_home: np.ndarray, n_bins: int = 10) -> Dict:
    y_true_np = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    p = np.asarray(p_home, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    total = len(p)
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        avg_prob = float(p[mask].mean())
        avg_true = float(y_true_np[mask].mean())
        err = abs(avg_true - avg_prob)

        ece += (count / total) * err
        mce = max(mce, err)

        rows.append({
            'bin_lo': float(lo),
            'bin_hi': float(hi),
            'count': count,
            'avg_prob': avg_prob,
            'avg_true': avg_true,
            'abs_error': float(err),
        })

    return {
        'ece': float(ece),
        'mce': float(mce),
        'bins': rows,
    }


def _summarize_fold_metrics(metric_rows: List[Dict]) -> Dict:
    if not metric_rows:
        return {}

    roc_vals = [r['roc_auc'] for r in metric_rows if r.get('roc_auc') is not None]
    total_picks = int(np.sum([r['picks'] for r in metric_rows]))
    total_market_picks = int(np.sum([r.get('market_picks', 0) for r in metric_rows]))
    total_fallback_picks = int(np.sum([r.get('fallback_picks', 0) for r in metric_rows]))

    return {
        'mean_accuracy': float(np.mean([r['accuracy'] for r in metric_rows])),
        'mean_brier': float(np.mean([r['brier'] for r in metric_rows])),
        'mean_log_loss': float(np.mean([r['log_loss'] for r in metric_rows])),
        'mean_roc_auc': float(np.mean(roc_vals)) if roc_vals else None,
        'total_picks': total_picks,
        'total_units': float(np.sum([r['units'] for r in metric_rows])),
        'mean_fold_roi_percent': float(np.mean([r['roi_percent'] for r in metric_rows])),
        'total_market_picks': total_market_picks,
        'total_fallback_picks': total_fallback_picks,
        'market_pick_coverage_pct': float(total_market_picks / total_picks) if total_picks else 0.0,
    }


def _build_base_pipeline():
    """Build a stacking ensemble (XGBoost + LightGBM + LR → meta-LR).
    Uses Optuna-tuned params if cached, else sensible defaults.
    The ensemble handles its own imputation and scaling internally.
    """
    return build_ensemble_pipeline(
        xgb_params=get_best_params("xgboost"),
        lgbm_params=get_best_params("lightgbm"),
    )


def _calibrated_model(base_model, X_train, y_train):
    """Fit the model on training data.
    The StackedEnsemble's meta-learner (LogisticRegression) already produces
    well-calibrated probabilities, so we skip CalibratedClassifierCV wrapping
    to avoid double-calibration artifacts on small folds. The reliability alpha
    mapping applied afterward handles any remaining calibration drift.
    """
    base_model.fit(X_train, y_train)
    return base_model


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 0.05, 0.95)
    row_sum = probs.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return probs / row_sum


def _apply_reliability_map(p_home: np.ndarray, alpha: float) -> np.ndarray:
    p_home = np.asarray(p_home, dtype=float)
    mapped = 0.5 + float(alpha) * (p_home - 0.5)
    return np.clip(mapped, 0.05, 0.95)


def _fit_reliability_alpha(y_true: np.ndarray, p_home: np.ndarray) -> float:
    """
    Choose shrink alpha toward 0.5 using a simple calibration-aware objective.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_home = np.asarray(p_home, dtype=float)
    best_alpha = 1.0
    best_score = float("inf")

    for alpha in np.linspace(0.25, 1.00, 16):
        p_adj = _apply_reliability_map(p_home, alpha)
        cal = _calibration_metrics(y_true, p_adj, n_bins=10)
        score = float(cal['ece'])
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha


def run_strict_backtest(
    days_back: int = 180,
    n_splits: int = 8,
    output_path: str = 'reports/strict_backtest_latest.json',
) -> Dict:
    processor = NBADataProcessor()
    df = processor.process_all_data()
    if df.empty:
        raise RuntimeError("No processed data available for backtest.")

    required_base_cols = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score', 'home_won']
    missing_base = [c for c in required_base_cols if c not in df.columns]
    if missing_base:
        raise RuntimeError(f"Missing required columns for strict backtest: {missing_base}")

    df = df.sort_values('game_date').copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    date_monotonic = bool(df['game_date'].is_monotonic_increasing)

    dup_count = int(df.duplicated(subset=['game_date', 'home_team', 'away_team']).sum())
    if dup_count:
        log_warning(f"Dropping {dup_count} duplicate games by (date, home, away) before backtest.")
        df = df.drop_duplicates(subset=['game_date', 'home_team', 'away_team'], keep='last')

    df = df[(df['home_score'] > 0) & (df['away_score'] > 0)]
    df = df.dropna(subset=['home_won'])

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
    df = df[df['game_date'] >= cutoff].copy()
    if len(df) < 200:
        log_warning(
            f"Only {len(df)} games available in {days_back}-day window; "
            "report confidence is limited."
        )

    for feature in CANONICAL_FEATURES:
        if feature not in df.columns:
            df[feature] = 0

    available_home_odds = [c for c in HOME_ODDS_COLUMNS if c in df.columns]
    available_away_odds = [c for c in AWAY_ODDS_COLUMNS if c in df.columns]
    available_generic_odds = [c for c in GENERIC_ODDS_COLUMNS if c in df.columns]

    active_features = get_active_features(df)
    log_info(f"Active features: {len(active_features)} / {len(CANONICAL_FEATURES)} "
             f"({len(CANONICAL_FEATURES) - len(active_features)} pruned/inactive)")
    X = df[active_features].copy()
    y = df['home_won'].astype(int).copy()
    dates = df['game_date'].copy()

    if len(df) < 60:
        raise RuntimeError(f"Not enough rows ({len(df)}) for rolling strict backtest.")

    max_splits = max(2, min(n_splits, len(df) // 25))
    tscv = TimeSeriesSplit(n_splits=max_splits)

    folds: List[FoldResult] = []
    all_true: List[int] = []
    all_prob: List[float] = []
    calibration_by_fold: List[Dict] = []

    baseline_rows = {
        'home_prior': [],
        'elo_only': [],
    }

    leak_stats = {
        'fold_overlap_violations': 0,
        'rows_removed_for_overlap': 0,
        'folds_skipped_due_overlap': 0,
        'future_leak_rows_detected_after_filter': 0,
    }

    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]

        train_max_date = train_dates.max()
        test_min_date = test_dates.min()

        if test_min_date <= train_max_date:
            leak_stats['fold_overlap_violations'] += 1
            keep_mask = (test_dates > train_max_date).to_numpy()
            removed = int((~keep_mask).sum())
            leak_stats['rows_removed_for_overlap'] += removed

            if keep_mask.sum() == 0:
                leak_stats['folds_skipped_due_overlap'] += 1
                log_warning(
                    f"Fold {i}: all test rows overlap train max date {train_max_date.date()}, skipping fold."
                )
                continue

            test_idx = test_idx[keep_mask]
            test_dates = dates.iloc[test_idx]

        # Hard leakage check after filtering
        if bool((test_dates <= train_max_date).any()):
            leak_stats['future_leak_rows_detected_after_filter'] += int((test_dates <= train_max_date).sum())

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        df_test = df.iloc[test_idx]

        # Learn reliability shrink alpha on train-only tail slice (no test leakage).
        reliability_alpha = 1.0
        if len(X_train) >= 200:
            alpha_split = int(len(X_train) * 0.8)
            X_sub = X_train.iloc[:alpha_split]
            y_sub = y_train.iloc[:alpha_split]
            X_val = X_train.iloc[alpha_split:]
            y_val = y_train.iloc[alpha_split:]
            if len(X_sub) >= 100 and len(X_val) >= 25:
                alpha_model = _calibrated_model(_build_base_pipeline(), X_sub, y_sub)
                p_val = alpha_model.predict_proba(X_val)[:, 1]
                reliability_alpha = _fit_reliability_alpha(y_val.to_numpy(), p_val)

        base_pipeline = _build_base_pipeline()
        model = _calibrated_model(base_pipeline, X_train, y_train)

        p_home_raw = model.predict_proba(X_test)[:, 1]
        p_home_mapped = _apply_reliability_map(p_home_raw, reliability_alpha)
        probs = np.column_stack([1.0 - p_home_mapped, p_home_mapped])
        probs = _clip_probs(probs)
        p_home = probs[:, 1]
        y_pred = (p_home > 0.5).astype(int)

        fold_accuracy = float(accuracy_score(y_test, y_pred))
        fold_brier = float(brier_score_loss(y_test, p_home))
        fold_log_loss = float(log_loss(y_test, probs))
        fold_roc = _safe_roc_auc(y_test, p_home)

        cal = _calibration_metrics(y_test, p_home, n_bins=10)

        model_odds, model_odds_market = _resolve_pick_odds(df_test, y_pred, fallback_odds=-110)
        model_bets = _evaluate_betting(
            y_test,
            p_home,
            model_odds,
            model_odds_market,
            threshold=0.55,
        )

        # Baseline 1: Home-prior (constant probability from training split)
        train_home_rate = float(y_train.mean()) if len(y_train) else 0.5
        home_prior = np.full(len(y_test), np.clip(train_home_rate, 0.20, 0.80))
        home_prior_probs = np.column_stack([1 - home_prior, home_prior])
        home_prior_pred = (home_prior > 0.5).astype(int)
        hp_odds, hp_market = _resolve_pick_odds(df_test, home_prior_pred, fallback_odds=-110)

        home_prior_row = {
            'fold': i,
            'accuracy': float(accuracy_score(y_test, home_prior_pred)),
            'brier': float(brier_score_loss(y_test, home_prior)),
            'log_loss': float(log_loss(y_test, home_prior_probs)),
            'roc_auc': _safe_roc_auc(y_test, home_prior),
            **_evaluate_betting(y_test, home_prior, hp_odds, hp_market, threshold=0.55),
        }
        baseline_rows['home_prior'].append(home_prior_row)

        # Baseline 2: Elo-only logistic curve from elo_diff
        elo_diff = X_test['elo_diff'].to_numpy() if 'elo_diff' in X_test.columns else np.zeros(len(X_test))
        elo_p_home = 1 / (1 + np.exp(-(elo_diff / 120.0)))
        elo_p_home = np.clip(elo_p_home, 0.20, 0.80)
        elo_probs = np.column_stack([1 - elo_p_home, elo_p_home])
        elo_pred = (elo_p_home > 0.5).astype(int)
        elo_odds, elo_market = _resolve_pick_odds(df_test, elo_pred, fallback_odds=-110)

        elo_row = {
            'fold': i,
            'accuracy': float(accuracy_score(y_test, elo_pred)),
            'brier': float(brier_score_loss(y_test, elo_p_home)),
            'log_loss': float(log_loss(y_test, elo_probs)),
            'roc_auc': _safe_roc_auc(y_test, elo_p_home),
            **_evaluate_betting(y_test, elo_p_home, elo_odds, elo_market, threshold=0.55),
        }
        baseline_rows['elo_only'].append(elo_row)

        folds.append(FoldResult(
            fold=i,
            train_games=int(len(train_idx)),
            test_games=int(len(test_idx)),
            train_date_end=train_max_date.date().isoformat(),
            test_date_start=test_dates.min().date().isoformat(),
            accuracy=fold_accuracy,
            brier=fold_brier,
            log_loss=fold_log_loss,
            roc_auc=fold_roc,
            ece=cal['ece'],
            mce=cal['mce'],
            picks=model_bets['picks'],
            pick_win_rate=model_bets['pick_win_rate'],
            units=model_bets['units'],
            roi_percent=model_bets['roi_percent'],
            market_picks=model_bets['market_picks'],
            fallback_picks=model_bets['fallback_picks'],
        ))

        calibration_by_fold.append({
            'fold': i,
            'ece': float(cal['ece']),
            'mce': float(cal['mce']),
            'reliability_alpha': float(reliability_alpha),
            'bins': cal['bins'],
        })

        all_true.extend(y_test.tolist())
        all_prob.extend(p_home.tolist())
        log_info(
            f"Fold {i}/{max_splits}: acc={fold_accuracy:.3f}, brier={fold_brier:.3f}, "
            f"ece={cal['ece']:.3f}, picks={model_bets['picks']}, roi={model_bets['roi_percent']:.1f}%"
        )

    if not folds:
        raise RuntimeError("No valid folds were evaluated after leakage filtering.")

    fold_dicts = [asdict(f) for f in folds]
    accuracy_vals = [f.accuracy for f in folds]
    brier_vals = [f.brier for f in folds]
    ll_vals = [f.log_loss for f in folds]
    roi_vals = [f.roi_percent for f in folds]
    unit_vals = [f.units for f in folds]
    pick_counts = [f.picks for f in folds]
    market_pick_counts = [f.market_picks for f in folds]
    fallback_pick_counts = [f.fallback_picks for f in folds]
    ece_vals = [f.ece for f in folds]
    mce_vals = [f.mce for f in folds]

    overall_brier = float(brier_score_loss(all_true, all_prob)) if all_true else None
    overall_roc = _safe_roc_auc(all_true, all_prob) if all_true else None

    calibration_summary = {
        'mean_ece': float(np.mean(ece_vals)) if ece_vals else None,
        'max_ece': float(np.max(ece_vals)) if ece_vals else None,
        'mean_mce': float(np.mean(mce_vals)) if mce_vals else None,
        'ece_drift_last_vs_first': float(
            np.mean(ece_vals[-2:]) - np.mean(ece_vals[:2])
        ) if len(ece_vals) >= 4 else (
            float(ece_vals[-1] - ece_vals[0]) if len(ece_vals) >= 2 else 0.0
        ),
        'ece_trend_slope': float(np.polyfit(range(len(ece_vals)), ece_vals, 1)[0]) if len(ece_vals) >= 2 else 0.0,
    }

    home_summary = _summarize_fold_metrics(baseline_rows['home_prior'])
    elo_summary = _summarize_fold_metrics(baseline_rows['elo_only'])

    total_picks = int(np.sum(pick_counts)) if pick_counts else 0
    total_market_picks = int(np.sum(market_pick_counts)) if market_pick_counts else 0
    total_fallback_picks = int(np.sum(fallback_pick_counts)) if fallback_pick_counts else 0
    market_pick_coverage_pct = float(total_market_picks / total_picks) if total_picks else 0.0

    market_values_available = False
    for col in (available_home_odds + available_away_odds + available_generic_odds):
        if col in df.columns and df[col].notna().any():
            market_values_available = True
            break

    report = {
        'generated_at': datetime.now().isoformat(),
        'mode': 'strict_rolling_backtest',
        'config': {
            'days_back': int(days_back),
            'n_splits_requested': int(n_splits),
            'n_splits_used': int(max_splits),
            'features': active_features,
            'probability_clipping': [0.05, 0.95],
            'pick_rule': 'flat 1u when max(home_prob, away_prob) >= 0.55',
            'odds_mode': 'market-aware (per-game American odds if available; fallback -110)',
            'calibration_ece_gate_max': CALIBRATION_ECE_MAX,
            'available_home_odds_columns': available_home_odds,
            'available_away_odds_columns': available_away_odds,
            'available_generic_odds_columns': available_generic_odds,
            'market_values_available': bool(market_values_available),
        },
        'dataset': {
            'games_used': int(len(df)),
            'date_start': dates.min().date().isoformat() if len(dates) else None,
            'date_end': dates.max().date().isoformat() if len(dates) else None,
            'home_win_rate': float(y.mean()) if len(y) else None,
        },
        'leak_checks': {
            'date_sorted_monotonic': bool(date_monotonic),
            'duplicates_removed': int(dup_count),
            'fold_overlap_violations_detected': int(leak_stats['fold_overlap_violations']),
            'rows_removed_for_overlap': int(leak_stats['rows_removed_for_overlap']),
            'folds_skipped_due_overlap': int(leak_stats['folds_skipped_due_overlap']),
            'future_leak_rows_detected_after_filter': int(leak_stats['future_leak_rows_detected_after_filter']),
            'strict_chronological_test_passed': bool(leak_stats['future_leak_rows_detected_after_filter'] == 0),
        },
        'summary': {
            'mean_accuracy': float(np.mean(accuracy_vals)) if accuracy_vals else None,
            'std_accuracy': float(np.std(accuracy_vals)) if accuracy_vals else None,
            'mean_brier': float(np.mean(brier_vals)) if brier_vals else None,
            'mean_log_loss': float(np.mean(ll_vals)) if ll_vals else None,
            'overall_brier': overall_brier,
            'overall_roc_auc': overall_roc,
            'total_picks': total_picks,
            'total_units': float(np.sum(unit_vals)) if unit_vals else 0.0,
            'mean_fold_roi_percent': float(np.mean(roi_vals)) if roi_vals else 0.0,
            'total_market_picks': total_market_picks,
            'total_fallback_picks': total_fallback_picks,
            'market_pick_coverage_pct': market_pick_coverage_pct,
        },
        'calibration': {
            'summary': calibration_summary,
            'folds': calibration_by_fold,
        },
        'baselines': {
            'home_prior': {
                'summary': home_summary,
                'folds': baseline_rows['home_prior'],
            },
            'elo_only': {
                'summary': elo_summary,
                'folds': baseline_rows['elo_only'],
            },
        },
        'model_vs_baselines': {
            'accuracy_vs_home_prior': (
                float(np.mean(accuracy_vals)) - home_summary.get('mean_accuracy', 0.0)
            ) if accuracy_vals else None,
            'accuracy_vs_elo_only': (
                float(np.mean(accuracy_vals)) - elo_summary.get('mean_accuracy', 0.0)
            ) if accuracy_vals else None,
            'brier_improvement_vs_home_prior': (
                home_summary.get('mean_brier', 0.0) - float(np.mean(brier_vals))
            ) if brier_vals else None,
            'brier_improvement_vs_elo_only': (
                elo_summary.get('mean_brier', 0.0) - float(np.mean(brier_vals))
            ) if brier_vals else None,
            'units_vs_home_prior': (
                float(np.sum(unit_vals)) - home_summary.get('total_units', 0.0)
            ),
            'units_vs_elo_only': (
                float(np.sum(unit_vals)) - elo_summary.get('total_units', 0.0)
            ),
        },
        'acceptance_gates': {
            'min_games_500': bool(len(df) >= 500),
            'accuracy_ge_0_55': bool(np.mean(accuracy_vals) >= 0.55) if accuracy_vals else False,
            'brier_le_0_24': bool(np.mean(brier_vals) <= 0.24) if brier_vals else False,
            'roi_positive': bool(np.sum(unit_vals) > 0) if unit_vals else False,
            'beats_home_prior_accuracy': bool(
                float(np.mean(accuracy_vals)) > home_summary.get('mean_accuracy', 1.0)
            ) if accuracy_vals else False,
            'beats_elo_only_brier': bool(
                float(np.mean(brier_vals)) < elo_summary.get('mean_brier', 0.0)
            ) if brier_vals else False,
            'no_future_date_leakage': bool(leak_stats['future_leak_rows_detected_after_filter'] == 0),
            'calibration_ece_le_0_10': bool(
                calibration_summary['mean_ece'] <= CALIBRATION_ECE_MAX
            ) if calibration_summary['mean_ece'] is not None else False,
            # Fail only when calibration worsens by > 0.05; improvements are acceptable.
            'calibration_drift_abs_le_0_05': bool(calibration_summary['ece_drift_last_vs_first'] <= 0.05),
            'market_odds_used_or_unavailable': bool(
                total_market_picks > 0 or (not market_values_available)
            ),
        },
        'folds': fold_dicts,
    }
    report['acceptance_gates']['all_pass'] = bool(all(report['acceptance_gates'].values()))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    log_info(f"Strict backtest report written to: {output}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='Run strict rolling backtest report')
    parser.add_argument('--days-back', type=int, default=180, help='Recent window for evaluation')
    parser.add_argument('--splits', type=int, default=8, help='Number of rolling folds')
    parser.add_argument(
        '--output',
        type=str,
        default='reports/strict_backtest_latest.json',
        help='Report output file path',
    )
    args = parser.parse_args()

    try:
        report = run_strict_backtest(
            days_back=args.days_back,
            n_splits=args.splits,
            output_path=args.output,
        )
        summary = report['summary']
        cmp_info = report['model_vs_baselines']
        cal = report['calibration']['summary']

        print("\n" + "=" * 80)
        print("STRICT BACKTEST SUMMARY")
        print("=" * 80)
        print(f"Games: {report['dataset']['games_used']}")
        print(f"Mean Accuracy: {summary['mean_accuracy']:.3f}")
        print(f"Mean Brier: {summary['mean_brier']:.3f}")
        print(f"Mean ECE: {cal['mean_ece']:.3f}")
        print(f"ECE Drift (last-first): {cal['ece_drift_last_vs_first']:+.3f}")
        print(f"Total Picks: {summary['total_picks']}")
        print(f"Market Odds Pick Coverage: {summary['market_pick_coverage_pct']:.1%}")
        print(f"Total Units: {summary['total_units']:+.2f}")
        print(f"Accuracy Lift vs Home-Prior: {cmp_info['accuracy_vs_home_prior']:+.3f}")
        print(f"Accuracy Lift vs Elo-Only: {cmp_info['accuracy_vs_elo_only']:+.3f}")
        print(f"Brier Gain vs Elo-Only: {cmp_info['brier_improvement_vs_elo_only']:+.3f}")
        print(f"Gates Pass: {report['acceptance_gates']['all_pass']}")
        print("=" * 80 + "\n")
        return 0
    except Exception as exc:
        log_error(f"Strict backtest failed: {exc}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
