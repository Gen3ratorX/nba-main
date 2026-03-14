"""
feature_selection.py - Feature importance analysis and pruning recommendations.

Runs permutation importance across rolling TimeSeriesSplit folds,
computes pairwise correlations, and optionally runs SHAP analysis.
Produces a JSON report with pruning recommendations for human review.

Usage:
    python main.py --analyze-features --days-back 180
    python feature_selection.py          # standalone
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from data_processor import NBADataProcessor
from ensemble import build_ensemble_pipeline
from hyperopt import get_best_params
from nbautils import log_info, log_warning
from strict_backtest import CANONICAL_FEATURES

# Feature category mapping for grouped analysis
FEATURE_CATEGORIES = {
    "Home Court & Form": [
        "home_court_advantage", "form_diff", "elo_diff", "fatigue_index", "rest_diff",
    ],
    "Pace": ["pace_home_L5", "pace_away_L5", "combined_pace"],
    "Efficiency Ratings": [
        "off_rating_home_L10", "off_rating_away_L10",
        "def_rating_home_L10", "def_rating_away_L10",
    ],
    "Back-to-Back": ["back_to_back_home", "back_to_back_away"],
    "Schedule & Ratings": ["schedule_strength_diff", "net_rating_diff", "def_rating_diff"],
    "EWMA": [
        "ewma_ppg_home", "ewma_ppg_away",
        "ewma_papg_home", "ewma_papg_away",
        "ewma_net_diff",
    ],
    "Vegas": [
        "vegas_implied_home", "vegas_implied_away", "vegas_implied_diff",
        "vegas_total_line", "has_vegas_odds",
    ],
    "Player Impact": [
        "minutes_missing_home", "minutes_missing_away",
        "star_missing_home", "star_missing_away",
        "roster_strength_home", "roster_strength_away",
    ],
    "Line Movement": [
        "ml_movement_home", "ml_movement_magnitude",
        "total_line_movement", "odds_snapshot_count",
    ],
}

_CORR_THRESHOLD = 0.80
_REPORT_DIR = Path("reports")


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def _permutation_importance_fold(
    model, X_test: np.ndarray, y_test: np.ndarray, n_repeats: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run permutation importance on one fold. Returns (means, stds)."""
    try:
        from joblib import parallel_backend
        with parallel_backend("threading"):
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_repeats, random_state=42, n_jobs=1, scoring="accuracy",
            )
    except Exception:
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats, random_state=42, n_jobs=1, scoring="accuracy",
        )
    return result.importances_mean, result.importances_std


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def _correlation_analysis(
    X: pd.DataFrame, features: List[str], threshold: float = _CORR_THRESHOLD,
) -> Dict:
    """Compute Pearson + Spearman correlations and identify redundant pairs."""
    X_feat = X[features].copy()

    pearson = X_feat.corr(method="pearson")
    spearman = X_feat.corr(method="spearman")

    # Find high-correlation pairs (upper triangle only)
    high_pairs = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            p = pearson.iloc[i, j]
            s = spearman.iloc[i, j]
            if abs(p) > threshold or abs(s) > threshold:
                high_pairs.append({
                    "feature_a": features[i],
                    "feature_b": features[j],
                    "pearson": round(float(p), 4),
                    "spearman": round(float(s), 4),
                })

    # Build correlation clusters via union-find
    clusters = _union_find_clusters(high_pairs, features)

    return {
        "high_correlation_pairs": high_pairs,
        "correlation_clusters": clusters,
    }


def _union_find_clusters(pairs: List[Dict], all_features: List[str]) -> List[List[str]]:
    """Group correlated features into connected components."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in pairs:
        union(pair["feature_a"], pair["feature_b"])

    groups: Dict[str, List[str]] = defaultdict(list)
    for pair in pairs:
        for feat in (pair["feature_a"], pair["feature_b"]):
            root = find(feat)
            if feat not in groups[root]:
                groups[root].append(feat)

    return [sorted(members) for members in groups.values() if len(members) > 1]


# ---------------------------------------------------------------------------
# SHAP analysis (optional)
# ---------------------------------------------------------------------------

def _shap_analysis(
    model, X_test: np.ndarray, feature_names: List[str],
) -> Optional[Dict]:
    """Run SHAP TreeExplainer on base learners. Returns None if shap not installed."""
    try:
        import shap
    except ImportError:
        return None

    shap_importances = np.zeros(len(feature_names))
    meta_coefs = model.meta_learner_.coef_[0] if hasattr(model.meta_learner_, "coef_") else None
    weights = []

    for idx, (name, pipeline) in enumerate(model.base_learners_):
        clf = pipeline.named_steps.get("clf")
        if clf is None:
            continue
        # Only tree-based models support TreeExplainer
        if name not in ("xgb", "lgbm"):
            continue

        try:
            # Pre-transform through imputer + scaler
            X_transformed = pipeline[:-1].transform(X_test)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
            # For binary classification, shap_values may be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs = np.abs(shap_values).mean(axis=0)

            weight = abs(float(meta_coefs[idx])) if meta_coefs is not None and idx < len(meta_coefs) else 1.0
            weights.append(weight)
            shap_importances += mean_abs * weight
        except Exception as e:
            log_warning(f"SHAP failed for {name}: {e}")
            continue

    if sum(weights) == 0:
        return None

    shap_importances /= sum(weights)
    ranked = sorted(
        zip(feature_names, shap_importances),
        key=lambda x: x[1], reverse=True,
    )

    return {
        "rankings": [
            {"rank": i + 1, "feature": f, "mean_abs_shap": round(float(v), 6)}
            for i, (f, v) in enumerate(ranked)
        ],
        "base_learner_weights": {
            name: round(float(abs(meta_coefs[idx])) if meta_coefs is not None and idx < len(meta_coefs) else 1.0, 4)
            for idx, (name, _) in enumerate(model.base_learners_)
            if name in ("xgb", "lgbm")
        },
    }


# ---------------------------------------------------------------------------
# Pruning recommendations
# ---------------------------------------------------------------------------

def _detect_zero_variance(X: pd.DataFrame, features: List[str], tol: float = 1e-8) -> List[str]:
    """Find features with near-zero variance."""
    return [f for f in features if X[f].std() < tol]


def _recommend_pruning(
    rankings: List[Dict],
    correlation_data: Dict,
    zero_features: List[str],
) -> List[Dict]:
    """Generate pruning recommendations with rationale."""
    recs = []

    # 1. Zero-variance features
    for feat in zero_features:
        recs.append({
            "feature": feat,
            "reason": "always_zero",
            "detail": "Near-zero variance across all training data. No signal available yet.",
            "confidence": "high",
        })

    # Build rank lookup
    rank_map = {r["feature"]: r for r in rankings}
    n = len(rankings)
    bottom_quartile = set(r["feature"] for r in rankings if r["rank"] > n * 0.75)

    # 2. Negative importance
    for r in rankings:
        if r["mean_importance"] < 0 and r["feature"] not in zero_features:
            recs.append({
                "feature": r["feature"],
                "reason": "negative_importance",
                "detail": f"Mean permutation importance = {r['mean_importance']:.4f}. "
                          f"Shuffling this feature improves accuracy.",
                "confidence": "high",
            })

    # 3. Low importance + high correlation with better feature
    for pair in correlation_data.get("high_correlation_pairs", []):
        a, b = pair["feature_a"], pair["feature_b"]
        if a in zero_features or b in zero_features:
            continue
        ra = rank_map.get(a, {}).get("rank", n)
        rb = rank_map.get(b, {}).get("rank", n)
        # The worse-ranked one in the correlated pair is the prune candidate
        worse, better = (a, b) if ra > rb else (b, a)
        if worse in bottom_quartile and worse not in [r["feature"] for r in recs]:
            recs.append({
                "feature": worse,
                "reason": "redundant",
                "detail": f"Bottom-quartile importance (rank {rank_map.get(worse, {}).get('rank', '?')}/{n}) "
                          f"and |r|>{_CORR_THRESHOLD:.2f} with '{better}' "
                          f"(Pearson={pair['pearson']}, Spearman={pair['spearman']}).",
                "confidence": "medium",
            })

    return recs


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _print_console_report(report: Dict):
    """Print formatted feature analysis summary."""
    print(f"\n{'='*80}")
    print(f"   FEATURE IMPORTANCE ANALYSIS ({report['config']['n_splits']} folds, "
          f"{report['dataset']['games_used']} games)")
    print(f"{'='*80}")

    # Top features
    rankings = report["permutation_importance"]["rankings"]
    print(f"\n   TOP 10 FEATURES (by mean permutation importance):")
    print(f"   {'Rank':<6}{'Feature':<30}{'Importance':<14}{'Std':<10}")
    print(f"   {'-'*60}")
    for r in rankings[:10]:
        print(f"   {r['rank']:<6}{r['feature']:<30}{r['mean_importance']:>+.5f}     "
              f"{r['std_importance']:.5f}")

    # Bottom features
    print(f"\n   BOTTOM 10 FEATURES:")
    print(f"   {'Rank':<6}{'Feature':<30}{'Importance':<14}{'Std':<10}")
    print(f"   {'-'*60}")
    for r in rankings[-10:]:
        print(f"   {r['rank']:<6}{r['feature']:<30}{r['mean_importance']:>+.5f}     "
              f"{r['std_importance']:.5f}")

    # Category breakdown
    by_cat = report["permutation_importance"].get("by_category", {})
    if by_cat:
        print(f"\n   IMPORTANCE BY CATEGORY:")
        print(f"   {'Category':<25}{'Sum Importance':<18}{'Features':<10}")
        print(f"   {'-'*53}")
        for cat, info in sorted(by_cat.items(), key=lambda x: x[1]["sum_importance"], reverse=True):
            print(f"   {cat:<25}{info['sum_importance']:>+.5f}       {info['feature_count']}")

    # Correlation
    corr = report["correlation_analysis"]
    pairs = corr.get("high_correlation_pairs", [])
    clusters = corr.get("correlation_clusters", [])
    if pairs:
        print(f"\n   HIGH-CORRELATION PAIRS (|r| > {_CORR_THRESHOLD}):")
        for p in pairs[:15]:
            print(f"   {p['feature_a']:<30} <-> {p['feature_b']:<30} "
                  f"Pearson={p['pearson']:+.3f}  Spearman={p['spearman']:+.3f}")
    if clusters:
        print(f"\n   CORRELATION CLUSTERS:")
        for i, cl in enumerate(clusters, 1):
            print(f"   Cluster {i}: {', '.join(cl)}")

    # SHAP
    shap_data = report.get("shap_analysis")
    if shap_data:
        print(f"\n   SHAP ANALYSIS (tree-based learners):")
        for r in shap_data["rankings"][:10]:
            print(f"   {r['rank']:<6}{r['feature']:<30}{r['mean_abs_shap']:.6f}")
    else:
        print(f"\n   SHAP: not available (install shap for directional analysis)")

    # Zero-variance
    zero = report.get("zero_variance_features", [])
    if zero:
        print(f"\n   ZERO-VARIANCE FEATURES ({len(zero)}):")
        for f in zero:
            print(f"   - {f}")

    # Recommendations
    recs = report.get("pruning_recommendations", [])
    if recs:
        print(f"\n   PRUNING RECOMMENDATIONS ({len(recs)}):")
        for r in recs:
            conf = r["confidence"].upper()
            print(f"   [{conf}] {r['feature']} — {r['reason']}")
            print(f"          {r['detail']}")
    else:
        print(f"\n   No pruning recommendations.")

    # Summary
    s = report["summary"]
    print(f"\n{'='*80}")
    print(f"   Total features: {s['total_features']}  |  "
          f"Zero-var: {s['zero_variance_count']}  |  "
          f"Negative-imp: {s['negative_importance_count']}  |  "
          f"Recommended prune: {s['recommended_prune_count']}")
    if s["recommended_prune_count"] > 0:
        est = s["total_features"] - s["recommended_prune_count"]
        print(f"   Estimated post-prune feature count: {est}")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_feature_analysis(
    days_back: int = 180,
    n_splits: int = 8,
    n_repeats: int = 10,
    output_path: str = "reports/feature_analysis_latest.json",
) -> Dict:
    """
    Run full feature importance analysis across rolling backtest folds.

    Returns the analysis report dict and writes it to JSON.
    """
    features = list(CANONICAL_FEATURES)
    log_info(f"Feature analysis: {len(features)} features, {n_splits} folds, {days_back} days")

    # ---- Load data (same as strict_backtest.py) ----
    processor = NBADataProcessor()
    df = processor.process_all_data()
    if df.empty:
        log_warning("No data available for feature analysis.")
        return {}

    df = df.sort_values("game_date").reset_index(drop=True)

    # Ensure features exist
    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    X = df[features].fillna(0).values
    df = df.dropna(subset=["home_won"])
    y = df["home_won"].values.astype(int)

    # ---- Detect zero-variance features ----
    X_df = df[features].fillna(0)
    zero_var_features = _detect_zero_variance(X_df, features)

    # ---- Rolling folds with permutation importance ----
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_importances = []
    last_model = None

    print(f"\nRunning permutation importance across {n_splits} folds...")
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit ensemble
        model = build_ensemble_pipeline(
            xgb_params=get_best_params("xgboost"),
            lgbm_params=get_best_params("lightgbm"),
        )
        model.fit(X_train, y_train)

        # Permutation importance on test fold
        means, stds = _permutation_importance_fold(model, X_test, y_test, n_repeats=n_repeats)
        fold_importances.append({"means": means, "stds": stds})
        last_model = model

        acc = (model.predict(X_test) == y_test).mean()
        print(f"  Fold {fold_idx}/{n_splits}: test_acc={acc:.3f}")

    # ---- Aggregate importance ----
    all_means = np.array([fi["means"] for fi in fold_importances])
    all_stds = np.array([fi["stds"] for fi in fold_importances])
    agg_mean = all_means.mean(axis=0)
    agg_std = np.sqrt((all_stds ** 2).mean(axis=0) + all_means.var(axis=0))

    # Build ranked list
    order = np.argsort(-agg_mean)
    rankings = []
    for rank, idx in enumerate(order, 1):
        rankings.append({
            "rank": rank,
            "feature": features[idx],
            "mean_importance": round(float(agg_mean[idx]), 6),
            "std_importance": round(float(agg_std[idx]), 6),
            "fold_values": [round(float(all_means[f, idx]), 6) for f in range(len(fold_importances))],
        })

    # Category breakdown
    by_category = {}
    for cat_name, cat_features in FEATURE_CATEGORIES.items():
        cat_indices = [features.index(f) for f in cat_features if f in features]
        if cat_indices:
            by_category[cat_name] = {
                "sum_importance": round(float(agg_mean[cat_indices].sum()), 6),
                "mean_importance": round(float(agg_mean[cat_indices].mean()), 6),
                "feature_count": len(cat_indices),
            }

    # ---- Correlation analysis ----
    print("Computing feature correlations...")
    corr_data = _correlation_analysis(X_df, features)

    # ---- SHAP analysis (optional) ----
    print("Attempting SHAP analysis...")
    shap_data = _shap_analysis(last_model, X[-len(fold_importances[-1]["means"]):], features) if last_model else None

    # ---- Pruning recommendations ----
    recs = _recommend_pruning(rankings, corr_data, zero_var_features)

    # ---- Build report ----
    neg_count = sum(1 for r in rankings if r["mean_importance"] < 0)
    report = {
        "generated_at": datetime.now().isoformat(),
        "mode": "feature_analysis",
        "config": {
            "days_back": days_back,
            "n_splits": n_splits,
            "n_repeats_permutation": n_repeats,
            "correlation_threshold": _CORR_THRESHOLD,
            "features_analyzed": features,
            "shap_available": shap_data is not None,
        },
        "dataset": {
            "games_used": len(df),
            "date_start": str(df["game_date"].min()),
            "date_end": str(df["game_date"].max()),
        },
        "permutation_importance": {
            "rankings": rankings,
            "by_category": by_category,
        },
        "shap_analysis": shap_data,
        "correlation_analysis": corr_data,
        "zero_variance_features": zero_var_features,
        "pruning_recommendations": recs,
        "summary": {
            "total_features": len(features),
            "zero_variance_count": len(zero_var_features),
            "negative_importance_count": neg_count,
            "high_correlation_pairs_count": len(corr_data.get("high_correlation_pairs", [])),
            "recommended_prune_count": len(recs),
        },
    }

    # Write report
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log_info(f"Feature analysis report written to: {out}")

    # Console summary
    _print_console_report(report)

    return report


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Feature Importance Analysis")
    parser.add_argument("--days-back", type=int, default=180)
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    run_feature_analysis(
        days_back=args.days_back,
        n_splits=args.splits,
        n_repeats=args.repeats,
    )
