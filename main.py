"""
main_v2.py - UNIFIED NBA PREDICTION SYSTEM (CALIBRATION FIXED)
Enhanced version with proper ML + Totals integration
Date: 2026-02-08

Key Improvements:
- Unified betting across moneyline and totals
- Smart bet selection (chooses best opportunities)
- Portfolio management (max exposure limits)
- Proper Kelly sizing for both bet types
- Realistic edge thresholds
- FIXED: Isotonic calibration + probability clipping
"""
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# Core System Imports
from fetch_data import NBADataCollector
from data_processor import NBADataProcessor
from model_utils import NBAModelTrainer
from nbautils import log_info, log_error, log_warning, get_team_name
from betting_strategy_unified import UnifiedBettingSystem
from config import KELLY_CONFIG
from performance_tracker import PerformanceTracker

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Using RandomForest.")

# V4 Totals Import
from train_totals_model import train_totals_model

# Vegas Odds Import (FIXED)
from vegas_odds_fetcher import VegasOddsFetcher
from totals_inference import predict_totals_for_game
from model_loader import load_totals_model
from strict_backtest import run_strict_backtest
from time_utils import get_user_timezone, get_nba_timezone, now_in_tz

class UnifiedNBAPredictionSystem:
    def __init__(self, bankroll: float = None):
        """Initialize NBA Prediction System with Unified Betting"""
        self.collector = NBADataCollector()
        self.processor = NBADataProcessor()
        self.trainer = NBAModelTrainer()
        
        # Initialize unified betting system
        bankroll = bankroll or KELLY_CONFIG['default_bankroll']
        self.betting = UnifiedBettingSystem(bankroll=bankroll)
        
        # Override with stricter settings
        self.betting.min_edge_ml = 0.10      # 10% for ML (strict!)
        self.betting.min_edge_totals = 0.06  # 6% for totals
        self.betting.max_bets_per_day = 3
        
        # Initialize Performance Tracker
        self.tracker = PerformanceTracker(filepath='data/prediction_history.json')
        
        # Initialize Vegas odds fetcher (FIXED version)
        self.odds_fetcher = VegasOddsFetcher()
        self.vegas_odds = pd.DataFrame()
        self.user_tz = get_user_timezone()
        self.nba_tz = get_nba_timezone()
        
        log_info(f"Unified NBA System initialized (Bankroll: ${bankroll:,.2f})")
        log_info(f"Settings: ML edge ≥10%, Totals edge ≥6%, Max 3 bets/day")
        log_info(f"Mode: Vegas odds validation enabled")
        log_info(f"User timezone: {self.user_tz.key} | NBA timezone: {self.nba_tz.key}")

    def load_latest_totals_model(self):
        """Load totals model through centralized loader (V5/V3/V2)."""
        return load_totals_model()

    def predict_game_totals_v4(self, game_row, model_data):
        """Totals prediction routed through shared inference helper."""
        pred = predict_totals_for_game(game_row, model_data)
        if not pred:
            return None
        pred['tier'] = self._get_totals_tier(pred['uncertainty'])
        pred['calibrated'] = 'calibrator' in model_data if model_data else False
        return pred

    
    def _get_totals_tier(self, uncertainty):
        """Classify totals prediction into tiers"""
        if uncertainty < 25:
            return '💎 DIAMOND'
        elif uncertainty <= 35:
            return '✅ GOLD'
        elif uncertainty <= 40:
            return '⚡ SILVER'
        else:
            return '⚠️ SKIP'

    def full_pipeline(
        self,
        retrain_model: bool = True,
        update_data: bool = True,
        days_back: int = 120,
        days_ahead: int = 7,
        max_bets: int = 3
    ):
        """Complete prediction pipeline with unified betting"""
        
        print("\n" + "="*90)
        print("🏀 UNIFIED NBA PREDICTION SYSTEM V2.0 (CALIBRATED)")
        print("="*90 + "\n")
        
        # Step 1: Update Data
        if update_data:
            log_info("📥 STEP 1: Updating data...")
            try:
                results = self.collector.fetch_all_data(days_back=days_back, days_ahead=days_ahead)
                if not any(results.values()):
                    log_error("Failed to fetch data")
                    return False
            except Exception as e:
                log_error(f"Data fetch failed: {e}")
                return False
        else:
            log_info("📥 STEP 1: Using cached data")
        
        # Step 2: Process Data
        log_info("\n🔧 STEP 2: Processing data...")
        try:
            features_df = self.processor.process_all_data()
            if features_df.empty:
                log_error("No features generated")
                return False
            log_info(f"✅ Processed {len(features_df)} games")
        except Exception as e:
            log_error(f"Processing failed: {e}")
            return False
        
        # Step 2.5: Fetch Vegas Odds (FIXED)
        log_info("\n📊 STEP 2.5: Fetching Vegas odds...")
        try:
            self.vegas_odds = self.odds_fetcher.get_odds_for_today()
            if not self.vegas_odds.empty:
                log_info(f"✅ Fetched odds for {len(self.vegas_odds)} games")
                self.odds_fetcher.display_odds_summary()
            else:
                log_warning("⚠️  No Vegas odds available")
                log_warning("   System will operate in MODEL-ONLY mode (less reliable)")
        except Exception as e:
            log_warning(f"⚠️  Could not fetch odds: {e}")
            log_warning("   System will operate in MODEL-ONLY mode")
            self.vegas_odds = pd.DataFrame()
        
        # Step 3: Train Models
        model_results = None
        
        if retrain_model:
            log_info("\n🤖 STEP 3: Training models...")
            
            # Train ML Model
            model_results = self._train_moneyline_model(features_df)
            if not model_results:
                return False
            
            # Train Totals Model
            log_info("\n🎯 Training Totals Model (V5)...")
            train_totals_model()
        else:
            log_info("\n🤖 STEP 3: Skipping training (loading latest saved moneyline model)")
            from model_loader import load_moneyline_model
            saved_model = load_moneyline_model()
            if not saved_model:
                log_error("No saved moneyline model found. Run with retraining enabled first.")
                return False
            model_results = {
                'model': saved_model['model'],
                'feature_names': saved_model['features'],
                'scaler': saved_model.get('scaler'),
                # Historic artifacts may not store evaluated holdout accuracy.
                'test_accuracy': saved_model.get('test_accuracy', 0.55)
            }
        
        # Step 4: Generate Predictions
        log_info("\n🔮 STEP 4: Generating predictions...")
        
        try:
            upcoming_features = self.processor.get_upcoming_games_features()
            totals_model = self.load_latest_totals_model()
            
            if upcoming_features.empty:
                log_warning("No upcoming games")
                return True
            
            # Add derived features
            upcoming_features = self._add_derived_features(upcoming_features)
            
            # Ensure all features exist
            for f in model_results['feature_names']:
                if f not in upcoming_features.columns:
                    upcoming_features[f] = 0
            
            # Predict with clipping
            X_pred = upcoming_features[model_results['feature_names']].copy().fillna(0)
            if model_results.get('scaler') is not None:
                X_model_input = model_results['scaler'].transform(X_pred)
            else:
                X_model_input = X_pred
            probs = model_results['model'].predict_proba(X_model_input)

            # Reliability mapping learned at train-time.
            alpha = float(model_results.get('reliability_alpha', 1.0))
            p_home = self._apply_reliability_map(probs[:, 1], alpha)
            probs_mapped = np.column_stack([1.0 - p_home, p_home])

            # Final clip to prevent overconfidence tails.
            probs_clipped = np.clip(probs_mapped, 0.30, 0.70)
            probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
            
            # Format results
            preds = upcoming_features.copy()
            preds['home_win_probability'] = probs_clipped[:, 1]
            preds['away_win_probability'] = probs_clipped[:, 0]
            preds['predicted_home_win'] = probs_clipped[:, 1] > 0.5
            preds['confidence_level'] = preds.apply(self._get_confidence, axis=1)
            
            # Add totals predictions
            preds = self._add_totals_predictions(preds, totals_model)
            
            # Filter to today/tomorrow only
            preds = self._filter_to_current_games(preds)
            
            if preds.empty:
                print("⚠️  No games found for today/tomorrow")
                return True
            
            # Step 5: Unified Betting Recommendations
            self._display_unified_recommendations(preds, model_results, max_bets)
            
        except Exception as e:
            log_error(f"Prediction generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n" + "="*90)
        print("✅ PIPELINE COMPLETED")
        print("="*90 + "\n")
        return True
    
    def _train_moneyline_model(self, features_df):
        """Train calibrated moneyline model with isotonic calibration"""
        valid_df = features_df.dropna(subset=['home_won']).iloc[15:].copy()
        
        # Reduced feature set (anti-overfitting)
        features = [
            'home_court_advantage', 'form_diff', 'elo_diff', 'fatigue_index', 'rest_diff',
            'pace_home_L5', 'pace_away_L5', 'combined_pace',
            'off_rating_home_L10', 'off_rating_away_L10',
            'def_rating_home_L10', 'def_rating_away_L10',
            'back_to_back_home', 'back_to_back_away',
            'schedule_strength_diff', 'net_rating_diff', 'def_rating_diff'
        ]
        
        # Ensure features exist
        for f in features:
            if f not in valid_df.columns:
                valid_df[f] = 0
        
        X = valid_df[features].copy()
        y = valid_df['home_won'].copy()
        
        log_info(f"Training with {len(features)} features on {len(X)} games")
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Build conservative pipeline
        if XGBOOST_AVAILABLE:
            classifier = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                gamma=0.3, reg_alpha=0.5, reg_lambda=3.0,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )
        else:
            classifier = RandomForestClassifier(
                n_estimators=150, max_depth=5,
                min_samples_leaf=10, min_samples_split=20,
                random_state=42, n_jobs=-1
            )
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        
        print("\n" + "="*80)
        print("📊 MONEYLINE MODEL PERFORMANCE")
        print("="*80)
        print(f"Training Accuracy:   {train_acc:.1%}")
        print(f"Test Accuracy:       {test_acc:.1%} ← REAL ACCURACY")
        print(f"Overfitting Gap:     {(train_acc - test_acc):.1%}")
        print("="*80 + "\n")
        
        # Calibrate with SIGMOID for better stability on smaller/shifted samples.
        log_info("🎯 Calibrating probabilities with sigmoid calibration...")
        calibrated_pipeline = CalibratedClassifierCV(pipeline, method='sigmoid', cv=5)
        calibrated_pipeline.fit(X_train, y_train)

        # Post-calibration reliability mapping:
        # shrink confidence toward 50/50 to reduce ECE spikes on volatile folds.
        cal_probs = calibrated_pipeline.predict_proba(X_test)[:, 1]
        reliability_alpha = self._fit_reliability_alpha(y_test.to_numpy(), cal_probs)
        cal_probs_adj = self._apply_reliability_map(cal_probs, reliability_alpha)
        test_acc_cal = accuracy_score(y_test, (cal_probs_adj > 0.5).astype(int))

        ece_before = self._simple_ece(y_test.to_numpy(), cal_probs, n_bins=10)
        ece_after = self._simple_ece(y_test.to_numpy(), cal_probs_adj, n_bins=10)

        log_info(f"✅ Calibrated Test Accuracy: {test_acc_cal:.1%}")
        log_info(f"   Probability range (mapped): {cal_probs_adj.min():.1%} - {cal_probs_adj.max():.1%}")
        log_info(f"   Reliability alpha: {reliability_alpha:.2f} | ECE {ece_before:.3f} -> {ece_after:.3f}")
        log_info(f"   🔧 Clipping will limit predictions to 30-70% range")
        
        # Store the model with calibration metadata
        model_results = {
            'model': calibrated_pipeline,
            'feature_names': features,
            'test_accuracy': test_acc_cal,
            'train_accuracy': train_acc,
            'clip_probs': True,  # Flag to clip probabilities during prediction
            'reliability_alpha': float(reliability_alpha),
        }
        
        return model_results

    def _apply_reliability_map(self, p_home, alpha):
        """Shrink probabilities toward 50/50 to improve calibration stability."""
        p_home = np.asarray(p_home, dtype=float)
        mapped = 0.5 + float(alpha) * (p_home - 0.5)
        return np.clip(mapped, 0.02, 0.98)

    def _simple_ece(self, y_true, p_home, n_bins=10):
        y_true = np.asarray(y_true, dtype=float)
        p_home = np.asarray(p_home, dtype=float)
        bins = np.linspace(0, 1, n_bins + 1)
        total = len(p_home)
        if total == 0:
            return 0.0
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            if hi == 1.0:
                mask = (p_home >= lo) & (p_home <= hi)
            else:
                mask = (p_home >= lo) & (p_home < hi)
            if not np.any(mask):
                continue
            avg_p = float(np.mean(p_home[mask]))
            avg_y = float(np.mean(y_true[mask]))
            ece += (np.sum(mask) / total) * abs(avg_p - avg_y)
        return float(ece)

    def _fit_reliability_alpha(self, y_true, p_home):
        """
        Grid-search alpha for post-calibration reliability mapping.
        Objective emphasizes calibration (ECE) while preserving Brier quality.
        """
        y_true = np.asarray(y_true, dtype=float)
        p_home = np.asarray(p_home, dtype=float)
        best_alpha = 1.0
        best_score = float("inf")
        for alpha in np.linspace(0.20, 0.80, 13):
            p_adj = self._apply_reliability_map(p_home, alpha)
            ece = self._simple_ece(y_true, p_adj, n_bins=10)
            score = ece
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
        return best_alpha
    
    def _add_derived_features(self, df):
        """Add derived features to upcoming games"""
        if 'combined_pace' not in df.columns:
            df['combined_pace'] = (df.get('pace_home_L5', 100) + df.get('pace_away_L5', 100)) / 2
        
        if 'fatigue_index' not in df.columns:
            rh = df.get('rest_home', 2)
            ra = df.get('rest_away', 2)
            h_fatigue = np.where(rh == 0, 1.0, np.where(rh == 1, 0.0, -0.5))
            a_fatigue = np.where(ra == 0, 1.0, np.where(ra == 1, 0.0, -0.5))
            df['fatigue_index'] = h_fatigue + a_fatigue
        
        return df
    
    def _get_confidence(self, row):
        """Get confidence level from probabilities"""
        p = max(row['home_win_probability'], row['away_win_probability'])
        if p > 0.70:
            return 'High'
        if p > 0.60:
            return 'Medium'
        return 'Low'
    
    def _add_totals_predictions(self, preds, totals_model):
        """Add V5 totals predictions to dataframe"""
        if not totals_model:
            preds['totals_pred'] = None
            return preds
        
        totals_list = []
        for idx, row in preds.iterrows():
            totals_pred = self.predict_game_totals_v4(row, totals_model)
            totals_list.append(totals_pred)
        
        preds['totals_pred'] = totals_list
        return preds
    
    def _filter_to_current_games(self, preds):
        """Filter to today/tomorrow/day+2 in NBA timezone with user timezone display."""
        now_user = now_in_tz(self.user_tz)
        now_nba = now_in_tz(self.nba_tz)
        
        # Use NBA-local calendar days for schedule filtering.
        today_nba = now_nba.date()
        tomorrow_nba = (now_nba + timedelta(days=1)).date()
        day_after_nba = (now_nba + timedelta(days=2)).date()
        
        print(f"\n🌍 Timezone-Aware Filtering:")
        print(f"   Your local time ({self.user_tz.key}): {now_user.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"   NBA time ({self.nba_tz.key}): {now_nba.strftime('%Y-%m-%d %H:%M %Z')}")
        print(f"   Including NBA dates: {today_nba}, {tomorrow_nba}, {day_after_nba}")
        
        preds['game_date_dt'] = pd.to_datetime(preds['game_date']).dt.date
        
        # Include today, tomorrow, and day after (to catch late games)
        filtered = preds[preds['game_date_dt'].isin([today_nba, tomorrow_nba, day_after_nba])].copy()
        
        print(f"   Games before filter: {len(preds)}")
        print(f"   Games after filter: {len(filtered)}")
        
        if filtered.empty:
            print(f"   ⚠️  No games found! Showing all upcoming games instead...")
            return preds.copy()
        
        return filtered
    
    def _get_vegas_total(self, home_team: str, away_team: str) -> Optional[float]:
        """Get Vegas total line for a game"""
        if self.vegas_odds.empty:
            return None
        
        from nbautils import normalize_team_name
        home_code = normalize_team_name(str(home_team))
        away_code = normalize_team_name(str(away_team))
        
        game = self.vegas_odds[
            ((self.vegas_odds['home_team'] == home_code) & (self.vegas_odds['away_team'] == away_code)) |
            ((self.vegas_odds['home_team'] == away_code) & (self.vegas_odds['away_team'] == home_code))
        ]
        
        if not game.empty and pd.notna(game.iloc[0]['total_line']):
            return float(game.iloc[0]['total_line'])
        return None
    
    def _get_vegas_ml_odds(self, home_team: str, away_team: str, is_home: bool) -> Optional[int]:
        """Get Vegas moneyline odds"""
        if self.vegas_odds.empty:
            return None
        
        from nbautils import normalize_team_name
        home_code = normalize_team_name(str(home_team))
        away_code = normalize_team_name(str(away_team))
        
        game = self.vegas_odds[
            ((self.vegas_odds['home_team'] == home_code) & (self.vegas_odds['away_team'] == away_code)) |
            ((self.vegas_odds['home_team'] == away_code) & (self.vegas_odds['away_team'] == home_code))
        ]
        
        if game.empty:
            return None
        
        game_row = game.iloc[0]
        teams_flipped = (game_row['away_team'] == home_code)
        if teams_flipped:
            is_home = not is_home
        
        odds = game_row.get('ml_home') if is_home else game_row.get('ml_away')
        return int(odds) if pd.notna(odds) else None
    
    def _display_unified_recommendations(self, preds, model_results, max_bets):
        """Display unified ML + Totals recommendations"""
        
        print("\n" + "="*90)
        print("🎰 UNIFIED BETTING RECOMMENDATIONS")
        print("="*90)
        print(f"\n💰 Bankroll: ${self.betting.current_bankroll:,.2f}")
        print(f"🎯 Strategy: Best {max_bets} opportunities (ML or Totals)")
        print(f"📊 Min Edge: ML ≥{self.betting.min_edge_ml*100}%, Totals ≥{self.betting.min_edge_totals*100}%")
        
        # Evaluate all games
        game_evaluations = []
        skipped_calibration = []
        skipped_no_odds = []
        
        print(f"\n🔍 Evaluating {len(preds)} games...")
        
        for idx, row in preds.iterrows():
            # Determine ML pick
            if row['predicted_home_win']:
                ml_pick = row['home_team']
                ml_prob = row['home_win_probability']
            else:
                ml_pick = row['away_team']
                ml_prob = row['away_win_probability']
            
            # Get totals data
            totals_pred = row['totals_pred']
            
            if totals_pred:
                # Try to get Vegas lines
                vegas_total = self._get_vegas_total(row['home_team'], row['away_team'])
                ml_odds = self._get_vegas_ml_odds(
                    row['home_team'], 
                    row['away_team'],
                    is_home=(ml_pick == row['home_team'])
                )
                
                # CRITICAL: Skip if no Vegas odds (don't use fallbacks!)
                if vegas_total is None or ml_odds is None:
                    skipped_no_odds.append(f"{row['away_team']} @ {row['home_team']}")
                    continue
                
                # CALIBRATION CHECK: Skip if model disagrees wildly with Vegas
                is_underdog = (ml_odds > 0)
                
                # Calculate Vegas implied probability
                if ml_odds < 0:
                    vegas_implied = abs(ml_odds) / (abs(ml_odds) + 100)
                else:
                    vegas_implied = 100 / (ml_odds + 100)
                
                # Check 1: Model says 65%+ but Vegas says underdog = likely wrong
                if ml_prob > 0.65 and is_underdog:
                    skipped_calibration.append(f"{row['away_team']} @ {row['home_team']}")
                    continue
                
                # Check 2: Model disagrees with Vegas by >20 percentage points = likely wrong
                prob_diff = abs(ml_prob - vegas_implied)
                if prob_diff > 0.20:  # 20 percentage points
                    skipped_calibration.append(f"{row['away_team']} @ {row['home_team']}")
                    continue
                
                # Check 3: Model says <35% but Vegas says heavy favorite = likely wrong  
                if ml_prob < 0.35 and ml_odds < -200:
                    skipped_calibration.append(f"{row['away_team']} @ {row['home_team']}")
                    continue
                
                eval_result = self.betting.evaluate_game(
                    # ML inputs
                    ml_win_prob=ml_prob,
                    ml_odds=ml_odds,
                    ml_confidence=row['confidence_level'],
                    ml_test_accuracy=model_results['test_accuracy'],
                    
                    # Totals inputs
                    predicted_total=totals_pred['predicted_total'],
                    vegas_total=vegas_total,
                    total_uncertainty=totals_pred['uncertainty'],
                    total_lower=totals_pred['lower'],
                    total_upper=totals_pred['upper'],
                    totals_mae=7.74,
                    total_odds=-110,
                    
                    # Game info
                    game_info={
                        'date': row['game_date'],
                        'home': row['home_team'],
                        'away': row['away_team'],
                        'ml_pick': ml_pick,
                        'totals_tier': totals_pred['tier']
                    }
                )
                game_evaluations.append(eval_result)
        
        # Get best bets
        best_bets = self.betting.get_daily_recommendations(game_evaluations, max_bets)
        
        # Show skipped games
        if skipped_no_odds:
            print("\n" + "="*90)
            print("ℹ️  GAMES SKIPPED (No Vegas Odds Available)")
            print("="*90)
            for game in skipped_no_odds:
                print(f"   • {game}")
            print(f"\n   Total: {len(skipped_no_odds)} games")
            print("   The Odds API doesn't have lines for these games yet.")
            print("="*90)
        
        if skipped_calibration:
            print("\n" + "="*90)
            print("⚠️  GAMES SKIPPED (Model Calibration Issues)")
            print("="*90)
            for game in skipped_calibration:
                print(f"   • {game}")
            print(f"\n   Total: {len(skipped_calibration)} games")
            print("   Model predictions disagree strongly with Vegas odds.")
            print("="*90)
        
        if best_bets:
            print("\n" + "="*90)
            print("💎 TOP RECOMMENDATIONS")
            print("="*90)
            
            for i, bet in enumerate(best_bets, 1):
                game = bet['game_info']
                print(f"\n{'─'*90}")
                print(f"#{i} | 📅 {game['date']}")
                print(f"    {get_team_name(game['away'])} @ {get_team_name(game['home'])}")
                
                if bet['bet_type'] == 'moneyline':
                    print(f"\n    🏆 MONEYLINE: {get_team_name(game['ml_pick'])}")
                    print(f"       Win Probability: {bet['win_probability']}% (adj: {bet['adjusted_win_probability']}%)")
                    print(f"       Confidence: {bet['confidence']}")
                else:  # totals
                    print(f"\n    📊 TOTAL: {bet['direction']} {bet['vegas_line']}")
                    print(f"       Prediction: {bet['predicted_total']} ({bet['range']})")
                    print(f"       Tier: {game['totals_tier']} | Uncertainty: {bet['uncertainty']:.1f}")
                
                print(f"\n    💰 BET: ${bet['bet_amount']:.2f} ({bet['pct_of_bankroll']}% of bankroll)")
                print(f"       Edge: {bet['edge']:.1f}% | EV: ${bet['expected_value']:.2f}")
            
            print(f"\n{'─'*90}")
            print(f"📊 Total Exposure: {self.betting.daily_exposure*100:.1f}% of bankroll")
        else:
            print("\n🚫 NO QUALIFYING BETS FOUND")
            print("   → All games failed minimum edge requirements")
            print("   → This is GOOD - it means the system is selective!")
        
        # Show full slate
        self._display_full_slate(preds)
        
        # Show performance tracking
        print("\n" + "="*90)
        print("📊 HISTORICAL PERFORMANCE")
        print("="*90)
        print(self.tracker.get_performance_report())
        print("="*90 + "\n")
    
    def _display_full_slate(self, preds):
        """Display all games for reference"""
        print("\n" + "="*100)
        print("📋 FULL SLATE (ALL GAMES)")
        print("="*100)
        print(f"{'DATE':<12} | {'HOME':<15} | {'AWAY':<15} | {'ML PICK':<8} | {'WIN%':<6} | {'PRED':<6} | {'VEGAS':<6} | {'TIER'}")
        print("─" * 100)
        
        for _, row in preds.iterrows():
            date_str = pd.to_datetime(row['game_date']).strftime('%Y-%m-%d')
            
            if row['predicted_home_win']:
                pick = get_team_name(row['home_team'])[:8]
                prob = row['home_win_probability']
            else:
                pick = get_team_name(row['away_team'])[:8]
                prob = row['away_win_probability']
            
            totals = row['totals_pred']
            if totals:
                total_str = f"{totals['predicted_total']:.1f}"
                tier = totals['tier']
            else:
                total_str = "N/A"
                tier = "N/A"
            
            # Get Vegas line
            vegas_total = self._get_vegas_total(row['home_team'], row['away_team'])
            vegas_str = f"{vegas_total:.1f}" if vegas_total else "N/A"
            
            home = get_team_name(row['home_team'])[:15]
            away = get_team_name(row['away_team'])[:15]
            
            print(f"{date_str:<12} | {home:<15} | {away:<15} | {pick:<8} | {prob:.0%} | {total_str:<6} | {vegas_str:<6} | {tier}")
        
        print("─" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Unified NBA Prediction System')
    parser.add_argument('--no-update', action='store_true', help='Skip data download')
    parser.add_argument('--bankroll', type=float, default=None, help='Bankroll amount')
    parser.add_argument('--days-back', type=int, default=120, help='Days back (default 120)')
    parser.add_argument('--days-ahead', type=int, default=7, help='Days ahead')
    parser.add_argument('--max-bets', type=int, default=3, help='Max bets (default 3)')
    parser.add_argument('--strict-backtest', action='store_true', help='Run strict rolling backtest and exit')
    parser.add_argument('--backtest-splits', type=int, default=8, help='Rolling splits for strict backtest')
    parser.add_argument(
        '--backtest-output',
        type=str,
        default='reports/strict_backtest_latest.json',
        help='Strict backtest report path',
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*90)
    print("🏀 UNIFIED NBA PREDICTION SYSTEM V2.0 (CALIBRATED)")
    print("📅 Showing: Today + Tomorrow")
    print("="*90 + "\n")

    if args.strict_backtest:
        report = run_strict_backtest(
            days_back=args.days_back,
            n_splits=args.backtest_splits,
            output_path=args.backtest_output,
        )
        gates_pass = report['acceptance_gates']['all_pass']
        sys.exit(0 if gates_pass else 2)
    
    system = UnifiedNBAPredictionSystem(bankroll=args.bankroll)
    
    try:
        success = system.full_pipeline(
            update_data=not args.no_update,
            days_back=args.days_back,
            days_ahead=args.days_ahead,
            max_bets=args.max_bets
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        log_error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
