"""
main.py - NBA Prediction System with Professional Gambling Strategy
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from fetch_data import NBADataCollector
from data_processor import NBADataProcessor
from model_utils import create_enhanced_model, predict_upcoming_games_enhanced, NBAModelTrainer
from nbautils import log_info, log_error, log_warning, get_team_name
from betting_strategy import KellyBetting
from config import KELLY_CONFIG
from performance_tracker import PerformanceTracker  # ADDED IMPORT

class NBAPredictionSystem:
    def __init__(self, bankroll: float = None):
        """Initialize NBA Prediction System with Professional Gambling Logic
        
        Args:
            bankroll: Starting bankroll for betting (defaults to config)
        """
        self.collector = NBADataCollector()
        self.processor = NBADataProcessor()
        self.trainer = NBAModelTrainer()
        
        # Initialize betting system
        bankroll = bankroll or KELLY_CONFIG['default_bankroll']
        self.betting = KellyBetting(bankroll=bankroll)
        
        # ADDED: Initialize Performance Tracker
        self.tracker = PerformanceTracker()
        
        log_info(f"NBA Prediction System initialized (Bankroll: ${bankroll:,.2f})")

    def full_pipeline(
        self, 
        retrain_model: bool = True, 
        update_data: bool = True, 
        tune_hyperparameters: bool = False, 
        run_backtest: bool = False,
        days_back: int = 60,
        days_ahead: int = 7,
        min_edge_override: float = None,
        max_bets_per_day: int = 5
    ):
        """Complete prediction pipeline with smart betting logic"""
        
        print("\n" + "="*80)
        print("🏀 NBA PREDICTION SYSTEM - PROFESSIONAL GAMBLING MODE")
        print("="*80 + "\n")
        
        # Apply edge override if provided
        if min_edge_override:
            self.betting.min_edge = min_edge_override
            log_info(f"⚙️  Min edge threshold set to {min_edge_override*100}%")
        
        # ========================================================================
        # STEP 1: Update Data
        # ========================================================================
        if update_data:
            log_info("📥 STEP 1: Updating data...")
            try:
                results = self.collector.fetch_all_data(
                    days_back=days_back, 
                    days_ahead=days_ahead
                )
                
                if not any(results.values()):
                    log_error("Failed to fetch any data. Cannot proceed.")
                    return False
                    
            except Exception as e:
                log_error(f"Data fetch failed: {e}")
                return False
        else:
            log_info("📥 STEP 1: Skipping data update (using cached data)")
            
        # ========================================================================
        # STEP 2: Process Data
        # ========================================================================
        log_info("\n🔧 STEP 2: Processing data and calculating features...")
        try:
            features_df = self.processor.process_all_data()
            
            if features_df.empty:
                log_error("No features generated. Check your data files.")
                return False
            
            log_info(f"✅ Processed {len(features_df)} games with {len(features_df.columns)} features")
            
        except Exception as e:
            log_error(f"Data processing failed: {e}")
            return False
            
        # ========================================================================
        # STEP 3: Train Model
        # ========================================================================
        if retrain_model:
            log_info(f"\n🤖 STEP 3: Training model on {len(features_df)} games...")
            try:
                model_results = create_enhanced_model(
                    features_df, 
                    target_column='home_won',
                    tune_hyperparameters=tune_hyperparameters
                )
                
                if not model_results:
                    log_error("Model training failed")
                    return False
                
                # Display training metrics
                log_info(f"\n📊 Training Results:")
                log_info(f"   • CV Accuracy: {model_results['cv_mean']:.4f} (±{model_results['cv_std']:.4f})")
                
                # Access metrics from the right place
                evaluation = model_results.get('evaluation', {})
                test_acc = model_results.get('test_accuracy', evaluation.get('test_accuracy', 0))
                
                log_info(f"   • Test Accuracy: {test_acc:.4f}")
                log_info(f"   • ROC-AUC: {model_results.get('roc_auc', evaluation.get('roc_auc', 0)):.4f}")
                log_info(f"   • F1 Score: {model_results.get('f1_score', evaluation.get('f1_score', 0)):.4f}")
                
                # Show walk-forward if available
                if 'walk_forward_accuracy' in model_results:
                    log_info(f"   • Walk-Forward Accuracy: {model_results['walk_forward_accuracy']:.4f}")
                
                # REALITY CHECK
                cv_acc = model_results['cv_mean']
                if abs(test_acc - cv_acc) > 0.10:
                    log_warning("⚠️  WARNING: Large gap between CV and test accuracy!")
                    log_warning(f"   CV: {cv_acc:.1%}, Test: {test_acc:.1%}")
                    log_warning("   This may indicate overfitting or data issues")
                
                # Show expected edge
                if cv_acc > 0.524:
                    expected_edge = (cv_acc - 0.5238) * 100
                    log_info(f"\n💰 Expected Edge: ~{expected_edge:.1f}% (if calibration holds)")
                    log_info(f"   (Accuracy {cv_acc:.1%} vs Breakeven 52.38%)")
                else:
                    log_warning("\n⚠️  WARNING: Accuracy below breakeven threshold!")
                    log_warning("   Not recommended for betting without improvements")
                
            except Exception as e:
                log_error(f"Model training failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            log_info("\n🤖 STEP 3: Skipping model training (using existing model)")
            log_warning("Loading existing models not yet implemented")
            return False

        # ========================================================================
        # STEP 4: Optional Backtest
        # ========================================================================
        if run_backtest and model_results:
            log_info("\n📈 STEP 4: Running backtest...")
            try:
                backtest_results = self.trainer.backtest_model(
                    features_df, 
                    model_results['model'], 
                    model_results['scaler'], 
                    model_results['feature_names'],
                    n_splits=10
                )
                
                if backtest_results:
                    log_info("\n✅ Backtest completed successfully")
                
            except Exception as e:
                log_error(f"Backtesting failed: {e}")
        else:
            log_info("\n📈 STEP 4: Skipping backtest")

        # ========================================================================
        # STEP 5: Predict Upcoming Games with Smart Betting
        # ========================================================================
        log_info("\n🔮 STEP 5: Generating predictions for upcoming games...")
        try:
            upcoming_features = self.processor.get_upcoming_games_features()
            
            if upcoming_features.empty:
                log_warning("No upcoming games found to predict.")
                log_info("\n✅ Pipeline completed (no predictions to make)")
                return True
            
            preds = predict_upcoming_games_enhanced(
                model_results['model'], 
                model_results['scaler'], 
                upcoming_features, 
                model_results['feature_names']
            )
            
            log_info(f"✅ Generated predictions for {len(preds)} games")
            
            # Display predictions with professional betting strategy
            self.display_professional_betting_advice(preds, max_bets_per_day)
            
        except Exception as e:
            log_error(f"Prediction generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        return True

    def display_professional_betting_advice(self, df, max_bets_per_day: int = 5):
        """Display predictions with professional gambling strategy"""
        
        print("\n" + "="*80)
        print("🎰 PROFESSIONAL GAMBLING RECOMMENDATIONS")
        print("="*80)
        print(f"\n💰 Starting Bankroll: ${self.betting.current_bankroll:,.2f}")
        print(f"📊 Min Edge Threshold: {self.betting.min_edge*100}%")
        print(f"🎯 Kelly Fraction: {self.betting.kelly_fraction} (Conservative 1/4 Kelly)")
        print(f"🔒 Max Bet Per Game: {self.betting.max_bet_pct*100}% of bankroll")
        print(f"📅 Max Bets Per Day: {max_bets_per_day}")
        
        # ====================================================================
        # SMART BET FILTERING & RANKING
        # ====================================================================
        
        potential_bets = []
        
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            home_prob = row['home_win_probability']
            away_prob = row['away_win_probability']
            game_date = row.get('game_date', 'TBD')
            
            # Determine predicted winner
            if row['predicted_home_win']:
                predicted_winner = home
                predicted_winner_prob = home_prob
            else:
                predicted_winner = away
                predicted_winner_prob = away_prob
            
            # Calculate bet recommendation
            bet_rec = self.betting.calculate_bet_size(
                win_probability=predicted_winner_prob,
                american_odds=-110,
                confidence_level=row['confidence_level']
            )
            
            if bet_rec:
                # Calculate value score (combines edge and confidence)
                value_score = bet_rec['edge'] * (predicted_winner_prob ** 0.5)
                
                potential_bets.append({
                    'game_date': game_date,
                    'home_team': home,
                    'away_team': away,
                    'predicted_winner': predicted_winner,
                    'win_prob': predicted_winner_prob,
                    'confidence': row['confidence'],
                    'confidence_level': row['confidence_level'],
                    'bet_rec': bet_rec,
                    'value_score': value_score,
                    'net_rating_diff': row.get('net_rating_home', 0) - row.get('net_rating_away', 0),
                    'back_to_back_home': row.get('back_to_back_home', 0),
                    'back_to_back_away': row.get('back_to_back_away', 0),
                    'h2h_win_rate': row.get('h2h_win_rate', 0.5),
                    'row': row
                })
        
        if not potential_bets:
            print("\n" + "="*80)
            print("🚫 NO VALUE BETS FOUND")
            print("="*80)
            print(f"\nNo games meet the minimum edge threshold of {self.betting.min_edge*100}%")
            return
        
        # Sort by value score (best bets first)
        potential_bets.sort(key=lambda x: x['value_score'], reverse=True)
        
        # ====================================================================
        # TIER-BASED BET CLASSIFICATION
        # ====================================================================
        
        tier_1_bets = []  # Edge > 20%, High confidence
        tier_2_bets = []  # Edge > 15%, Medium+ confidence
        tier_3_bets = []  # Edge > 10%, Any confidence
        tier_4_bets = []  # Edge > 5%, Low edge plays
        
        for bet in potential_bets:
            edge = bet['bet_rec']['edge']
            confidence = bet['confidence_level']
            
            if edge >= 20 and confidence == 'High':
                tier_1_bets.append(bet)
            elif edge >= 15 and confidence in ['High', 'Medium']:
                tier_2_bets.append(bet)
            elif edge >= 10:
                tier_3_bets.append(bet)
            else:
                tier_4_bets.append(bet)
        
        # ====================================================================
        # SMART BET SELECTION WITH BANKROLL MANAGEMENT
        # ====================================================================
        
        print("\n" + "="*80)
        print("💎 RECOMMENDED BETS (SMART SELECTION)")
        print("="*80)
        
        total_allocated = 0.0
        recommended_bets = []
        remaining_bankroll = self.betting.current_bankroll
        
        # Priority 1: All Tier 1 bets (if within limits)
        for bet in tier_1_bets[:max_bets_per_day]:
            bet_amount = bet['bet_rec']['bet_amount']
            if remaining_bankroll >= bet_amount:
                recommended_bets.append(('TIER 1 🔥', bet))
                remaining_bankroll -= bet_amount
                total_allocated += bet_amount
        
        # Priority 2: Fill remaining slots with Tier 2
        remaining_slots = max_bets_per_day - len(recommended_bets)
        for bet in tier_2_bets[:remaining_slots]:
            bet_amount = bet['bet_rec']['bet_amount']
            if remaining_bankroll >= bet_amount:
                recommended_bets.append(('TIER 2 ✅', bet))
                remaining_bankroll -= bet_amount
                total_allocated += bet_amount
        
        # Priority 3: If still have slots, add best Tier 3
        remaining_slots = max_bets_per_day - len(recommended_bets)
        for bet in tier_3_bets[:remaining_slots]:
            bet_amount = bet['bet_rec']['bet_amount']
            if remaining_bankroll >= bet_amount:
                recommended_bets.append(('TIER 3 ⚠️', bet))
                remaining_bankroll -= bet_amount
                total_allocated += bet_amount
        
        # ====================================================================
        # DISPLAY SELECTED BETS AND RECORD TO TRACKER
        # ====================================================================
        
        if not recommended_bets:
            print("\n⚠️  Insufficient bankroll for any bets!")
            return
        
        print(f"\n📅 Selected {len(recommended_bets)} of {len(potential_bets)} potential bets")
        print(f"💰 Total Allocated: ${total_allocated:.2f} ({total_allocated/self.betting.current_bankroll*100:.1f}% of bankroll)")
        print(f"🔒 Reserved: ${remaining_bankroll:.2f}\n")
        
        for tier_label, bet in recommended_bets:
            print("-" * 80)
            print(f"\n{tier_label} | 📅 {bet['game_date']}")
            print(f"   {get_team_name(bet['away_team'])} @ {get_team_name(bet['home_team'])}")
            print(f"   🏆 Pick: {get_team_name(bet['predicted_winner'])} ({bet['win_prob']:.1%})")
            print(f"   📊 Confidence: {bet['confidence']:.1%} ({bet['confidence_level']})")
            
            rec = bet['bet_rec']
            print(f"\n   💰 BET RECOMMENDATION:")
            print(f"      • Amount: ${rec['bet_amount']:,.2f} ({rec['pct_of_bankroll']}%)")
            print(f"      • Edge: {rec['edge']:.1f}%")
            print(f"      • Expected Profit: ${rec['expected_profit']:,.2f}")
            print(f"      • Value Score: {bet['value_score']:.2f}")

            # ADDED: Record prediction for tracking
            try:
                self.tracker.record_prediction(
                    game_date=bet['game_date'],
                    home_team=bet['home_team'],
                    away_team=bet['away_team'],
                    predicted_winner=bet['predicted_winner'],
                    win_probability=bet['win_prob'],
                    confidence_level=bet['confidence_level'],
                    edge=bet['bet_rec']['edge'] / 100,  # Convert to decimal
                    bet_amount=bet['bet_rec']['bet_amount'],
                    odds=-110
                )
            except Exception as e:
                log_warning(f"Failed to track prediction: {e}")

        # ====================================================================
        # EXPECTED VALUE ANALYSIS
        # ====================================================================
        
        if recommended_bets:
            print("\n" + "="*80)
            print("📊 EXPECTED VALUE ANALYSIS")
            print("="*80)
            
            total_ev = sum(bet['bet_rec']['expected_profit'] for _, bet in recommended_bets)
            total_risk = sum(bet['bet_rec']['bet_amount'] for _, bet in recommended_bets)
            avg_edge = np.mean([bet['bet_rec']['edge'] for _, bet in recommended_bets])
            
            print(f"\nTotal Expected Value: ${total_ev:.2f}")
            print(f"Total Risk (if all lose): ${total_risk:.2f}")
            print(f"Average Edge: {avg_edge:.1f}%")
            if total_risk > 0:
                print(f"Risk-Reward Ratio: {total_ev/total_risk:.2%}")
            
            # Simple simulation: what if all win? what if all lose?
            all_win_profit = sum(
                bet['bet_rec']['bet_amount'] * (100 / 110) if bet['win_prob'] > 0.5 else 0
                for _, bet in recommended_bets
            )
            all_lose_loss = -total_risk
            
            print(f"\nScenario Analysis:")
            print(f"   Best case (all win): +${all_win_profit:.2f}")
            print(f"   Worst case (all lose): ${all_lose_loss:.2f}")
            print(f"   Expected outcome: +${total_ev:.2f}")
            
            # Reality check
            if avg_edge > 15:
                print(f"\n⚠️  CAUTION: Average edge of {avg_edge:.1f}% is very high")
                print(f"   Professional bettors typically find 2-5% edges")
                print(f"   Verify model calibration before betting")
            elif avg_edge > 10:
                print(f"\n⚠️  High edges detected ({avg_edge:.1f}%)")
                print(f"   This is above typical professional benchmarks")
            else:
                print(f"\n✅ Edge range ({avg_edge:.1f}%) is within professional norms")

        # ====================================================================
        # ADDED: HISTORICAL PERFORMANCE REPORT
        # ====================================================================
        print("\n" + "="*80)
        print("📊 HISTORICAL PERFORMANCE")
        print("="*80)
        print(self.tracker.get_performance_report())
        
        # ====================================================================
        # PRO TIPS
        # ====================================================================
        print("\n" + "="*80)
        print("💡 PROFESSIONAL GAMBLING TIPS:")
        print("   1. Only bet on Tier 1 & 2 for consistent profits")
        print("   2. Track results daily - adjust if win rate drops")
        print("   3. Never chase losses - stick to the system")
        print("   4. Shop for best odds (we assume -110 standard)")
        print("="*80 + "\n")


def main():
    """Main entry point with professional gambling options"""
    parser = argparse.ArgumentParser(
        description='NBA Prediction System with Professional Gambling Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Professional Gambling Examples:
  python main.py                              # Full pipeline with defaults
  python main.py --no-update                  # Skip data download
  python main.py --max-bets 3                 # Only top 3 bets
  python main.py --min-edge 0.05              # 5% minimum edge
  python main.py predict --max-bets 3         # Quick daily picks
  python main.py --backtest                   # Run full validation
        """
    )
    
    # Command line arguments
    parser.add_argument('--no-update', action='store_true', 
                        help='Skip data download (use cached data)')
    parser.add_argument('--tune', action='store_true', 
                        help='Enable hyperparameter tuning (slower but better)')
    parser.add_argument('--backtest', action='store_true', 
                        help='Run backtesting on historical data')
    parser.add_argument('--bankroll', type=float, default=None,
                        help='Starting bankroll amount (default: $1000)')
    parser.add_argument('--days-back', type=int, default=60,
                        help='Days of current season data to fetch (default: 60)')
    parser.add_argument('--days-ahead', type=int, default=7,
                        help='Days of upcoming games to fetch (default: 7)')
    parser.add_argument('--max-bets', type=int, default=5,
                        help='Maximum bets per day (default: 5, recommended: 3-5)')
    parser.add_argument('--min-edge', type=float, default=None,
                        help='Minimum edge threshold (e.g., 0.05 for 5%%, default: 3%%)')
    parser.add_argument('mode', nargs='?', default='full', 
                        choices=['full', 'predict'], 
                        help='Operation mode: full pipeline or predict only')
    
    args = parser.parse_args()
    
    # Display startup banner
    print("\n" + "="*80)
    print("🏀 NBA PREDICTION SYSTEM v2.0 (FIXED & VALIDATED)")
    print("="*80)
    print("\n✅ Key Improvements:")
    print("   • Proper train/test split (no data leakage)")
    print("   • Realistic edge calculation (accounts for vig)")
    print("   • Walk-forward validation (real-world simulation)")
    print("   • Expected Value Analysis")
    print("   • Automated Performance Tracking")
    print("\n" + "="*80 + "\n")
    
    # Initialize system
    system = NBAPredictionSystem(bankroll=args.bankroll)
    
    try:
        if args.mode == 'predict':
            # Quick prediction mode (no data update)
            log_info("🚀 Quick Prediction Mode (using cached data)")
            success = system.full_pipeline(
                update_data=False,
                retrain_model=True,
                run_backtest=False,
                max_bets_per_day=args.max_bets,
                min_edge_override=args.min_edge
            )
        else:
            # Full pipeline mode
            success = system.full_pipeline(
                update_data=not args.no_update,
                tune_hyperparameters=args.tune,
                run_backtest=args.backtest,
                days_back=args.days_back,
                days_ahead=args.days_ahead,
                max_bets_per_day=args.max_bets,
                min_edge_override=args.min_edge
            )
        
        if success:
            # Final summary
            print("\n" + "="*80)
            print("✅ SYSTEM STATUS: READY")
            print("="*80)
            print("\n💡 Next Steps:")
            print("   1. Review the predictions above")
            print("   2. Verify edges are realistic (2-8% range)")
            print("   3. Start with small bet sizes (1-2% max)")
            print("   4. Track results with performance_tracker.py")
            print("\n" + "="*80 + "\n")
            sys.exit(0)
        else:
            log_error("Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        log_error(f"System error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()