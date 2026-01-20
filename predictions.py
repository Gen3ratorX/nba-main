"""
predictions.py - Unified NBA Predictions (replaces ultimate.py, today.py, tomorrow.py)
Complete system: Moneyline + Totals + Confluence analysis
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_processor import NBADataProcessor
from model_utils import predict_upcoming_games_enhanced
from betting_strategy import KellyBetting
from performance_tracker import PerformanceTracker
from nbautils import log_info, get_team_name
from model_loader import load_moneyline_model, load_totals_model

class UnifiedPredictor:
    """Unified predictor supporting moneyline, totals, and combined analysis"""
    
    def __init__(self, bankroll: float = 1000):
        self.processor = NBADataProcessor()
        self.betting = KellyBetting(bankroll=bankroll)
        self.tracker = PerformanceTracker()
        self.totals_model = load_totals_model()
    
    def calculate_totals_formula(self, game_features):
        """Formula-based totals (backup method)"""
        pace_avg = (game_features['pace_home'] + game_features['pace_away']) / 2
        home_expected = (pace_avg / 100) * game_features['off_rating_home']
        away_expected = (pace_avg / 100) * game_features['off_rating_away']
        return home_expected + away_expected, home_expected, away_expected
    
    def predict_totals_ml(self, game_features):
        """ML model-based totals (primary method if model exists)"""
        if not self.totals_model:
            return None
        
        try:
            X = pd.DataFrame([game_features[self.totals_model['features']]], 
                           columns=self.totals_model['features'])
            X_scaled = self.totals_model['scaler'].transform(X)
            pred_rf = self.totals_model['rf'].predict(X_scaled)[0]
            pred_gb = self.totals_model['gb'].predict(X_scaled)[0]
            return (pred_rf + pred_gb) / 2
        except Exception as e:
            log_info(f"Error in ML totals prediction: {e}")
            return None
    
    def get_predictions(self, target_date=None, max_bets: int = 5, min_edge: float = 0.05,
                       prediction_type: str = 'all', typical_total: float = 220.0, 
                       show_all: bool = False):
        """
        Get predictions for target date
        
        Args:
            target_date: Date object or None for today
            max_bets: Maximum number of bets to recommend
            min_edge: Minimum edge threshold
            prediction_type: 'moneyline', 'totals', or 'all' (default)
            typical_total: Typical O/U line for totals comparison
            show_all: Show all games, not just value plays
        """
        if target_date is None:
            target_date = datetime.now().date()
        
        date_str = target_date.strftime('%A, %B %d, %Y')
        
        # Header
        if prediction_type == 'all':
            print("\n" + "="*80)
            print(f"🏀 NBA BETTING SYSTEM - {date_str}")
            print("="*80)
        elif prediction_type == 'moneyline':
            print("\n" + "="*80)
            print(f"📅 NBA MONEYLINE PREDICTIONS - {date_str}")
            print("="*80)
        else:
            print("\n" + "="*80)
            print(f"📊 NBA TOTALS PREDICTIONS - {date_str}")
            print("="*80)
        
        # Load models
        ml_model = None
        if prediction_type in ['moneyline', 'all']:
            ml_model = load_moneyline_model()
            if not ml_model:
                print("\n❌ Moneyline model not found. Run: python main.py --days-back 30")
                return
        
        if prediction_type in ['totals', 'all']:
            if self.totals_model:
                print(f"✅ Totals ML model loaded ({self.totals_model['timestamp']})")
            else:
                print("⚠️  Totals ML model not found (using formula). Run: python train_totals_model.py")
        
        # Get upcoming games
        print("📥 Loading games...")
        upcoming_features = self.processor.get_upcoming_games_features()
        
        if upcoming_features.empty:
            print("\n❌ No upcoming games found")
            print("\n💡 Fetch upcoming games first:")
            print("   python main.py --days-ahead 7")
            return
        
        # Filter for target date
        upcoming_features['game_date'] = pd.to_datetime(upcoming_features['game_date']).dt.date
        target_games = upcoming_features[upcoming_features['game_date'] == target_date].copy()
        
        if target_games.empty:
            future_games = upcoming_features[upcoming_features['game_date'] > target_date]
            if not future_games.empty:
                next_game_date = future_games['game_date'].min()
                print(f"\n📭 NO GAMES SCHEDULED FOR {date_str}")
                print(f"   Next games: {next_game_date.strftime('%A, %B %d, %Y')}")
            else:
                print(f"\n📭 NO GAMES FOUND AFTER {date_str}")
            return
        
        print(f"🏀 Analyzing {len(target_games)} games...")
        
        # Handle different prediction types
        if prediction_type == 'moneyline':
            self._get_moneyline_only(target_games, ml_model, max_bets, min_edge)
        elif prediction_type == 'totals':
            self._get_totals_only(target_games, typical_total)
        else:  # 'all'
            self._get_complete_analysis(target_games, ml_model, max_bets, min_edge, 
                                      typical_total, show_all)
    
    def _get_moneyline_only(self, target_games, ml_model, max_bets, min_edge):
        """Moneyline-only predictions"""
        preds = predict_upcoming_games_enhanced(
            ml_model['model'], ml_model['scaler'], target_games, ml_model['features']
        )
        
        print(f"\n💰 Bankroll: ${self.betting.current_bankroll:,.2f}")
        print(f"📊 Min Edge: {min_edge*100}%")
        print(f"🎯 Max Bets: {max_bets}")
        
        self.betting.min_edge = min_edge
        potential_bets = []
        all_games = []
        
        for idx, row in preds.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            if row['predicted_home_win']:
                predicted_winner = home
                win_prob = row['home_win_probability']
            else:
                predicted_winner = away
                win_prob = row['away_win_probability']
            
            all_games.append({
                'home_team': home,
                'away_team': away,
                'predicted_winner': predicted_winner,
                'win_prob': win_prob,
                'confidence_level': row['confidence_level']
            })
            
            bet_rec = self.betting.calculate_bet_size(
                win_probability=win_prob,
                american_odds=-110,
                confidence_level=row['confidence_level']
            )
            
            if bet_rec:
                potential_bets.append({
                    'home_team': home,
                    'away_team': away,
                    'predicted_winner': predicted_winner,
                    'win_prob': win_prob,
                    'confidence_level': row['confidence_level'],
                    'bet_rec': bet_rec
                })
        
        if not potential_bets:
            print(f"\n🚫 No games meet {min_edge*100}% edge threshold")
            print("\nAll Games (No Betting Value):")
            print("-" * 80)
            for game in all_games:
                print(f"\n   🏀 {game['home_team']} @ {game['away_team']}")
                print(f"      Pick: {game['predicted_winner']} ({game['win_prob']:.1%})")
                print(f"      Confidence: {game['confidence_level']}")
            return
        
        potential_bets.sort(key=lambda x: x['bet_rec']['edge'], reverse=True)
        selected_bets = potential_bets[:max_bets]
        
        print(f"\n💎 {len(selected_bets)} RECOMMENDED BETS")
        print("="*80)
        
        for i, bet in enumerate(selected_bets, 1):
            rec = bet['bet_rec']
            print(f"\n🏀 BET #{i}")
            print("-" * 80)
            print(f"   {bet['home_team']} @ {bet['away_team']}")
            print(f"   🏆 Pick: {bet['predicted_winner']}")
            print(f"   📊 Win Probability: {bet['win_prob']:.1%}")
            print(f"   ⭐ Confidence: {bet['confidence_level']}")
            print(f"\n   💰 BET RECOMMENDATION:")
            print(f"      Amount: ${rec['bet_amount']:.2f} ({rec['pct_of_bankroll']}%)")
            print(f"      Edge: {rec['edge']:.1f}%")
            print(f"      Expected Profit: ${rec['expected_profit']:.2f}")
            if rec.get('warning'):
                print(f"\n      ⚠️  {rec['warning']}")
        
        total_risk = sum(b['bet_rec']['bet_amount'] for b in selected_bets)
        total_ev = sum(b['bet_rec']['expected_profit'] for b in selected_bets)
        
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(f"Total Risk: ${total_risk:.2f}")
        print(f"Total Expected Value: ${total_ev:.2f}")
        if total_risk > 0:
            print(f"Risk-Reward: {(total_ev/total_risk)*100:.1f}%")
        print("="*80 + "\n")
    
    def _get_totals_only(self, target_games, typical_total):
        """Totals-only predictions"""
        predictions = []
        
        for idx, game in target_games.iterrows():
            home = game['home_team']
            away = game['away_team']
            
            total_ml = self.predict_totals_ml(game)
            if total_ml is None:
                total_ml, _, _ = self.calculate_totals_formula(game)
                total_method = "Formula"
            else:
                total_method = "ML Model"
            
            total_diff = total_ml - typical_total
            
            if total_diff > 8:
                rec = "🔥 STRONG OVER"
                emoji = "📈"
            elif total_diff > 3:
                rec = "📈 LEAN OVER"
                emoji = "⬆️"
            elif total_diff < -8:
                rec = "❄️ STRONG UNDER"
                emoji = "📉"
            elif total_diff < -3:
                rec = "📉 LEAN UNDER"
                emoji = "⬇️"
            else:
                rec = "⚖️ NO BET (too close)"
                emoji = "⚠️"
            
            predictions.append({
                'home_team': home,
                'away_team': away,
                'total': total_ml,
                'diff': total_diff,
                'rec': rec,
                'emoji': emoji,
                'method': total_method
            })
        
        df = pd.DataFrame(predictions)
        df = df.sort_values('total', ascending=False)
        
        print("="*80)
        for _, pred in df.iterrows():
            print(f"\n{pred['emoji']} {pred['home_team']} @ {pred['away_team']}")
            print(f"   Predicted Total: {pred['total']:.1f} points")
            print(f"   vs Typical Line: {typical_total:.1f}")
            print(f"   Difference: {pred['diff']:+.1f} points")
            print(f"   🎯 Recommendation: {pred['rec']}")
        
        strong = len(df[abs(df['diff']) > 8])
        print("\n" + "="*80)
        print("📊 TOTALS SUMMARY")
        print("="*80)
        print(f"Total Games: {len(df)}")
        print(f"Strong Plays (8+ pts): {strong}")
        if strong > 0:
            print(f"\n💎 {strong} STRONG VALUE PLAYS")
        print("="*80 + "\n")
    
    def _get_complete_analysis(self, target_games, ml_model, max_bets, min_edge, 
                              typical_total, show_all):
        """Complete analysis: Moneyline + Totals + Confluence"""
        ml_preds = predict_upcoming_games_enhanced(
            ml_model['model'], ml_model['scaler'], target_games, ml_model['features']
        )
        
        self.betting.min_edge = min_edge
        all_games = []
        
        for idx, row in ml_preds.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Moneyline
            if row['predicted_home_win']:
                ml_pick = home
                ml_prob = row['home_win_probability']
            else:
                ml_pick = away
                ml_prob = row['away_win_probability']
            
            bet_rec = self.betting.calculate_bet_size(
                win_probability=ml_prob,
                american_odds=-110,
                confidence_level=row['confidence_level']
            )
            
            # Totals
            total_ml = self.predict_totals_ml(row)
            if total_ml is None:
                total_ml, _, _ = self.calculate_totals_formula(row)
                total_method = "Formula"
            else:
                total_method = "ML Model"
            
            total_diff = total_ml - typical_total
            
            if total_diff > 8:
                total_rec = "STRONG OVER"
                total_emoji = "🔥"
                total_value = True
            elif total_diff > 3:
                total_rec = "LEAN OVER"
                total_emoji = "📈"
                total_value = False
            elif total_diff < -8:
                total_rec = "STRONG UNDER"
                total_emoji = "❄️"
                total_value = True
            elif total_diff < -3:
                total_rec = "LEAN UNDER"
                total_emoji = "📉"
                total_value = False
            else:
                total_rec = "NO BET"
                total_emoji = "⚖️"
                total_value = False
            
            all_games.append({
                'home_team': home,
                'away_team': away,
                'ml_pick': ml_pick,
                'ml_prob': ml_prob,
                'confidence_level': row['confidence_level'],
                'bet_rec': bet_rec,
                'total_pred': total_ml,
                'total_diff': total_diff,
                'total_rec': total_rec,
                'total_emoji': total_emoji,
                'total_value': total_value,
                'total_method': total_method,
            })
        
        # Categorize games
        ml_value = [g for g in all_games if g['bet_rec'] is not None]
        total_value = [g for g in all_games if g['total_value']]
        confluence = [g for g in all_games if g['bet_rec'] is not None and g['total_value']]
        
        # Display: Confluence plays first
        if confluence:
            print("\n" + "="*80)
            print("⭐ BEST PLAYS - CONFLUENCE (Moneyline + Totals Agree)")
            print("="*80)
            for game in confluence:
                rec = game['bet_rec']
                print(f"\n💎 {game['home_team']} @ {game['away_team']}")
                print(f"   Moneyline: {game['ml_pick']} ({game['ml_prob']:.1%}) - ${rec['bet_amount']:.0f}")
                print(f"   Totals: {game['total_rec']} ({game['total_pred']:.1f})")
                print(f"   💰 Double value opportunity!")
        
        # Moneyline only
        ml_only = [g for g in ml_value if not g['total_value']]
        if ml_only:
            ml_only.sort(key=lambda x: x['bet_rec']['edge'], reverse=True)
            selected = ml_only[:max_bets]
            
            print("\n" + "="*80)
            print("💎 MONEYLINE VALUE PLAYS")
            print("="*80)
            for i, game in enumerate(selected, 1):
                rec = game['bet_rec']
                print(f"\n🏀 #{i}: {game['home_team']} @ {game['away_team']}")
                print(f"   Pick: {game['ml_pick']} ({game['ml_prob']:.1%})")
                print(f"   Confidence: {game['confidence_level']}")
                print(f"   Bet: ${rec['bet_amount']:.2f} | Edge: {rec['edge']:.1f}%")
        
        # Totals only
        total_only = [g for g in total_value if g['bet_rec'] is None]
        if total_only:
            print("\n" + "="*80)
            print("📊 TOTALS VALUE PLAYS")
            print("="*80)
            print(f"⚠️  Check actual lines (using {typical_total} as reference)\n")
            for game in total_only:
                print(f"{game['total_emoji']} {game['home_team']} @ {game['away_team']}")
                print(f"   Predicted: {game['total_pred']:.1f} | Diff: {game['total_diff']:+.1f}")
                print(f"   Play: {game['total_rec']} ({game['total_method']})")
                print()
        
        # All games summary
        if show_all:
            print("\n" + "="*80)
            print("📋 ALL GAMES TODAY")
            print("="*80)
            for game in all_games:
                ml_icon = "✅" if game['bet_rec'] else "⚪"
                total_icon = "📊" if game['total_value'] else "⚪"
                print(f"\n{ml_icon}{total_icon} {game['home_team']} @ {game['away_team']}")
                print(f"   ML: {game['ml_pick']} ({game['ml_prob']:.1%})")
                print(f"   Total: {game['total_pred']:.1f} ({game['total_rec']})")
        
        # Summary
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(f"Total Games: {len(all_games)}")
        print(f"Confluence Plays: {len(confluence)} 🔥")
        print(f"Moneyline Value: {len(ml_value)}")
        print(f"Totals Value: {len(total_value)}")
        print("="*80)
        
        # Performance tracking
        self._show_performance()
    
    def _show_performance(self):
        """Quick performance stats"""
        stats = self.tracker.get_overall_stats()
        if stats.get('completed', 0) > 0:
            print("\n" + "="*80)
            print("📈 TRACKED PERFORMANCE")
            print("="*80)
            print(f"Bets: {stats['completed']} | Win Rate: {stats['accuracy']:.1%} | ROI: {stats['roi']:.1f}%")
            if stats['accuracy'] < 0.524:
                print("⚠️  Win rate below breakeven - review strategy!")
            elif stats['accuracy'] > 0.58:
                print("✅ Strong performance - keep tracking!")
            print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified NBA Predictions - Moneyline, Totals, or Complete Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predictions.py                    # Complete analysis (default)
  python predictions.py --moneyline        # Moneyline only
  python predictions.py --totals           # Totals only
  python predictions.py --tomorrow         # Tomorrow's complete analysis
  python predictions.py --date 2026-01-25  # Specific date
  python predictions.py --min-edge 0.08    # Conservative picks
        """
    )
    parser.add_argument('--bankroll', type=float, default=1000,
                       help='Starting bankroll (default: $1000)')
    parser.add_argument('--max-bets', type=int, default=5,
                       help='Max bets (default: 5)')
    parser.add_argument('--min-edge', type=float, default=0.05,
                       help='Minimum edge threshold (default: 0.05 = 5%%)')
    parser.add_argument('--moneyline', action='store_true',
                       help='Moneyline predictions only')
    parser.add_argument('--totals', action='store_true',
                       help='Totals predictions only')
    parser.add_argument('--tomorrow', action='store_true',
                       help='Get tomorrow\'s predictions')
    parser.add_argument('--date', type=str, default=None,
                       help='Specific date (YYYY-MM-DD format)')
    parser.add_argument('--typical-total', type=float, default=220.0,
                       help='Typical O/U line for totals (default: 220.0)')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all games, not just value plays')
    
    args = parser.parse_args()
    
    # Determine prediction type
    if args.moneyline:
        pred_type = 'moneyline'
    elif args.totals:
        pred_type = 'totals'
    else:
        pred_type = 'all'  # Default: complete analysis
    
    # Determine target date
    if args.tomorrow:
        target_date = (datetime.now() + timedelta(days=1)).date()
    elif args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("   Use YYYY-MM-DD format (e.g., 2026-01-25)")
            sys.exit(1)
    else:
        target_date = datetime.now().date()
    
    predictor = UnifiedPredictor(bankroll=args.bankroll)
    predictor.get_predictions(
        target_date=target_date,
        max_bets=args.max_bets,
        min_edge=args.min_edge,
        prediction_type=pred_type,
        typical_total=args.typical_total,
        show_all=args.show_all
    )
    
    if pred_type == 'all':
        print("\n💡 WORKFLOW:")
        print("   1. Review confluence plays first (both ML + totals)")
        print("   2. Check actual sportsbook lines")
        print("   3. Paper trade or bet conservatively")
        print("   4. Track: python update_results.py --days 1")
        print("\n⚠️  Remember: Edges are likely overestimated 3-5x!\n")


if __name__ == "__main__":
    main()
