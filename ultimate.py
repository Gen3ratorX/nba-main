"""
ultimate.py - Complete NBA Betting System
Combines: Moneyline + Totals (Formula) + Totals (ML Model) + Performance
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import joblib

# Import your existing modules
from data_processor import NBADataProcessor
from model_utils import predict_upcoming_games_enhanced
from betting_strategy import KellyBetting
from performance_tracker import PerformanceTracker
from nbautils import log_info, get_team_name

class UltimateNBASystem:
    """Complete NBA betting system with all prediction types"""
    
    def __init__(self, bankroll: float = 1000):
        self.processor = NBADataProcessor()
        self.betting = KellyBetting(bankroll=bankroll)
        self.tracker = PerformanceTracker()
        
    def load_moneyline_model(self):
        """Load moneyline (win/loss) model"""
        models_dir = Path('models')
        model_files = list(models_dir.glob('nba_model_*.pkl'))
        
        if not model_files:
            return None
        
        latest = max(model_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest.stem.replace('nba_model_', '')
        
        try:
            return {
                'model': joblib.load(latest),
                'scaler': joblib.load(models_dir / f'scaler_{timestamp}.pkl'),
                'features': joblib.load(models_dir / f'features_{timestamp}.pkl'),
                'timestamp': timestamp
            }
        except:
            return None
    
    def load_totals_model(self):
        """Load trained totals regression model"""
        models_dir = Path('models')
        rf_files = list(models_dir.glob('totals_rf_*.pkl'))
        gb_files = list(models_dir.glob('totals_gb_*.pkl'))
        
        if not rf_files or not gb_files:
            return None
        
        latest_rf = max(rf_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_rf.stem.replace('totals_rf_', '')
        
        try:
            return {
                'rf': joblib.load(latest_rf),
                'gb': joblib.load(models_dir / f'totals_gb_{timestamp}.pkl'),
                'scaler': joblib.load(models_dir / f'totals_scaler_{timestamp}.pkl'),
                'features': joblib.load(models_dir / f'totals_features_{timestamp}.pkl'),
                'timestamp': timestamp
            }
        except:
            return None
    
    def calculate_totals_formula(self, game_features):
        """Formula-based totals (backup method)"""
        pace_avg = (game_features['pace_home'] + game_features['pace_away']) / 2
        home_expected = (pace_avg / 100) * game_features['off_rating_home']
        away_expected = (pace_avg / 100) * game_features['off_rating_away']
        return home_expected + away_expected, home_expected, away_expected
    
    def predict_totals_ml(self, game_features, totals_model):
        """ML model-based totals (primary method)"""
        # Prepare features
        X = game_features[totals_model['features']].values.reshape(1, -1)
        X_scaled = totals_model['scaler'].transform(X)
        
        # Ensemble prediction
        pred_rf = totals_model['rf'].predict(X_scaled)[0]
        pred_gb = totals_model['gb'].predict(X_scaled)[0]
        pred_avg = (pred_rf + pred_gb) / 2
        
        return pred_avg, pred_rf, pred_gb
    
    def get_complete_analysis(self, target_date=None, max_ml_bets=5, min_edge=0.05, 
                             typical_total=220.0, show_all=False):
        """
        Complete daily analysis with all prediction types
        """
        if target_date is None:
            target_date = datetime.now().date()
        
        day_name = target_date.strftime('%A, %B %d, %Y')
        
        # Header
        print("\n" + "="*80)
        print(f"🏀 ULTIMATE NBA BETTING SYSTEM - {day_name}")
        print("="*80)
        
        # Load models
        print("\n🤖 Loading models...")
        ml_model = self.load_moneyline_model()
        totals_model = self.load_totals_model()
        
        if not ml_model:
            print("❌ Moneyline model not found. Run: python main.py --days-back 30")
            return
        
        print(f"✅ Moneyline model loaded ({ml_model['timestamp']})")
        
        if totals_model:
            print(f"✅ Totals ML model loaded ({totals_model['timestamp']})")
        else:
            print("⚠️  Totals ML model not found (using formula). Run: python train_totals_model.py")
        
        # Get games
        print("📥 Loading games...")
        upcoming_features = self.processor.get_upcoming_games_features()
        
        if upcoming_features.empty:
            print("❌ No upcoming games")
            return
        
        # Filter for target date
        upcoming_features['game_date'] = pd.to_datetime(upcoming_features['game_date']).dt.date
        target_games = upcoming_features[upcoming_features['game_date'] == target_date].copy()
        
        if target_games.empty:
            next_date = upcoming_features['game_date'].min()
            print(f"\n📭 NO GAMES on {day_name}")
            print(f"   Next: {next_date.strftime('%A, %B %d, %Y')}")
            return
        
        print(f"🏀 Analyzing {len(target_games)} games...\n")
        
        # Moneyline predictions
        ml_preds = predict_upcoming_games_enhanced(
            ml_model['model'], 
            ml_model['scaler'], 
            target_games, 
            ml_model['features']
        )
        
        # Analyze each game
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
            
            # Totals (ML model if available, else formula)
            if totals_model:
                try:
                    total_ml, _, _ = self.predict_totals_ml(row, totals_model)
                    total_method = "ML Model"
                except:
                    total_ml, _, _ = self.calculate_totals_formula(row)
                    total_method = "Formula"
            else:
                total_ml, _, _ = self.calculate_totals_formula(row)
                total_method = "Formula"
            
            total_diff = total_ml - typical_total
            
            # Totals recommendation
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
                'pace': (row['pace_home'] + row['pace_away']) / 2,
                'game_data': row
            })
        
        # Display
        self._display_complete_dashboard(
            all_games, max_ml_bets, typical_total, show_all
        )
        
        # Performance
        self._show_performance()
    
    def _display_complete_dashboard(self, games, max_ml_bets, typical_total, show_all):
        """Display unified dashboard"""
        
        # Categorize games
        ml_value = [g for g in games if g['bet_rec'] is not None]
        total_value = [g for g in games if g['total_value']]
        confluence = [g for g in games if g['bet_rec'] is not None and g['total_value']]
        
        # Section 1: Best Plays (Confluence)
        if confluence:
            print("\n" + "="*80)
            print("⭐ BEST PLAYS - CONFLUENCE (Moneyline + Totals Agree)")
            print("="*80)
            print("🔥 These have BOTH moneyline AND totals value!\n")
            
            for game in confluence:
                rec = game['bet_rec']
                print(f"💎 {game['away_team']} @ {game['home_team']}")
                print(f"   Moneyline: {game['ml_pick']} ({game['ml_prob']:.1%}) - ${rec['bet_amount']:.0f}")
                print(f"   Totals: {game['total_rec']} ({game['total_pred']:.1f})")
                print(f"   💰 Double value opportunity!")
                print()
        
        # Section 2: Moneyline Only
        ml_only = [g for g in ml_value if not g['total_value']]
        if ml_only:
            ml_only.sort(key=lambda x: x['bet_rec']['edge'], reverse=True)
            selected = ml_only[:max_ml_bets]
            
            print("\n" + "="*80)
            print("💎 MONEYLINE VALUE PLAYS")
            print("="*80)
            
            for i, game in enumerate(selected, 1):
                rec = game['bet_rec']
                print(f"\n🏀 #{i}: {game['away_team']} @ {game['home_team']}")
                print(f"   Pick: {game['ml_pick']} ({game['ml_prob']:.1%})")
                print(f"   Confidence: {game['confidence_level']}")
                print(f"   Bet: ${rec['bet_amount']:.2f} | Edge: {rec['edge']:.1f}%")
        
        # Section 3: Totals Only
        total_only = [g for g in total_value if g['bet_rec'] is None]
        if total_only:
            print("\n" + "="*80)
            print("📊 TOTALS VALUE PLAYS")
            print("="*80)
            print(f"⚠️  Check actual lines (using {typical_total} as reference)\n")
            
            for game in total_only:
                print(f"{game['total_emoji']} {game['away_team']} @ {game['home_team']}")
                print(f"   Predicted: {game['total_pred']:.1f} | Diff: {game['total_diff']:+.1f}")
                print(f"   Play: {game['total_rec']} ({game['total_method']})")
                print()
        
        # Section 4: All Games Summary
        print("\n" + "="*80)
        print("📋 ALL GAMES TODAY")
        print("="*80)
        
        for game in games:
            ml_icon = "✅" if game['bet_rec'] else "⚪"
            total_icon = "📊" if game['total_value'] else "⚪"
            
            print(f"\n{ml_icon}{total_icon} {game['away_team']} @ {game['home_team']}")
            print(f"   ML: {game['ml_pick']} ({game['ml_prob']:.1%})")
            print(f"   Total: {game['total_pred']:.1f} ({game['total_rec']})")
        
        # Summary
        print("\n" + "="*80)
        print("📊 DAILY SUMMARY")
        print("="*80)
        print(f"Total Games: {len(games)}")
        print(f"Confluence Plays: {len(confluence)} 🔥")
        print(f"Moneyline Value: {len(ml_value)}")
        print(f"Totals Value: {len(total_value)}")
        print("="*80)
    
    def _show_performance(self):
        """Quick performance stats"""
        stats = self.tracker.get_overall_stats()
        
        if stats['completed'] > 0:
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
    parser = argparse.ArgumentParser(
        description='Ultimate NBA Betting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete system combining:
  • Moneyline predictions (win/loss)
  • Totals predictions (over/under)
  • ML-trained totals model
  • Performance tracking

Examples:
  python ultimate.py                  # Today's complete analysis
  python ultimate.py --tomorrow       # Tomorrow's analysis
  python ultimate.py --show-all       # All games
  python ultimate.py --min-edge 0.08  # Conservative picks
        """
    )
    
    parser.add_argument('--bankroll', type=float, default=1000)
    parser.add_argument('--max-bets', type=int, default=5,
                       help='Max moneyline bets')
    parser.add_argument('--min-edge', type=float, default=0.05,
                       help='Min edge for moneyline (5%% default)')
    parser.add_argument('--typical-total', type=float, default=220.0,
                       help='Typical O/U line for comparison')
    parser.add_argument('--tomorrow', action='store_true')
    parser.add_argument('--show-all', action='store_true')
    
    args = parser.parse_args()
    
    system = UltimateNBASystem(bankroll=args.bankroll)
    
    target_date = (datetime.now() + timedelta(days=1)).date() if args.tomorrow else datetime.now().date()
    
    system.get_complete_analysis(
        target_date=target_date,
        max_ml_bets=args.max_bets,
        min_edge=args.min_edge,
        typical_total=args.typical_total,
        show_all=args.show_all
    )
    
    print("\n💡 WORKFLOW:")
    print("   1. Review confluence plays first (both ML + totals)")
    print("   2. Check actual sportsbook lines")
    print("   3. Paper trade or bet conservatively")
    print("   4. Track: python update_results.py --days 1")
    print("\n⚠️  Remember: Edges are likely overestimated 3-5x!\n")

if __name__ == "__main__":
    main()