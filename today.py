"""
today.py - Get predictions for TODAY only
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from fetch_data import NBADataCollector
from data_processor import NBADataProcessor
from model_utils import create_enhanced_model, predict_upcoming_games_enhanced
from betting_strategy import KellyBetting
from nbautils import log_info, get_team_name
from config import KELLY_CONFIG

class TodayPredictor:
    def __init__(self, bankroll: float = 1000):
        self.collector = NBADataCollector()
        self.processor = NBADataProcessor()
        self.betting = KellyBetting(bankroll=bankroll)
        
    def get_todays_predictions(self, max_bets: int = 5, min_edge: float = 0.05):
        """Get predictions for today only"""
        
        today = datetime.now().date()
        
        print("\n" + "="*80)
        print(f"📅 TODAY'S NBA PREDICTIONS - {today.strftime('%A, %B %d, %Y')}")
        print("="*80)
        
        # Load model - FIXED to find any timestamp
        models_dir = Path('models')
        model_files = list(models_dir.glob('nba_model_*.pkl'))
        
        if not model_files:
            print("\n❌ No trained model found. Run this first:")
            print("   python main.py --days-back 30")
            return
        
        # Get the latest model file
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        # Extract timestamp from filename (e.g., nba_model_20260115_082914.pkl -> 20260115_082914)
        timestamp = latest_model_file.stem.replace('nba_model_', '')
        
        print(f"🤖 Loading model (timestamp: {timestamp})...")
        
        # Load all components
        import joblib
        try:
            model = joblib.load(latest_model_file)
            scaler = joblib.load(models_dir / f'scaler_{timestamp}.pkl')
            features = joblib.load(models_dir / f'features_{timestamp}.pkl')
            log_info(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"\nAvailable model files:")
            for f in model_files:
                print(f"   {f.name}")
            return
        
        # Get upcoming games
        print("📥 Loading upcoming games...")
        upcoming_features = self.processor.get_upcoming_games_features()
        
        if upcoming_features.empty:
            print("\n❌ No upcoming games found")
            print("\n💡 Fetch upcoming games first:")
            print("   python main.py --days-ahead 7")
            return
        
        # Filter for today
        upcoming_features['game_date'] = pd.to_datetime(upcoming_features['game_date']).dt.date
        today_games = upcoming_features[upcoming_features['game_date'] == today].copy()
        
        if today_games.empty:
            # Show next available games
            next_game_date = upcoming_features['game_date'].min()
            next_games_count = len(upcoming_features[upcoming_features['game_date'] == next_game_date])
            
            print(f"\n📭 NO GAMES SCHEDULED FOR TODAY")
            print("="*80)
            print(f"\n📅 Next games: {next_game_date.strftime('%A, %B %d, %Y')}")
            print(f"   ({next_games_count} games)")
            print("\n💡 To see tomorrow's games, run:")
            print("   python tomorrow.py")
            print("\n💡 To see all upcoming games, run:")
            print("   python main.py predict --max-bets 10")
            return
        
        # Make predictions
        print(f"🎯 Analyzing {len(today_games)} games...")
        preds = predict_upcoming_games_enhanced(model, scaler, today_games, features)
        
        print(f"\n🏀 Found {len(preds)} games today")
        print(f"💰 Bankroll: ${self.betting.current_bankroll:,.2f}")
        print(f"📊 Min Edge: {min_edge*100}%")
        print(f"🎯 Max Bets: {max_bets}")
        
        # Apply min edge override
        self.betting.min_edge = min_edge
        
        # Analyze bets
        potential_bets = []
        
        for idx, row in preds.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            if row['predicted_home_win']:
                predicted_winner = home
                win_prob = row['home_win_probability']
            else:
                predicted_winner = away
                win_prob = row['away_win_probability']
            
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
                    'confidence': row['confidence'],
                    'confidence_level': row['confidence_level'],
                    'bet_rec': bet_rec
                })
        
        if not potential_bets:
            print(f"\n🚫 No games meet {min_edge*100}% edge threshold today")
            print("\nAll Today's Games (No Betting Value):")
            print("-" * 80)
            for idx, row in preds.iterrows():
                print(f"\n   🏀 {row['away_team']} @ {row['home_team']}")
                if row['predicted_home_win']:
                    print(f"      Pick: {row['home_team']} ({row['home_win_probability']:.1%})")
                else:
                    print(f"      Pick: {row['away_team']} ({row['away_win_probability']:.1%})")
                print(f"      Confidence: {row['confidence_level']}")
            print("\n💡 Lower min-edge to see more picks:")
            print("   python today.py --min-edge 0.03")
            return
        
        # Sort by value
        potential_bets.sort(key=lambda x: x['bet_rec']['edge'], reverse=True)
        
        # Limit to max_bets
        selected_bets = potential_bets[:max_bets]
        
        print(f"\n💎 {len(selected_bets)} RECOMMENDED BETS FOR TODAY")
        print("="*80)
        
        total_risk = 0
        total_ev = 0
        
        for i, bet in enumerate(selected_bets, 1):
            rec = bet['bet_rec']
            
            print(f"\n🏀 BET #{i}")
            print("-" * 80)
            print(f"   {bet['away_team']} @ {bet['home_team']}")
            print(f"   🏆 Pick: {bet['predicted_winner']}")
            print(f"   📊 Win Probability: {bet['win_prob']:.1%}")
            print(f"   ⭐ Confidence: {bet['confidence_level']}")
            print(f"\n   💰 BET RECOMMENDATION:")
            print(f"      Amount: ${rec['bet_amount']:.2f} ({rec['pct_of_bankroll']}%)")
            print(f"      Edge: {rec['edge']:.1f}%")
            print(f"      Expected Profit: ${rec['expected_profit']:.2f}")
            
            if rec.get('warning'):
                print(f"\n      ⚠️  {rec['warning']}")
            
            total_risk += rec['bet_amount']
            total_ev += rec['expected_profit']
        
        print("\n" + "="*80)
        print("📊 TODAY'S SUMMARY")
        print("="*80)
        print(f"Total Risk: ${total_risk:.2f}")
        print(f"Total Expected Value: ${total_ev:.2f}")
        if total_risk > 0:
            print(f"Risk-Reward: {(total_ev/total_risk)*100:.1f}%")
        print("="*80 + "\n")
        
        print("💡 NEXT STEPS:")
        print("   1. Review picks above")
        print("   2. Verify games on your sportsbook")
        print("   3. Track results: python update_results.py --days 1")
        print("\n⚠️  REMEMBER: These edges are likely overestimated!")
        print("   Real edges are probably 2-5%, not 20%+")
        print("   Start with paper trading or minimum bets.\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Get NBA predictions for TODAY only')
    parser.add_argument('--bankroll', type=float, default=1000,
                       help='Starting bankroll (default: $1000)')
    parser.add_argument('--max-bets', type=int, default=5,
                       help='Max bets for today (default: 5)')
    parser.add_argument('--min-edge', type=float, default=0.05,
                       help='Minimum edge threshold (default: 0.05 = 5%%)')
    
    args = parser.parse_args()
    
    predictor = TodayPredictor(bankroll=args.bankroll)
    predictor.get_todays_predictions(
        max_bets=args.max_bets,
        min_edge=args.min_edge
    )

if __name__ == "__main__":
    main()