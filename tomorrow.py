"""
tomorrow.py - Get predictions for TOMORROW's games
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from fetch_data import NBADataCollector
from data_processor import NBADataProcessor
from model_utils import predict_upcoming_games_enhanced
from betting_strategy import KellyBetting
from nbautils import log_info, get_team_name
from config import KELLY_CONFIG

class TomorrowPredictor:
    def __init__(self, bankroll: float = 1000):
        self.collector = NBADataCollector()
        self.processor = NBADataProcessor()
        self.betting = KellyBetting(bankroll=bankroll)
        
    def get_tomorrows_predictions(self, max_bets: int = 5, min_edge: float = 0.05):
        """Get predictions for tomorrow only"""
        
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        
        print("\n" + "="*80)
        print(f"📅 TOMORROW'S NBA PREDICTIONS - {tomorrow.strftime('%A, %B %d, %Y')}")
        print("="*80)
        
        # Load model
        models_dir = Path('models')
        model_files = list(models_dir.glob('nba_model_*.pkl'))
        
        if not model_files:
            print("\n❌ No trained model found. Run this first:")
            print("   python main.py --days-back 30")
            return
        
        # Get the latest model file
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
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
            return
        
        # Get upcoming games
        print("📥 Loading upcoming games...")
        upcoming_features = self.processor.get_upcoming_games_features()
        
        if upcoming_features.empty:
            print("\n❌ No upcoming games found")
            print("\n💡 Fetch upcoming games first:")
            print("   python main.py --days-ahead 7")
            return
        
        # Filter for tomorrow
        upcoming_features['game_date'] = pd.to_datetime(upcoming_features['game_date']).dt.date
        tomorrow_games = upcoming_features[upcoming_features['game_date'] == tomorrow].copy()
        
        if tomorrow_games.empty:
            # Show next available games
            future_games = upcoming_features[upcoming_features['game_date'] > tomorrow]
            if not future_games.empty:
                next_game_date = future_games['game_date'].min()
                next_games_count = len(future_games[future_games['game_date'] == next_game_date])
                
                print(f"\n📭 NO GAMES SCHEDULED FOR TOMORROW")
                print("="*80)
                print(f"\n📅 Next games: {next_game_date.strftime('%A, %B %d, %Y')}")
                print(f"   ({next_games_count} games)")
            else:
                print(f"\n📭 NO GAMES FOUND AFTER TOMORROW")
                print("\n💡 Fetch more upcoming games:")
                print("   python main.py --days-ahead 7")
            return
        
        # Make predictions
        print(f"🎯 Analyzing {len(tomorrow_games)} games...")
        preds = predict_upcoming_games_enhanced(model, scaler, tomorrow_games, features)
        
        print(f"\n🏀 Found {len(preds)} games tomorrow")
        print(f"💰 Bankroll: ${self.betting.current_bankroll:,.2f}")
        print(f"📊 Min Edge: {min_edge*100}%")
        print(f"🎯 Max Bets: {max_bets}")
        
        # Apply min edge override
        self.betting.min_edge = min_edge
        
        # Analyze bets
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
            
            # Store all games info
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
                    'confidence': row['confidence'],
                    'confidence_level': row['confidence_level'],
                    'bet_rec': bet_rec
                })
        
        if not potential_bets:
            print(f"\n🚫 No games meet {min_edge*100}% edge threshold tomorrow")
            print("\nAll Tomorrow's Games (Low/No Betting Value):")
            print("-" * 80)
            for game in all_games:
                print(f"\n   🏀 {game['away_team']} @ {game['home_team']}")
                print(f"      Pick: {game['predicted_winner']} ({game['win_prob']:.1%})")
                print(f"      Confidence: {game['confidence_level']}")
            print("\n💡 Lower min-edge to see picks:")
            print("   python tomorrow.py --min-edge 0.03")
            return
        
        # Sort by value
        potential_bets.sort(key=lambda x: x['bet_rec']['edge'], reverse=True)
        
        # Limit to max_bets
        selected_bets = potential_bets[:max_bets]
        
        print(f"\n💎 {len(selected_bets)} RECOMMENDED BETS FOR TOMORROW")
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
        
        # Show other games without betting value
        other_games = [g for g in all_games if g['predicted_winner'] not in 
                      [b['predicted_winner'] for b in selected_bets]]
        
        if other_games:
            print(f"\n\n📋 OTHER GAMES (No Betting Value):")
            print("-" * 80)
            for game in other_games:
                print(f"\n   🏀 {game['away_team']} @ {game['home_team']}")
                print(f"      Pick: {game['predicted_winner']} ({game['win_prob']:.1%})")
                print(f"      Confidence: {game['confidence_level']}")
        
        print("\n" + "="*80)
        print("📊 TOMORROW'S SUMMARY")
        print("="*80)
        print(f"Total Games: {len(all_games)}")
        print(f"Betting Opportunities: {len(selected_bets)}")
        print(f"Total Risk: ${total_risk:.2f}")
        print(f"Total Expected Value: ${total_ev:.2f}")
        if total_risk > 0:
            print(f"Risk-Reward: {(total_ev/total_risk)*100:.1f}%")
        print("="*80 + "\n")
        
        print("💡 NEXT STEPS:")
        print("   1. Review picks above")
        print("   2. Check odds on your sportsbook")
        print("   3. Set reminders for game times")
        print("   4. Track results: python update_results.py --days 1")
        print("\n⚠️  REMEMBER: Claimed edges are likely 3-5x too high!")
        print("   Bet conservatively or paper trade first.\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Get NBA predictions for TOMORROW only')
    parser.add_argument('--bankroll', type=float, default=1000,
                       help='Starting bankroll (default: $1000)')
    parser.add_argument('--max-bets', type=int, default=5,
                       help='Max bets for tomorrow (default: 5)')
    parser.add_argument('--min-edge', type=float, default=0.05,
                       help='Minimum edge threshold (default: 0.05 = 5%%)')
    
    args = parser.parse_args()
    
    predictor = TomorrowPredictor(bankroll=args.bankroll)
    predictor.get_tomorrows_predictions(
        max_bets=args.max_bets,
        min_edge=args.min_edge
    )

if __name__ == "__main__":
    main()