"""
totals_predictor.py - FIXED VERSION
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
from data_processor import NBADataProcessor
from betting_strategy import KellyBetting
from nbautils import log_info, get_team_name

class TotalsPredictor:
    """Predict game totals (Over/Under)"""
    
    def __init__(self, bankroll: float = 1000):
        self.processor = NBADataProcessor()
        self.betting = KellyBetting(bankroll=bankroll)
        
    def calculate_expected_total(self, game_features):
        """
        Calculate expected total points using pace and ratings
        
        FIXED FORMULA:
        Points per team = (Pace / 100) * Offensive Rating
        Total = Home Points + Away Points
        """
        pace_avg = (game_features['pace_home'] + game_features['pace_away']) / 2
        
        # Each team's expected points
        # Formula: Points = (Pace / 100) * Offensive Rating
        home_expected = (pace_avg / 100) * game_features['off_rating_home']
        away_expected = (pace_avg / 100) * game_features['off_rating_away']
        
        # Total expected points
        expected_total = home_expected + away_expected
        
        return expected_total, home_expected, away_expected
    
    def predict_totals_today(self, target_date=None, typical_total=220.0):
        """
        Predict totals for games on target date
        
        Args:
            target_date: Date to predict (default: today)
            typical_total: Typical O/U line in NBA (default: 220)
        """
        if target_date is None:
            target_date = datetime.now().date()
        
        print("\n" + "="*80)
        print(f"📊 NBA TOTALS PREDICTIONS - {target_date.strftime('%A, %B %d, %Y')}")
        print("="*80)
        
        # Get upcoming games with features
        print("📥 Loading game features...")
        upcoming_features = self.processor.get_upcoming_games_features()
        
        if upcoming_features.empty:
            print("\n❌ No upcoming games found")
            return
        
        # Filter for target date
        upcoming_features['game_date'] = pd.to_datetime(upcoming_features['game_date']).dt.date
        target_games = upcoming_features[upcoming_features['game_date'] == target_date].copy()
        
        if target_games.empty:
            print(f"\n📭 No games scheduled for {target_date}")
            next_date = upcoming_features['game_date'].min()
            print(f"   Next games: {next_date.strftime('%A, %B %d, %Y')}")
            return
        
        print(f"🏀 Analyzing {len(target_games)} games...")
        print(f"💡 Typical NBA O/U Line: {typical_total}\n")
        
        predictions = []
        
        for idx, game in target_games.iterrows():
            home = game['home_team']
            away = game['away_team']
            
            # Calculate expected total (FIXED)
            expected_total, home_pts, away_pts = self.calculate_expected_total(game)
            
            # Get pace and ratings for context
            pace_avg = (game['pace_home'] + game['pace_away']) / 2
            ortg_home = game['off_rating_home']
            ortg_away = game['off_rating_away']
            drtg_home = game['def_rating_home']
            drtg_away = game['def_rating_away']
            
            # Calculate confidence based on variance
            pace_diff = abs(game['pace_home'] - game['pace_away'])
            
            # High/Low scoring indicators
            net_ortg = (ortg_home + ortg_away) / 2
            net_drtg = (drtg_home + drtg_away) / 2
            
            # Confidence based on consistency
            if pace_diff < 3:
                confidence = 'High'
            elif pace_diff < 7:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            predictions.append({
                'home_team': home,
                'away_team': away,
                'expected_total': expected_total,
                'home_pts': home_pts,
                'away_pts': away_pts,
                'pace_avg': pace_avg,
                'ortg_home': ortg_home,
                'ortg_away': ortg_away,
                'drtg_home': drtg_home,
                'drtg_away': drtg_away,
                'confidence': confidence
            })
        
        # Display predictions
        df = pd.DataFrame(predictions)
        df = df.sort_values('expected_total', ascending=False)
        
        print("="*80)
        
        over_count = 0
        under_count = 0
        
        for idx, pred in df.iterrows():
            diff = pred['expected_total'] - typical_total
            
            # Determine recommendation (need 8+ point edge for real value)
            if diff > 8:
                rec = "🔥 STRONG OVER"
                emoji = "📈"
                edge = diff
                over_count += 1
            elif diff > 3:
                rec = "📈 LEAN OVER"
                emoji = "⬆️"
                edge = diff
            elif diff < -8:
                rec = "❄️ STRONG UNDER"
                emoji = "📉"
                edge = abs(diff)
                under_count += 1
            elif diff < -3:
                rec = "📉 LEAN UNDER"
                emoji = "⬇️"
                edge = abs(diff)
            else:
                rec = "⚖️ NO BET (too close)"
                emoji = "⚠️"
                edge = abs(diff)
            
            print(f"\n{emoji} {pred['away_team']} @ {pred['home_team']}")
            print(f"   Predicted Total: {pred['expected_total']:.1f} points")
            print(f"      Home: {pred['home_pts']:.1f} | Away: {pred['away_pts']:.1f}")
            print(f"   vs Typical Line: {typical_total:.1f}")
            print(f"   Difference: {diff:+.1f} points")
            print(f"   🎯 Recommendation: {rec}")
            print(f"   ⭐ Confidence: {pred['confidence']}")
            print(f"   📊 Details:")
            print(f"      Pace: {pred['pace_avg']:.1f} possessions/game")
            print(f"      Off Ratings: {pred['ortg_home']:.1f} (H) | {pred['ortg_away']:.1f} (A)")
            print(f"      Def Ratings: {pred['drtg_home']:.1f} (H) | {pred['drtg_away']:.1f} (A)")
            
            if abs(diff) > 8:
                print(f"   💰 VALUE PLAY: ~{edge:.1f} point edge")
        
        # Summary
        print("\n" + "="*80)
        print("📊 TOTALS SUMMARY")
        print("="*80)
        
        strong_over = len(df[df['expected_total'] - typical_total > 8])
        lean_over = len(df[(df['expected_total'] - typical_total > 3) & 
                           (df['expected_total'] - typical_total <= 8)])
        strong_under = len(df[df['expected_total'] - typical_total < -8])
        lean_under = len(df[(df['expected_total'] - typical_total < -3) & 
                            (df['expected_total'] - typical_total >= -8)])
        no_play = len(df) - strong_over - lean_over - strong_under - lean_under
        
        print(f"Total Games: {len(df)}")
        print(f"Strong Over Plays (8+ pts): {strong_over}")
        print(f"Lean Over (3-8 pts): {lean_over}")
        print(f"Strong Under Plays (8+ pts): {strong_under}")
        print(f"Lean Under (3-8 pts): {lean_under}")
        print(f"No Plays (<3 pts): {no_play}")
        
        if strong_over > 0 or strong_under > 0:
            print(f"\n💎 {strong_over + strong_under} STRONG VALUE PLAYS")
        
        print("\n💡 HOW TO BET TOTALS:")
        print("   1. Get ACTUAL O/U line from sportsbook (not 220 default)")
        print("   2. Compare to predicted total above")
        print("   3. Strong plays: 8-10+ point difference")
        print("   4. Bet sizing: Same Kelly strategy")
        
        print("\n⚠️  IMPORTANT NOTES:")
        print("   • Default 220 is just for comparison")
        print("   • Actual lines vary by matchup (200-240 range)")
        print("   • Check injuries before betting!")
        print("   • Pace is very consistent = totals more predictable")
        print("="*80 + "\n")
        
        return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Totals (Over/Under) Predictions')
    parser.add_argument('--bankroll', type=float, default=1000)
    parser.add_argument('--typical-total', type=float, default=220.0,
                       help='Typical NBA O/U line (default: 220)')
    parser.add_argument('--tomorrow', action='store_true',
                       help='Predict tomorrow instead of today')
    
    args = parser.parse_args()
    
    predictor = TotalsPredictor(bankroll=args.bankroll)
    
    if args.tomorrow:
        target_date = (datetime.now() + timedelta(days=1)).date()
    else:
        target_date = datetime.now().date()
    
    predictor.predict_totals_today(
        target_date=target_date,
        typical_total=args.typical_total
    )

if __name__ == "__main__":
    main()