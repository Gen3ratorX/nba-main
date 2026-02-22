"""
totals_predictor.py - V5 CALIBRATED SNIPER PREDICTOR
Description:
    Loads the V5 Calibrated Model.
    Uses the built-in Vegas Calibration Layer to adjust predictions.
    Ranks games into tiers based on Quantile Uncertainty.
"""
import argparse
import pandas as pd
from datetime import timedelta
from data_processor import NBADataProcessor
from nbautils import get_team_name
from model_loader import load_totals_model
from totals_inference import predict_totals_for_game
from time_utils import get_user_timezone, get_nba_timezone, now_in_tz

def predict_totals(target_date=None):
    user_tz = get_user_timezone()
    nba_tz = get_nba_timezone()
    if target_date is None:
        target_date = now_in_tz(user_tz).date()
        
    date_str = target_date.strftime('%A, %B %d, %Y')
    
    # 1. Load totals model via centralized loader
    model_data = load_totals_model()
    if not model_data:
        print("❌ No totals model found. Run train_totals_model.py first.")
        return

    version = model_data.get('version', 'Unknown')
    print(f"\n" + "="*115)
    print(f"🎯 NBA SNIPER PREDICTOR - {date_str}")
    print(f"   🚀 Model: {version}")
    print(f"   🌍 User TZ: {user_tz.key} | NBA TZ: {nba_tz.key}")
    print("="*115)

    # 2. Get Upcoming Data
    print("📥 Processing upcoming games...")
    processor = NBADataProcessor()
    upcoming = processor.get_upcoming_games_features()
    
    if upcoming.empty:
        print("❌ No upcoming games found in schedule.")
        return

    upcoming['game_date'] = pd.to_datetime(upcoming['game_date']).dt.date
    games = upcoming[upcoming['game_date'] == target_date].copy()
    
    if games.empty:
        print(f"📭 No games found for {date_str}")
        return
        
    print(f"🏀 Analyzing {len(games)} games...")
    
    results = []

    # 3. Predict
    for idx, row in games.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        pred = predict_totals_for_game(row, model_data)
        if not pred:
            print(f"❌ Error predicting {home}")
            continue

        pred_main = pred['predicted_total']
        pred_lower = pred['lower']
        pred_upper = pred['upper']
        uncertainty = pred['uncertainty']
        range_str = f"{pred_lower:.0f}-{pred_upper:.0f}"

        # --- TIER LOGIC ---
        if uncertainty < 30:
            tier = "💎 DIAMOND"
            decision = "BET NOW"
            rank = 1
        elif uncertainty <= 40:
            tier = "✅ GOLD"
            decision = "ACTIVE"
            rank = 2
        else:
            tier = "⚠️ TRAP"
            decision = "SKIP"
            rank = 3

        results.append({
            'home': home, 'away': away,
            'pred': pred_main,
            'range': range_str,
            'unc': uncertainty,
            'tier': tier,
            'decision': decision,
            'rank': rank
        })

    # 4. Display Results
    if not results: return
    results.sort(key=lambda x: (x['rank'], x['unc']))
    
    print("\n" + "="*115)
    print(f"{'TIER':<12} | {'HOME vs AWAY':<30} | {'PRED':<6} | {'RANGE (Safety)':<15} | {'UNCERT':<6} | {'STATUS':<10}")
    print("-" * 115)
    
    for r in results:
        match = f"{get_team_name(r['home'])} vs {get_team_name(r['away'])}"
        if len(match) > 30: match = match[:30]
        print(f"{r['tier']:<12} | {match:<30} | {r['pred']:<6.1f} | {r['range']:<15} | {r['unc']:<6.1f} | {r['decision']:<10}")

    print("-" * 115)
    print("\n📌 V5 STRATEGY GUIDE:")
    print("   1. VEGAS GAP: Only play if PRED is > 3 points away from Vegas Line.")
    print("   2. SAFETY RANGE: If Vegas Line is OUTSIDE the 'Range (Safety)', the game is too volatile.")
    print("   3. TIER PRIORITY: Diamond (62% historical), Gold (56% historical).")
    print("="*115 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tomorrow', action='store_true')
    args = parser.parse_args()
    
    user_tz = get_user_timezone()
    t_date = now_in_tz(user_tz).date()
    if args.tomorrow: t_date += timedelta(days=1)
    predict_totals(target_date=t_date)
