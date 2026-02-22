import json
import os
from pathlib import Path

# Config
FILE_PATH = Path('data/prediction_history.json')
BAD_ID = "2026-01-29_IND_CHI"
BAD_TIMESTAMP = "2026-01-21T11:44:51.939964"

def clean_history():
    print(f"🧹 Cleaning {FILE_PATH}...")
    
    if not FILE_PATH.exists():
        print("❌ File not found!")
        return

    # Load Data
    with open(FILE_PATH, 'r') as f:
        data = json.load(f)
    
    preds = data.get('predictions', [])
    initial_count = len(preds)
    print(f"📉 Starting count: {initial_count} entries")

    # 1. Filter out the specific BAD Indiana entry
    # We filter out the one that matches BOTH the ID and the old timestamp
    preds = [p for p in preds if not (p['prediction_id'] == BAD_ID and p['timestamp'] == BAD_TIMESTAMP)]
    
    print(f"✅ Removed bad Indiana entry. Count: {len(preds)}")

    # 2. Remove Duplicates (Keep the most recent one)
    # Strategy: Sort by timestamp (newest first), then keep unique game_ids
    
    # Sort descending by timestamp (newest first)
    preds.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    unique_preds = []
    seen_games = set()
    
    for p in preds:
        # Create a unique key for the game
        game_key = f"{p['game_date']}_{p['home_team']}_{p['away_team']}"
        
        if game_key not in seen_games:
            unique_preds.append(p)
            seen_games.add(game_key)
    
    # Reverse back to chronological order (optional, but looks nicer)
    unique_preds.sort(key=lambda x: x.get('game_date', ''))
    
    final_count = len(unique_preds)
    removed_count = initial_count - final_count
    
    # Update data
    data['predictions'] = unique_preds
    
    # Save
    with open(FILE_PATH, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"✨ Cleanup Complete!")
    print(f"🗑️  Removed {removed_count} duplicates/garbage entries.")
    print(f"📝 Final Count: {final_count} clean predictions.")

if __name__ == "__main__":
    clean_history()
