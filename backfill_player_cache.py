"""
backfill_player_cache.py - One-time script to populate the player impact cache.

Iterates all historical games and computes player availability features,
caching results to data/player_impact_cache.csv. Subsequent pipeline runs
will hit the cache instead of making API calls.

Usage:
    python backfill_player_cache.py                  # Process all uncached games
    python backfill_player_cache.py --season 2025-26 # Specific season only
    python backfill_player_cache.py --limit 500      # Process at most 500 games
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from config import HISTORICAL_DATA_DIR, CURRENT_DATA_DIR
from nbautils import log_info, log_warning
from player_impact import PlayerImpactTracker, _IMPACT_CACHE_FILE


def load_all_games() -> pd.DataFrame:
    """Load all historical + current season games."""
    dfs = []
    for f in [
        HISTORICAL_DATA_DIR / "historical_games_balldontlie.csv",
        CURRENT_DATA_DIR / "current_season_espn.csv",
    ]:
        if f.exists():
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date", "home_team", "away_team"])
    df = df[(df["home_score"] > 0) & (df["away_score"] > 0)]
    df = df.sort_values("game_date").drop_duplicates(
        subset=["game_date", "home_team", "away_team"]
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Backfill player impact cache")
    parser.add_argument("--season", type=str, default=None, help="NBA season (e.g. 2025-26)")
    parser.add_argument("--limit", type=int, default=0, help="Max games to process (0=all)")
    parser.add_argument("--force", action="store_true", help="Re-compute even if cached")
    args = parser.parse_args()

    games = load_all_games()
    if games.empty:
        print("No games found.")
        return

    tracker = PlayerImpactTracker()
    total = len(games)
    processed = 0
    skipped = 0
    errors = 0

    print(f"Found {total} games. Existing cache: {len(tracker._impact_cache)} entries.")
    print(f"Season filter: {args.season or 'all'}")
    print()

    for idx, (_, game) in enumerate(games.iterrows()):
        home = str(game["home_team"])
        away = str(game["away_team"])
        game_date = str(game["game_date"])[:10]
        cache_key = f"{game_date}_{home}_{away}"

        # Skip if already cached (unless --force)
        if not args.force and cache_key in tracker._impact_cache:
            skipped += 1
            continue

        # Season filter
        season = args.season
        if season:
            gd = pd.to_datetime(game_date)
            # Determine the season this game belongs to
            if gd.month >= 10:
                game_season = f"{gd.year}-{str(gd.year + 1)[2:]}"
            else:
                game_season = f"{gd.year - 1}-{str(gd.year)[2:]}"
            if game_season != season:
                skipped += 1
                continue

        if args.limit > 0 and processed >= args.limit:
            break

        try:
            tracker.compute_impact_features(
                home_team=home,
                away_team=away,
                game_date=game_date,
                is_upcoming=False,
                season=season,
            )
            processed += 1
        except Exception as e:
            errors += 1
            log_warning(f"Error for {away}@{home} ({game_date}): {e}")

        # Progress every 100 games
        if processed > 0 and processed % 100 == 0:
            tracker.flush_cache()
            print(f"  Processed {processed}/{total - skipped} (cached {skipped}, errors {errors})")

    # Final flush
    tracker.flush_cache()

    print(f"\nDone! Processed: {processed}, Skipped (cached): {skipped}, Errors: {errors}")
    print(f"Total cache entries: {len(tracker._impact_cache)}")


if __name__ == "__main__":
    main()
