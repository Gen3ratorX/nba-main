"""
backfill_odds.py - Convert Kaggle historical odds CSV into pipeline format.

Reads oddsData.csv (Kaggle NBA Odds Data, 2008-2023) and writes to
data/odds_history/odds_history.csv in the format expected by data_processor.py.

Usage:
    python backfill_odds.py
    python backfill_odds.py --input path/to/oddsData.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Map Kaggle city names → pipeline team abbreviations
TEAM_MAP = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN",
    "Charlotte": "CHA", "Chicago": "CHI", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET",
    "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LA Clippers": "LAC", "LA Lakers": "LAL", "Memphis": "MEM",
    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "New Jersey": "BKN",  # Nets moved in 2012
    "New Orleans": "NOP", "New York": "NYK",
    "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI",
    "Phoenix": "PHX", "Portland": "POR", "Sacramento": "SAC",
    "San Antonio": "SAS", "Seattle": "OKC",  # Sonics became Thunder in 2008
    "Toronto": "TOR", "Utah": "UTA", "Washington": "WAS",
}

OUTPUT_PATH = Path("data/odds_history/odds_history.csv")


def backfill(input_path: str = "oddsData.csv") -> pd.DataFrame:
    """Transform Kaggle odds data into pipeline format."""
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Keep only home rows (home/visitor == 'vs') to get one row per game
    home_df = df[df["home/visitor"] == "vs"].copy()
    print(f"Home rows (unique games): {len(home_df)}")

    # Map team names
    home_df["home_team"] = home_df["team"].map(TEAM_MAP)
    home_df["away_team"] = home_df["opponent"].map(TEAM_MAP)

    # Drop games with unmapped teams
    unmapped = home_df[home_df["home_team"].isna() | home_df["away_team"].isna()]
    if len(unmapped) > 0:
        unmapped_teams = set(unmapped["team"].unique()) | set(unmapped["opponent"].unique())
        unmapped_teams = {t for t in unmapped_teams if t not in TEAM_MAP}
        print(f"Warning: {len(unmapped)} rows with unmapped teams: {unmapped_teams}")
    home_df = home_df.dropna(subset=["home_team", "away_team"])

    # Build output DataFrame
    out = pd.DataFrame({
        "game_date": pd.to_datetime(home_df["date"]).dt.strftime("%Y-%m-%d"),
        "home_team": home_df["home_team"].values,
        "away_team": home_df["away_team"].values,
        "ml_home": pd.to_numeric(home_df["moneyLine"], errors="coerce"),
        "ml_away": pd.to_numeric(home_df["opponentMoneyLine"], errors="coerce"),
        "total_line": pd.to_numeric(home_df["total"], errors="coerce"),
        "spread": pd.to_numeric(home_df["spread"], errors="coerce"),
        "source": "kaggle_backfill",
        "fetched_at": pd.to_datetime(home_df["date"]).dt.strftime("%Y-%m-%dT12:00:00"),
    })

    # Drop rows with missing odds
    before = len(out)
    out = out.dropna(subset=["ml_home", "ml_away"])
    print(f"Dropped {before - len(out)} rows with missing moneyline odds")

    # Sort by date
    out = out.sort_values("game_date").reset_index(drop=True)

    # Check for existing odds_history and merge (don't overwrite live snapshots)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        source_col = existing["source"] if "source" in existing.columns else pd.Series("", index=existing.index)
        live = existing[source_col != "kaggle_backfill"]
        if len(live) > 0:
            print(f"Preserving {len(live)} existing live odds snapshots")
            out = pd.concat([out, live], ignore_index=True)
            out = out.drop_duplicates(
                subset=["game_date", "home_team", "away_team", "source"],
                keep="last",
            )

    out = out.sort_values("game_date").reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)

    # Stats
    seasons = pd.to_datetime(out["game_date"]).dt.year.nunique()
    print(f"\nBackfill complete!")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Total games with odds: {len(out)}")
    print(f"  Date range: {out['game_date'].min()} to {out['game_date'].max()}")
    print(f"  Seasons covered: {seasons}")
    print(f"  Moneyline coverage: {out['ml_home'].notna().sum()} / {len(out)} ({out['ml_home'].notna().mean():.1%})")
    print(f"  Total line coverage: {out['total_line'].notna().sum()} / {len(out)} ({out['total_line'].notna().mean():.1%})")

    return out


def backfill_games(input_path: str = "oddsData.csv") -> pd.DataFrame:
    """
    Also create historical game records from the Kaggle data so the pipeline
    has games to match odds against. Appends to BallDontLie historical data.
    """
    df = pd.read_csv(input_path)
    home_df = df[df["home/visitor"] == "vs"].copy()

    home_df["home_team"] = home_df["team"].map(TEAM_MAP)
    home_df["away_team"] = home_df["opponent"].map(TEAM_MAP)
    home_df = home_df.dropna(subset=["home_team", "away_team"])

    games = pd.DataFrame({
        "game_id": range(100000, 100000 + len(home_df)),
        "game_date": pd.to_datetime(home_df["date"]).dt.strftime("%Y-%m-%d"),
        "home_team": home_df["home_team"].values,
        "away_team": home_df["away_team"].values,
        "home_score": pd.to_numeric(home_df["score"], errors="coerce"),
        "away_score": pd.to_numeric(home_df["opponentScore"], errors="coerce"),
        "season": home_df["season"].values,
        "data_source": "kaggle_backfill",
    })
    games = games.dropna(subset=["home_score", "away_score"])
    games["home_score"] = games["home_score"].astype(int)
    games["away_score"] = games["away_score"].astype(int)

    # Load existing BallDontLie data
    hist_path = Path("data/historical/historical_games_balldontlie.csv")
    if hist_path.exists():
        existing = pd.read_csv(hist_path)
        # Only add games that don't overlap
        existing_dates = set(existing["game_date"].astype(str))
        new_games = games[~games["game_date"].isin(existing_dates)]
        if len(new_games) > 0:
            combined = pd.concat([new_games, existing], ignore_index=True)
            combined = combined.sort_values("game_date").reset_index(drop=True)
            # Backup
            import shutil
            shutil.copy(hist_path, hist_path.with_suffix(".csv.pre_backfill"))
            combined.to_csv(hist_path, index=False)
            print(f"\nGames backfill: added {len(new_games)} historical games")
            print(f"  Total games now: {len(combined)}")
            print(f"  Date range: {combined['game_date'].min()} to {combined['game_date'].max()}")
        else:
            print("\nGames backfill: no new games to add (dates already covered)")
    else:
        games.to_csv(hist_path, index=False)
        print(f"\nGames backfill: created {len(games)} historical games")

    return games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical odds + games from Kaggle dataset")
    parser.add_argument("--input", default="oddsData.csv", help="Path to oddsData.csv")
    parser.add_argument("--games-only", action="store_true", help="Only backfill games, skip odds")
    args = parser.parse_args()
    if not args.games_only:
        backfill(args.input)
    backfill_games(args.input)
