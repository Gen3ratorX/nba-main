"""
player_impact.py - Player Availability Impact Tracker

Uses nba_api to detect missing key players and compute roster strength
features for the prediction model.

Features produced:
- minutes_missing_home/away: total avg minutes of absent key players
- star_missing_home/away: binary, any 28+ MPG player absent
- roster_strength_home/away: ratio of available key minutes to full key minutes
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nba_api.stats.static import teams as nba_teams_static
from nbautils import log_info, log_warning, log_error

# Rate limit: nba_api recommends >= 0.6s between calls
_RATE_LIMIT_SECONDS = 0.7
_CACHE_DIR = Path("data/player_cache")
_IMPACT_CACHE_FILE = Path("data/player_impact_cache.csv")
_KEY_PLAYER_MIN_MPG = 15.0   # Players averaging 15+ MPG are "key"
_STAR_PLAYER_MIN_MPG = 28.0  # Players averaging 28+ MPG are "stars"


def _rate_limit():
    time.sleep(_RATE_LIMIT_SECONDS)


def _team_abbr_to_id(abbr: str) -> Optional[int]:
    """Convert 3-letter team abbreviation to nba_api team ID."""
    for t in nba_teams_static.get_teams():
        if t["abbreviation"] == abbr:
            return t["id"]
    return None


def _current_season_str() -> str:
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[2:]}"


class PlayerImpactTracker:
    def __init__(self):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._season_stats_cache: Dict[str, pd.DataFrame] = {}
        self._game_log_cache: Dict[str, pd.DataFrame] = {}
        # Load persistent impact cache for historical games
        self._impact_cache = self._load_impact_cache()

    # ------------------------------------------------------------------
    # Persistent cache for historical game impact features
    # ------------------------------------------------------------------
    def _load_impact_cache(self) -> Dict[str, Dict]:
        if not _IMPACT_CACHE_FILE.exists():
            return {}
        try:
            df = pd.read_csv(_IMPACT_CACHE_FILE)
            cache = {}
            for _, row in df.iterrows():
                key = f"{row['game_date']}_{row['home_team']}_{row['away_team']}"
                cache[key] = row.to_dict()
            return cache
        except Exception:
            return {}

    def _save_impact_cache(self):
        if not self._impact_cache:
            return
        df = pd.DataFrame(list(self._impact_cache.values()))
        df.to_csv(_IMPACT_CACHE_FILE, index=False)

    # ------------------------------------------------------------------
    # Season stats: who plays and how many minutes on average
    # ------------------------------------------------------------------
    def get_season_player_stats(self, season: str = None) -> pd.DataFrame:
        """
        Get per-player season averages (MPG, PPG, plus_minus) for all players.
        Cached per session and to disk.
        """
        season = season or _current_season_str()
        cache_key = season

        if cache_key in self._season_stats_cache:
            return self._season_stats_cache[cache_key]

        cache_file = _CACHE_DIR / f"league_player_stats_{season}.csv"
        # Use disk cache if less than 24 hours old
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=24):
                df = pd.read_csv(cache_file)
                self._season_stats_cache[cache_key] = df
                return df

        try:
            from nba_api.stats.endpoints import LeagueDashPlayerStats

            _rate_limit()
            stats = LeagueDashPlayerStats(
                season=season,
                per_mode_detailed="PerGame",
                timeout=10,
            )
            df = stats.get_data_frames()[0]

            # Keep relevant columns
            cols = [
                "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
                "GP", "MIN", "PTS", "PLUS_MINUS",
            ]
            df = df[[c for c in cols if c in df.columns]].copy()
            df.to_csv(cache_file, index=False)
            self._season_stats_cache[cache_key] = df
            log_info(f"Fetched season player stats: {len(df)} players ({season})")
            return df
        except Exception as e:
            # Fall back to expired disk cache if available
            if cache_file.exists():
                df = pd.read_csv(cache_file)
                self._season_stats_cache[cache_key] = df
                log_warning(f"NBA.com unavailable, using cached player stats ({len(df)} players)")
                return df
            # Cache the empty result so we don't retry and timeout repeatedly
            log_warning(f"Could not fetch season player stats (will skip retries): {e}")
            self._season_stats_cache[cache_key] = pd.DataFrame()
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Player game logs: who played in a specific game window
    # ------------------------------------------------------------------
    def get_player_recent_games(
        self, player_id: int, season: str = None, last_n: int = 5
    ) -> pd.DataFrame:
        """Get a player's recent game log to detect absences."""
        season = season or _current_season_str()
        cache_key = f"{player_id}_{season}"

        if cache_key in self._game_log_cache:
            return self._game_log_cache[cache_key].tail(last_n)

        cache_file = _CACHE_DIR / f"gamelog_{player_id}_{season}.csv"
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=12):
                df = pd.read_csv(cache_file)
                self._game_log_cache[cache_key] = df
                return df.tail(last_n)

        try:
            from nba_api.stats.endpoints import PlayerGameLog

            _rate_limit()
            log = PlayerGameLog(player_id=str(player_id), season=season, timeout=10)
            df = log.get_data_frames()[0]
            df.to_csv(cache_file, index=False)
            self._game_log_cache[cache_key] = df
            return df.tail(last_n)
        except Exception as e:
            if cache_file.exists():
                df = pd.read_csv(cache_file)
                self._game_log_cache[cache_key] = df
                return df.tail(last_n)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Detect missing key players for a team on a given date
    # ------------------------------------------------------------------
    def _get_key_players(self, team_abbr: str, season: str = None) -> pd.DataFrame:
        """Return players on `team_abbr` averaging >= KEY_PLAYER_MIN_MPG."""
        stats = self.get_season_player_stats(season)
        if stats.empty:
            return pd.DataFrame()

        team_players = stats[stats["TEAM_ABBREVIATION"] == team_abbr].copy()
        team_players["MIN"] = pd.to_numeric(team_players["MIN"], errors="coerce")
        return team_players[team_players["MIN"] >= _KEY_PLAYER_MIN_MPG]

    def detect_missing_for_upcoming(
        self, team_abbr: str, season: str = None
    ) -> Tuple[float, bool, float]:
        """
        For an upcoming game: check each key player's recent game log.
        If a key player hasn't played in the last 3 games, assume they're out.

        Returns: (minutes_missing, star_missing, roster_strength)
        """
        key_players = self._get_key_players(team_abbr, season)
        if key_players.empty:
            return 0.0, False, 1.0

        total_key_minutes = float(key_players["MIN"].sum())
        minutes_missing = 0.0
        star_missing = False

        for _, player in key_players.iterrows():
            pid = int(player["PLAYER_ID"])
            avg_min = float(player["MIN"])

            recent = self.get_player_recent_games(pid, season, last_n=3)
            if recent.empty or len(recent) == 0:
                # No recent games found — assume out
                minutes_missing += avg_min
                if avg_min >= _STAR_PLAYER_MIN_MPG:
                    star_missing = True
                continue

            # Check if player played 0 minutes in all recent games
            if "MIN" in recent.columns:
                recent_mins = pd.to_numeric(recent["MIN"], errors="coerce")
                if recent_mins.sum() == 0:
                    minutes_missing += avg_min
                    if avg_min >= _STAR_PLAYER_MIN_MPG:
                        star_missing = True

        roster_strength = max(0.0, (total_key_minutes - minutes_missing) / total_key_minutes) if total_key_minutes > 0 else 1.0
        return minutes_missing, star_missing, roster_strength

    def detect_missing_for_historical(
        self, team_abbr: str, game_date: str, season: str = None
    ) -> Tuple[float, bool, float]:
        """
        For a historical game: check each key player's game log around that date.
        If they didn't play on game_date but were active before/after, count as missing.

        Returns: (minutes_missing, star_missing, roster_strength)
        """
        season = season or _current_season_str()
        key_players = self._get_key_players(team_abbr, season)
        if key_players.empty:
            return 0.0, False, 1.0

        total_key_minutes = float(key_players["MIN"].sum())
        minutes_missing = 0.0
        star_missing = False
        game_dt = pd.to_datetime(game_date).date()

        for _, player in key_players.iterrows():
            pid = int(player["PLAYER_ID"])
            avg_min = float(player["MIN"])

            # Get full season game log (cached)
            cache_key = f"{pid}_{season}"
            if cache_key in self._game_log_cache:
                log_df = self._game_log_cache[cache_key]
            else:
                log_df = self.get_player_recent_games(pid, season, last_n=999)

            if log_df.empty:
                continue  # Can't determine — assume played

            # Check if game_date appears in the log
            if "GAME_DATE" in log_df.columns:
                log_dates = pd.to_datetime(log_df["GAME_DATE"], errors="coerce").dt.date
                played_on_date = game_dt in log_dates.values

                if not played_on_date:
                    # Check player was active around this date (played within +/- 10 days)
                    nearby = log_dates[(log_dates >= game_dt - timedelta(days=10)) &
                                       (log_dates <= game_dt + timedelta(days=10))]
                    if len(nearby) > 0:
                        # Active player who missed this specific game
                        minutes_missing += avg_min
                        if avg_min >= _STAR_PLAYER_MIN_MPG:
                            star_missing = True

        roster_strength = max(0.0, (total_key_minutes - minutes_missing) / total_key_minutes) if total_key_minutes > 0 else 1.0
        return minutes_missing, star_missing, roster_strength

    def _detect_with_live_injuries(
        self, team_abbr: str, season: str = None
    ) -> Tuple[float, bool, float]:
        """
        Detect missing players using live ESPN injury report, with fallback
        to game-log inference if scraping fails.
        """
        try:
            from injury_scraper import InjuryScraper
            scraper = InjuryScraper()
            player_stats = self._get_key_players(team_abbr, season)

            if player_stats.empty:
                return 0.0, False, 1.0

            impact = scraper.get_missing_impact(team_abbr, player_stats)

            # If injury report found relevant data, use it
            if impact["minutes_missing"] > 0 or impact["star_missing"]:
                return impact["minutes_missing"], impact["star_missing"], impact["roster_strength"]

            # If no injuries found for this team on the report, check game logs too
            # (some absences may not be on the injury report — rest days, etc.)
            return self.detect_missing_for_upcoming(team_abbr, season)

        except Exception as e:
            log_warning(f"Live injury check failed for {team_abbr}: {e}")
            return self.detect_missing_for_upcoming(team_abbr, season)

    # ------------------------------------------------------------------
    # Main entry point: compute 6 features for a game
    # ------------------------------------------------------------------
    def compute_impact_features(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        is_upcoming: bool = False,
        season: str = None,
    ) -> Dict[str, float]:
        """
        Compute player availability features for a single game.
        Returns dict with 6 features. Never raises — returns neutral defaults on failure.
        """
        # Check persistent cache for historical games
        cache_key = f"{game_date}_{home_team}_{away_team}"
        if not is_upcoming and cache_key in self._impact_cache:
            cached = self._impact_cache[cache_key]
            return {
                "minutes_missing_home": float(cached.get("minutes_missing_home", 0)),
                "minutes_missing_away": float(cached.get("minutes_missing_away", 0)),
                "star_missing_home": float(cached.get("star_missing_home", 0)),
                "star_missing_away": float(cached.get("star_missing_away", 0)),
                "roster_strength_home": float(cached.get("roster_strength_home", 1.0)),
                "roster_strength_away": float(cached.get("roster_strength_away", 1.0)),
            }

        defaults = {
            "minutes_missing_home": 0.0,
            "minutes_missing_away": 0.0,
            "star_missing_home": 0.0,
            "star_missing_away": 0.0,
            "roster_strength_home": 1.0,
            "roster_strength_away": 1.0,
        }

        try:
            if is_upcoming:
                # Try live injury report first, fall back to game-log inference
                h_mm, h_star, h_rs = self._detect_with_live_injuries(home_team, season)
                a_mm, a_star, a_rs = self._detect_with_live_injuries(away_team, season)
            else:
                h_mm, h_star, h_rs = self.detect_missing_for_historical(home_team, game_date, season)
                a_mm, a_star, a_rs = self.detect_missing_for_historical(away_team, game_date, season)

            result = {
                "minutes_missing_home": float(h_mm),
                "minutes_missing_away": float(a_mm),
                "star_missing_home": float(h_star),
                "star_missing_away": float(a_star),
                "roster_strength_home": float(h_rs),
                "roster_strength_away": float(a_rs),
            }

            # Cache historical results
            if not is_upcoming:
                self._impact_cache[cache_key] = {
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    **result,
                }

            return result

        except Exception as e:
            log_warning(f"Player impact failed for {away_team}@{home_team} ({game_date}): {e}")
            return defaults

    def flush_cache(self):
        """Write the persistent impact cache to disk."""
        self._save_impact_cache()
