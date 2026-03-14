"""
injury_scraper.py - Live NBA injury report from ESPN.
Scrapes https://www.espn.com/nba/injuries for same-day player availability.
Used to enhance PlayerImpactTracker with real-time injury data for upcoming games.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from nbautils import log_info, log_warning

# ESPN injury page
_ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"

# Cache settings
_CACHE_FILE = Path("data/injury_cache.csv")
_CACHE_TTL_SECONDS = 7200  # 2 hours

# Team name normalization (ESPN full names → 3-letter codes)
_ESPN_TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Status weights: how likely the player actually misses the game
STATUS_WEIGHTS = {
    "out": 1.0,
    "doubtful": 0.85,
    "questionable": 0.25,
    "day-to-day": 0.50,
    "probable": 0.10,
    "suspension": 1.0,
}


class InjuryScraper:
    """Scrape ESPN NBA injury report and provide structured player availability data."""

    def __init__(self):
        self._cache_df: Optional[pd.DataFrame] = None
        self._cache_time: float = 0.0

    def fetch_injury_report(self, force: bool = False) -> pd.DataFrame:
        """
        Fetch current NBA injury report from ESPN.
        Returns DataFrame with: team, team_code, player_name, position, status,
                                status_weight, injury, est_return, updated_at
        """
        # Check cache
        if not force and self._cache_df is not None:
            if (time.time() - self._cache_time) < _CACHE_TTL_SECONDS:
                return self._cache_df.copy()

        # Check disk cache
        if not force and _CACHE_FILE.exists():
            try:
                cached = pd.read_csv(_CACHE_FILE)
                if "fetched_at" in cached.columns:
                    fetched = pd.to_datetime(cached["fetched_at"].iloc[0])
                    age_seconds = (datetime.now() - fetched).total_seconds()
                    if age_seconds < _CACHE_TTL_SECONDS:
                        self._cache_df = cached
                        self._cache_time = time.time()
                        log_info(f"Using cached injury report ({len(cached)} entries, {age_seconds/60:.0f}m old)")
                        return cached.copy()
            except Exception:
                pass

        # Fetch fresh data
        try:
            log_info("🏥 Fetching live NBA injury report from ESPN...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            resp = requests.get(_ESPN_INJURIES_URL, headers=headers, timeout=15)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            log_warning(f"Failed to fetch ESPN injuries: {e}")
            # Return empty or cached
            if self._cache_df is not None:
                return self._cache_df.copy()
            return pd.DataFrame()

        # Parse the HTML
        injuries = self._parse_espn_injuries(html)
        if not injuries:
            log_warning("No injury data parsed from ESPN page")
            if self._cache_df is not None:
                return self._cache_df.copy()
            return pd.DataFrame()

        df = pd.DataFrame(injuries)
        df["fetched_at"] = datetime.now().isoformat()

        # Save to disk cache
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(_CACHE_FILE, index=False)

        # Update memory cache
        self._cache_df = df
        self._cache_time = time.time()

        log_info(f"✅ Injury report: {len(df)} players across {df['team_code'].nunique()} teams")
        return df.copy()

    def _parse_espn_injuries(self, html: str) -> List[Dict]:
        """Parse ESPN injury page HTML into structured records."""
        injuries = []

        # ESPN structures injuries by team sections
        # Each team has a header with team name and a table with player rows
        # We use regex-based parsing to avoid BeautifulSoup dependency

        # Find team sections — look for team name patterns
        # ESPN uses patterns like: class="Table__Title">Team Name</
        team_sections = re.split(
            r'(?=<div[^>]*class="[^"]*ResponsiveTable[^"]*")',
            html,
        )

        current_team = None
        current_team_code = None

        for section in team_sections:
            # Try to find team name in this section
            team_match = re.search(
                r'class="[^"]*injuries__teamName[^"]*"[^>]*>([^<]+)<',
                section,
            )
            if not team_match:
                # Alternative pattern
                team_match = re.search(
                    r'class="[^"]*Table__Title[^"]*"[^>]*>([^<]+)<',
                    section,
                )
            if team_match:
                team_name = team_match.group(1).strip()
                current_team = team_name
                current_team_code = _ESPN_TEAM_MAP.get(team_name)
                if not current_team_code:
                    # Try fuzzy match
                    for full_name, code in _ESPN_TEAM_MAP.items():
                        if team_name.lower() in full_name.lower() or full_name.lower() in team_name.lower():
                            current_team_code = code
                            break

            if not current_team_code:
                continue

            # Find player rows in this section
            # ESPN table rows contain: Name, POS, Est Return, Status, Comment
            rows = re.findall(
                r'<tr[^>]*class="[^"]*Table__TR[^"]*"[^>]*>(.*?)</tr>',
                section,
                re.DOTALL,
            )

            for row in rows:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                if len(cells) < 4:
                    continue

                # Extract player name (may be in an <a> tag)
                name_match = re.search(r'>([^<]+)</a>', cells[0])
                if not name_match:
                    name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', cells[0])
                if not name_match:
                    continue

                player_name = name_match.group(1).strip()

                # Clean HTML from cells
                def clean(cell):
                    return re.sub(r'<[^>]+>', '', cell).strip()

                position = clean(cells[1]) if len(cells) > 1 else ""
                est_return = clean(cells[2]) if len(cells) > 2 else ""
                status = clean(cells[3]) if len(cells) > 3 else ""
                comment = clean(cells[4]) if len(cells) > 4 else ""

                # Normalize status
                status_lower = status.lower().strip()
                weight = STATUS_WEIGHTS.get(status_lower, 0.5)

                injuries.append({
                    "team": current_team,
                    "team_code": current_team_code,
                    "player_name": player_name,
                    "position": position,
                    "status": status,
                    "status_weight": weight,
                    "injury": comment,
                    "est_return": est_return,
                })

        return injuries

    def get_team_injuries(self, team_code: str) -> pd.DataFrame:
        """Get injuries for a specific team (3-letter code)."""
        df = self.fetch_injury_report()
        if df.empty:
            return df
        return df[df["team_code"] == team_code].copy()

    def get_missing_impact(
        self,
        team_code: str,
        player_stats: pd.DataFrame,
    ) -> Dict:
        """
        Cross-reference injury report with season stats to compute impact.

        Args:
            team_code: 3-letter team abbreviation
            player_stats: DataFrame with PLAYER_NAME, MIN columns (from nba_api)

        Returns:
            dict with minutes_missing, star_missing, roster_strength
        """
        team_injuries = self.get_team_injuries(team_code)
        if team_injuries.empty or player_stats.empty:
            return {"minutes_missing": 0.0, "star_missing": False, "roster_strength": 1.0}

        from player_impact import _KEY_PLAYER_MIN_MPG, _STAR_PLAYER_MIN_MPG

        key_players = player_stats[player_stats["MIN"] >= _KEY_PLAYER_MIN_MPG].copy()
        if key_players.empty:
            return {"minutes_missing": 0.0, "star_missing": False, "roster_strength": 1.0}

        total_key_minutes = float(key_players["MIN"].sum())
        minutes_missing = 0.0
        star_missing = False

        for _, injury_row in team_injuries.iterrows():
            inj_name = injury_row["player_name"].lower().strip()
            weight = float(injury_row.get("status_weight", 0.5))

            # Match by name (fuzzy: last name match)
            for _, player in key_players.iterrows():
                player_name = str(player.get("PLAYER_NAME", "")).lower().strip()
                # Check full name or last name match
                if inj_name == player_name or inj_name.split()[-1] == player_name.split()[-1]:
                    avg_min = float(player["MIN"])
                    minutes_missing += avg_min * weight
                    if avg_min >= _STAR_PLAYER_MIN_MPG and weight >= 0.5:
                        star_missing = True
                    break

        roster_strength = max(0.0, (total_key_minutes - minutes_missing) / total_key_minutes) if total_key_minutes > 0 else 1.0

        return {
            "minutes_missing": round(minutes_missing, 1),
            "star_missing": star_missing,
            "roster_strength": round(roster_strength, 3),
        }


def print_injury_report():
    """Standalone: print current injury report."""
    scraper = InjuryScraper()
    df = scraper.fetch_injury_report(force=True)
    if df.empty:
        print("No injury data available.")
        return

    print(f"\n{'='*80}")
    print(f"🏥 NBA INJURY REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*80}")

    for team_code in sorted(df["team_code"].unique()):
        team_df = df[df["team_code"] == team_code]
        team_name = team_df.iloc[0]["team"]
        print(f"\n{team_name} ({team_code}):")
        for _, row in team_df.iterrows():
            status = row["status"]
            name = row["player_name"]
            injury = row.get("injury", "")
            est = row.get("est_return", "")
            print(f"  {status:<15} {name:<25} {est:<12} {injury}")

    print(f"\n{'='*80}")
    print(f"Total: {len(df)} players on injury report")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print_injury_report()
