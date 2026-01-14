"""
nbautils.py - Shared utility functions
Handles logging, team name standardization, and data processing
"""

import os
import json
import logging
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

# Import configuration
try:
    from config import (
        LOGS_DIR, LOG_LEVEL, NBA_TEAMS, NBA_TEAMS_REVERSE, TEAM_NAME_VARIATIONS,
        DATA_DIR, MODELS_DIR, HISTORICAL_DATA_DIR, CURRENT_DATA_DIR, UPCOMING_DATA_DIR
    )
except ImportError:
    # Fallback defaults if config isn't found
    LOGS_DIR = Path('logs')
    LOG_LEVEL = 'INFO'
    NBA_TEAMS = {}
    NBA_TEAMS_REVERSE = {}
    TEAM_NAME_VARIATIONS = {}

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(name: str = 'nba_predictor') -> logging.Logger:
    """Setup logging with file and console handlers"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_DIR / f"{name}_{current_date}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    if logger.handlers:
        return logger
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

def log_info(msg): logger.info(msg)
def log_error(msg): logger.error(msg)
def log_warning(msg): logger.warning(msg)
def log_debug(msg): logger.debug(msg)

# ============================================================================
# Team Name Normalization & Helpers
# ============================================================================

def normalize_team_name(team_name: str) -> str:
    """Convert any team name variation to standard 3-letter code"""
    if pd.isna(team_name) or not team_name:
        return "UNKNOWN"
    
    name_clean = str(team_name).strip().upper()
    
    # 1. Check if it's already a standard code
    if name_clean in NBA_TEAMS:
        return name_clean
    
    # 2. ESPN/Data Source Specific Mappings
    espn_mappings = {
        'WSH': 'WAS', 'UT': 'UTA', 'NO': 'NOP', 'NY': 'NYK', 
        'GS': 'GSW', 'SA': 'SAS', 'PHX': 'PHX', 'BKN': 'BKN', 'BRK': 'BKN'
    }
    if name_clean in espn_mappings:
        return espn_mappings[name_clean]
        
    # 3. Check variations dictionary
    name_lower = str(team_name).strip().lower()
    if name_lower in TEAM_NAME_VARIATIONS:
        return TEAM_NAME_VARIATIONS[name_lower]
    
    # 4. Try to find by full name
    for code, full_name in NBA_TEAMS.items():
        if name_lower == full_name.lower() or name_lower in full_name.lower():
            return code
            
    log_warning(f"Could not normalize team name: {team_name}, using {name_clean}")
    return name_clean

def get_team_name(team_code: str) -> str:
    return NBA_TEAMS.get(team_code, team_code)

def expand_team_abbreviation(team_abbr: str) -> str:
    return get_team_name(team_abbr)

def get_team_code(team_name: str) -> str:
    return NBA_TEAMS_REVERSE.get(team_name, normalize_team_name(team_name))

# ============================================================================
# Date & File Utilities
# ============================================================================

def ensure_directories():
    dirs = [LOGS_DIR, DATA_DIR, MODELS_DIR, HISTORICAL_DATA_DIR, CURRENT_DATA_DIR, UPCOMING_DATA_DIR]
    for d in dirs:
        if d: d.mkdir(parents=True, exist_ok=True)

def parse_date(date_input: Union[str, datetime, pd.Timestamp]) -> datetime:
    if isinstance(date_input, datetime): return date_input
    if isinstance(date_input, pd.Timestamp): return date_input.to_pydatetime()
    try:
        return pd.to_datetime(date_input).to_pydatetime()
    except Exception as e:
        log_error(f"Failed to parse date: {date_input}. Error: {e}")
        return datetime.now()

def get_season_from_date(date_input: Union[str, datetime]) -> str:
    try:
        date_obj = parse_date(date_input)
        year = date_obj.year
        if date_obj.month >= 10:
            return f"{year}-{str(year + 1)[2:]}"
        else:
            return f"{year - 1}-{str(year)[2:]}"
    except:
        return "2024-25"

def save_json(data: dict, filepath: Union[str, Path]) -> bool:
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        log_error(f"Failed to save JSON to {filepath}: {e}")
        return False

def load_json(filepath: Union[str, Path]) -> Optional[dict]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        log_error(f"Failed to load JSON from {filepath}: {e}")
        return None

def get_latest_file(directory: Path, pattern: str = "*.csv") -> Optional[Path]:
    try:
        files = list(directory.glob(pattern))
        if not files: return None
        return max(files, key=lambda f: f.stat().st_mtime)
    except Exception as e:
        log_error(f"Failed to get latest file: {e}")
        return None

# ============================================================================
# ELO & Stats Calculation Logic
# ============================================================================

def init_elo(teams: List[str], base: int = 1500) -> Dict[str, float]:
    return {team: float(base) for team in teams}

def update_elo(elo_ratings: Dict[str, float], home_team: str, away_team: str, 
               home_score: float, away_score: float, k_factor: float = 20) -> None:
    home_elo = elo_ratings.get(home_team, 1500.0)
    away_elo = elo_ratings.get(away_team, 1500.0)
    
    expected_home = 1 / (1 + 10**((away_elo - home_elo) / 400))
    expected_away = 1 - expected_home
    
    actual_home = 1.0 if home_score > away_score else 0.0
    actual_away = 1.0 - actual_home
    
    elo_ratings[home_team] = home_elo + k_factor * (actual_home - expected_home)
    elo_ratings[away_team] = away_elo + k_factor * (actual_away - expected_away)

def calculate_form(games_df: pd.DataFrame, team: str, window: int = 5) -> float:
    if games_df.empty: return 0.5
    team_games = games_df[(games_df['home_team'] == team) | (games_df['away_team'] == team)].tail(window)
    if team_games.empty: return 0.5
    
    wins = 0
    for _, game in team_games.iterrows():
        if game['home_team'] == team and game['home_score'] > game['away_score']: wins += 1
        elif game['away_team'] == team and game['away_score'] > game['home_score']: wins += 1
    return wins / len(team_games)

def calculate_rest_days(games_df: pd.DataFrame, team: str, current_date: str) -> int:
    if games_df.empty: return 7
    team_games = games_df[(games_df['home_team'] == team) | (games_df['away_team'] == team)]
    if team_games.empty: return 7
    
    try:
        team_games = team_games.sort_values('game_date')
        last_game = team_games.iloc[-1]
        last_date = parse_date(last_game['game_date'])
        current = parse_date(current_date)
        return max(0, (current - last_date).days)
    except Exception as e:
        log_warning(f"Error calculating rest days for {team}: {e}")
        return 2

def validate_game_data(game_record: dict) -> bool:
    required_fields = ['home_team', 'away_team', 'home_score', 'away_score', 'game_date']
    for field in required_fields:
        if field not in game_record or pd.isna(game_record[field]): return False
    return game_record['home_team'] != game_record['away_team']

# ============================================================================
# Advanced Metrics Calculation (UPDATED)
# ============================================================================

def calculate_advanced_metrics(games_df: pd.DataFrame, team: str, window: int = 10) -> Dict:
    """
    Calculate advanced stats with better formulas
    """
    metrics = {
        'offensive_rating': 110.0,
        'defensive_rating': 110.0,
        'net_rating': 0.0,
        'pace': 100.0,
        'win_streak': 0,
        'true_shooting_pct': 0.55  # NEW
    }
    
    if games_df.empty:
        return metrics

    team_games = games_df[
        (games_df['home_team'] == team) | (games_df['away_team'] == team)
    ].tail(window)
    
    if team_games.empty:
        return metrics

    total_pts = 0
    total_opp_pts = 0
    games_count = 0
    
    for _, game in team_games.iterrows():
        is_home = (game['home_team'] == team)
        pts = game['home_score'] if is_home else game['away_score']
        opp_pts = game['away_score'] if is_home else game['home_score']
        
        total_pts += pts
        total_opp_pts += opp_pts
        games_count += 1
        
    if games_count > 0:
        avg_pace = 100  # League average possessions
        
        # Better rating calculation (normalized per 100 possessions)
        ppg = total_pts / games_count
        opp_ppg = total_opp_pts / games_count
        
        metrics['offensive_rating'] = (ppg / avg_pace) * 100
        metrics['defensive_rating'] = (opp_ppg / avg_pace) * 100
        metrics['net_rating'] = metrics['offensive_rating'] - metrics['defensive_rating']
        metrics['pace'] = (total_pts + total_opp_pts) / (2 * games_count)

    # Calculate Win Streak
    streak = 0
    for _, game in team_games.iloc[::-1].iterrows():
        is_home = (game['home_team'] == team)
        won = (game['home_score'] > game['away_score']) if is_home else (game['away_score'] > game['home_score'])
        if won:
            streak += 1
        else:
            break
            
    metrics['win_streak'] = streak
    
    return metrics

# ============================================================================
# NEW: Additional Advanced Features
# ============================================================================

def is_back_to_back(games_df: pd.DataFrame, team: str, current_date: datetime) -> int:
    """Returns 1 if team played yesterday, else 0"""
    if games_df.empty:
        return 0
    
    team_games = games_df[
        (games_df['home_team'] == team) | (games_df['away_team'] == team)
    ].sort_values('game_date', ascending=False)
    
    if team_games.empty:
        return 0
    
    try:
        last_game_date = parse_date(team_games.iloc[0]['game_date'])
        current = parse_date(current_date)
        days_diff = (current - last_game_date).days
        return 1 if days_diff == 1 else 0
    except:
        return 0

def get_h2h_record(games_df: pd.DataFrame, home_team: str, away_team: str, window: int = 10) -> float:
    """Get home team's win rate vs away team in recent matchups"""
    if games_df.empty:
        return 0.5
    
    h2h = games_df[
        ((games_df['home_team'] == home_team) & (games_df['away_team'] == away_team)) |
        ((games_df['home_team'] == away_team) & (games_df['away_team'] == home_team))
    ].tail(window)
    
    if h2h.empty:
        return 0.5
    
    home_wins = 0
    for _, game in h2h.iterrows():
        if game['home_team'] == home_team:
            if game['home_score'] > game['away_score']:
                home_wins += 1
        else:
            if game['away_score'] > game['home_score']:
                home_wins += 1
    
    return home_wins / len(h2h)

def calculate_home_court_strength(games_df: pd.DataFrame, team: str, window: int = 20) -> float:
    """Calculate team's actual home court advantage in points"""
    if games_df.empty:
        return 3.5  # League average
    
    recent_games = games_df[
        (games_df['home_team'] == team) | (games_df['away_team'] == team)
    ].tail(window)
    
    if recent_games.empty:
        return 3.5
    
    home_games = recent_games[recent_games['home_team'] == team]
    away_games = recent_games[recent_games['away_team'] == team]
    
    if home_games.empty or away_games.empty:
        return 3.5
    
    # Calculate win rates
    home_win_rate = (home_games['home_score'] > home_games['away_score']).mean()
    away_win_rate = (away_games['away_score'] > away_games['home_score']).mean()
    
    # Convert differential to points (rough approximation)
    hca_points = (home_win_rate - away_win_rate) * 10
    
    # Clamp between 0 and 8 points
    return max(0, min(8, 3.5 + hca_points))