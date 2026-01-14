"""
config.py - UPDATED FOR FREE APIS + NEW FEATURES
Configuration Management for NBA Prediction System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# API Configuration (FREE TIER)
# ============================================================================

# BallDontLie API (Required for Free Backup)
BALLDONTLIE_API_KEY = os.getenv('BALLDONTLIE_API_KEY', '0f20cae2-c584-4bd0-bc12-bb5020238cf0')

# ============================================================================
# Directory Configuration
# ============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.getenv('DATA_DIR', './data')).resolve()
MODELS_DIR = Path(os.getenv('MODELS_DIR', './models')).resolve()
LOGS_DIR = Path(os.getenv('LOGS_DIR', './logs')).resolve()

# Data subdirectories
HISTORICAL_DATA_DIR = DATA_DIR / 'historical'
CURRENT_DATA_DIR = DATA_DIR / 'current'
UPCOMING_DATA_DIR = DATA_DIR / 'upcoming'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PREDICTIONS_DIR = DATA_DIR / 'predictions' 

# ============================================================================
# Database Configuration
# ============================================================================

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///nba_predictions.db')

# ============================================================================
# Model Configuration
# ============================================================================

MIN_GAMES_FOR_TRAINING = int(os.getenv('MIN_GAMES_FOR_TRAINING', '50'))
MODEL_RETRAIN_DAYS = int(os.getenv('MODEL_RETRAIN_DAYS', '7'))
PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.60'))

# ============================================================================
# Feature Configuration (UPDATED WITH ALL NEW FEATURES)
# ============================================================================

# Core ELO Features
ELO_FEATURES = [
    'home_elo', 
    'away_elo', 
    'elo_diff',
]

# Form & Rest Features
FORM_REST_FEATURES = [
    'form_home', 
    'form_away', 
    'form_diff',
    'rest_home', 
    'rest_away', 
    'rest_diff',
]

# Home Court Advantage (now properly calculated)
HCA_FEATURES = [
    'home_court_advantage',
]

# Advanced Metrics (Offensive/Defensive Ratings)
ADVANCED_METRICS_FEATURES = [
    'off_rating_home', 
    'off_rating_away',
    'def_rating_home', 
    'def_rating_away',
    'net_rating_home', 
    'net_rating_away',
    'pace_home', 
    'pace_away',
    'win_streak_home', 
    'win_streak_away',
]

# NEW: Situational Features
SITUATIONAL_FEATURES = [
    'back_to_back_home',
    'back_to_back_away',
    'h2h_win_rate',
]

# Combined Feature List (ALL FEATURES)
FEATURES = (
    ELO_FEATURES + 
    FORM_REST_FEATURES + 
    HCA_FEATURES + 
    ADVANCED_METRICS_FEATURES + 
    SITUATIONAL_FEATURES
)

# Legacy support (keep CORE_FEATURES for backwards compatibility)
CORE_FEATURES = FEATURES

# ============================================================================
# Model Configuration (Enhanced Ensemble)
# ============================================================================

# RandomForest Configuration (Improved Parameters)
MODEL_CONFIG = {
    'n_estimators': 300,        # Increased from 200
    'max_depth': 12,            # Increased from 10
    'min_samples_split': 5,     # Decreased from 20 (more granular)
    'min_samples_leaf': 2,      # Decreased from 10
    'class_weight': 'balanced', # NEW: Handle class imbalance
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost Configuration (NEW)
XGBOOST_CONFIG = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# Gradient Boosting Configuration (Improved)
GB_CONFIG = {
    'n_estimators': 150,        # Increased from 100
    'learning_rate': 0.05,      # Decreased from 0.1 (more stable)
    'max_depth': 6,             # Increased from 5
    'subsample': 0.8,           # NEW: Prevent overfitting
    'random_state': 42
}

# Logistic Regression Configuration (Improved)
LR_CONFIG = {
    'C': 0.5,                   # Decreased from 1.0 (more regularization)
    'penalty': 'l2',
    'max_iter': 2000,           # Increased from 1000
    'class_weight': 'balanced', # NEW: Handle class imbalance
    'random_state': 42
}

# Calibration Configuration (NEW)
CALIBRATION_CONFIG = {
    'method': 'isotonic',       # Better for ensemble models
    'cv': 5
}

# ============================================================================
# Betting Strategy Configuration (UPDATED)
# ============================================================================

KELLY_CONFIG = {
    'kelly_fraction': 0.25,     # Use 1/4 Kelly
    'min_edge': 0.05,           # INCREASED from 0.02 to 0.05 (5% minimum)
    'max_bet_pct': 0.03,        # NEW: Max 3% of bankroll per bet
    'min_bet_amount': 5.0,      # NEW: Don't bet less than $5
    'default_bankroll': 1000.0
}

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE_DAYS = int(os.getenv('LOG_FILE_DAYS', '30'))
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# Historical Data Configuration
# ============================================================================

# Seasons to fetch (Format: YYYY-YY)
HISTORICAL_SEASONS_STR = os.getenv('HISTORICAL_SEASONS', '2024-25,2023-24,2022-23')
HISTORICAL_SEASONS = [s.strip() for s in HISTORICAL_SEASONS_STR.split(',')]

# ============================================================================
# ELO Configuration (Enhanced)
# ============================================================================

ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 100        # Increased from 65 (more realistic)
ELO_INITIAL_RATING = 1500
ELO_SEASON_REGRESSION = 0.75

# ============================================================================
# Advanced Metrics Configuration (NEW)
# ============================================================================

ADVANCED_METRICS_CONFIG = {
    'window': 10,                    # Last 10 games for metrics
    'league_avg_pace': 100.0,        # Average possessions per game
    'league_avg_ortg': 110.0,        # Average offensive rating
    'league_avg_drtg': 110.0,        # Average defensive rating
    'min_games_for_metrics': 5,      # Minimum games needed
}

# Home Court Advantage Calculation (NEW)
HCA_CONFIG = {
    'window': 20,                    # Last 20 games for HCA calculation
    'league_avg_hca': 3.5,           # League average in points
    'min_hca': 0.0,                  # Minimum HCA value
    'max_hca': 8.0,                  # Maximum HCA value
}

# Head-to-Head Configuration (NEW)
H2H_CONFIG = {
    'window': 10,                    # Last 10 matchups
    'default_win_rate': 0.5,         # Default if no H2H data
}

# ============================================================================
# NBA Teams Data
# ============================================================================

# NBA Teams mapping: code -> full name
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets', 
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

# Reverse mapping
NBA_TEAMS_REVERSE = {v: k for k, v in NBA_TEAMS.items()}

# Team name variations
TEAM_NAME_VARIATIONS = {
    'lakers': 'LAL', 'la lakers': 'LAL', 'los angeles lakers': 'LAL',
    'clippers': 'LAC', 'la clippers': 'LAC', 'los angeles clippers': 'LAC',
    'warriors': 'GSW', 'golden state': 'GSW',
    'nets': 'BKN', 'brooklyn': 'BKN', 'bkn': 'BKN', 'brk': 'BKN',
    'celtics': 'BOS', 'boston': 'BOS',
    'knicks': 'NYK', 'new york': 'NYK',
    'heat': 'MIA', 'miami': 'MIA',
    'bulls': 'CHI', 'chicago': 'CHI',
    'cavaliers': 'CLE', 'cavs': 'CLE', 'cleveland': 'CLE',
    'mavericks': 'DAL', 'mavs': 'DAL', 'dallas': 'DAL',
    'nuggets': 'DEN', 'denver': 'DEN',
    'rockets': 'HOU', 'houston': 'HOU',
    'thunder': 'OKC', 'okc': 'OKC', 'oklahoma city': 'OKC',
    'blazers': 'POR', 'trail blazers': 'POR', 'portland': 'POR',
    'spurs': 'SAS', 'san antonio': 'SAS',
    'suns': 'PHX', 'phoenix': 'PHX',
    'kings': 'SAC', 'sacramento': 'SAC',
    'jazz': 'UTA', 'utah': 'UTA',
    'bucks': 'MIL', 'milwaukee': 'MIL',
    'pistons': 'DET', 'detroit': 'DET',
    'pacers': 'IND', 'indiana': 'IND',
    'hornets': 'CHA', 'charlotte': 'CHA',
    'magic': 'ORL', 'orlando': 'ORL',
    'hawks': 'ATL', 'atlanta': 'ATL',
    'wizards': 'WAS', 'washington': 'WAS',
    'grizzlies': 'MEM', 'memphis': 'MEM',
    'pelicans': 'NOP', 'new orleans': 'NOP',
    'timberwolves': 'MIN', 'wolves': 'MIN', 'minnesota': 'MIN',
    'raptors': 'TOR', 'toronto': 'TOR',
    '76ers': 'PHI', 'sixers': 'PHI', 'philadelphia': 'PHI'
}

# ============================================================================
# Feature Importance Tracking (NEW)
# ============================================================================

# Track which features are most predictive
FEATURE_CATEGORIES = {
    'ELO': ELO_FEATURES,
    'Form & Rest': FORM_REST_FEATURES,
    'Home Court': HCA_FEATURES,
    'Advanced Metrics': ADVANCED_METRICS_FEATURES,
    'Situational': SITUATIONAL_FEATURES,
}

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories():
    """Create all required directories if they don't exist"""
    directories = [
        DATA_DIR, MODELS_DIR, LOGS_DIR,
        HISTORICAL_DATA_DIR, CURRENT_DATA_DIR, UPCOMING_DATA_DIR,
        PROCESSED_DATA_DIR, PREDICTIONS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Validate Kelly betting parameters
    if KELLY_CONFIG['min_edge'] < 0.03:
        errors.append("Warning: min_edge < 3% is very aggressive")
    
    if KELLY_CONFIG['kelly_fraction'] > 0.5:
        errors.append("Warning: kelly_fraction > 0.5 is risky")
    
    # Validate feature counts
    if len(FEATURES) < 10:
        errors.append("Warning: Less than 10 features may reduce accuracy")
    
    return errors

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        'API Status': 'Free Tier (NBA Stats + ESPN + BallDontLie)',
        'Directories': {
            'Data': str(DATA_DIR),
            'Models': str(MODELS_DIR),
        },
        'Model': {
            'Features': len(FEATURES),
            'Min Games': MIN_GAMES_FOR_TRAINING,
            'Retrain Days': MODEL_RETRAIN_DAYS,
        },
        'Betting': {
            'Min Edge': f"{KELLY_CONFIG['min_edge']*100}%",
            'Kelly Fraction': KELLY_CONFIG['kelly_fraction'],
            'Max Bet %': f"{KELLY_CONFIG['max_bet_pct']*100}%",
        },
        'Historical Seasons': HISTORICAL_SEASONS,
        'Feature Categories': {k: len(v) for k, v in FEATURE_CATEGORIES.items()},
    }

def print_config_info():
    """Print configuration information"""
    print("\n" + "="*70)
    print("NBA PREDICTION SYSTEM - CONFIGURATION")
    print("="*70)
    
    summary = get_config_summary()
    
    print(f"\n📊 MODEL CONFIGURATION:")
    print(f"  • Total Features: {summary['Model']['Features']}")
    print(f"  • Min Training Games: {summary['Model']['Min Games']}")
    print(f"  • Retrain Interval: {summary['Model']['Retrain Days']} days")
    
    print(f"\n💰 BETTING CONFIGURATION:")
    print(f"  • Minimum Edge: {summary['Betting']['Min Edge']}")
    print(f"  • Kelly Fraction: {summary['Betting']['Kelly Fraction']} (Conservative)")
    print(f"  • Max Bet Size: {summary['Betting']['Max Bet %']}")
    
    print(f"\n🏀 FEATURE BREAKDOWN:")
    for category, count in summary['Feature Categories'].items():
        print(f"  • {category}: {count} features")
    
    print(f"\n📁 DATA SOURCES:")
    print(f"  • Status: {summary['API Status']}")
    print(f"  • Historical Seasons: {', '.join(summary['Historical Seasons'])}")
    
    # Validation warnings
    errors = validate_config()
    if errors:
        print(f"\n⚠️  CONFIGURATION WARNINGS:")
        for error in errors:
            print(f"  • {error}")
    
    print("\n" + "="*70 + "\n")

# Create directories on import
ensure_directories()

# Optional: Print config on import (comment out if too verbose)
if __name__ == "__main__":
    print_config_info()