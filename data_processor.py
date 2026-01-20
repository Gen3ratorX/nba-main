"""
NBA Data Processor - Process standardized CSV data into Advanced Features
"""
import pandas as pd
from config import *
from nbautils import (
    log_info, log_error, log_warning, update_elo, calculate_form, 
    calculate_rest_days, calculate_advanced_metrics,
    init_elo, save_json,
    is_back_to_back, get_h2h_record, calculate_home_court_strength  # NEW IMPORTS
)
from datetime import datetime

class NBADataProcessor:
    def __init__(self):
        ensure_directories()
        self.nba_teams = list(NBA_TEAMS.keys())
        self.elo_ratings = init_elo(self.nba_teams, base=1500)
        log_info("NBA Data Processor initialized (Advanced Mode)")
    
    def process_all_data(self) -> pd.DataFrame:
        """Main processing pipeline"""
        log_info("Starting data processing...")
        
        # 1. Load Data
        dfs = []
        for d_dir in [HISTORICAL_DATA_DIR, CURRENT_DATA_DIR]:
            for f in d_dir.glob("*.csv"):
                try: dfs.append(pd.read_csv(f))
                except: pass
        
        if not dfs: return pd.DataFrame()
            
        # 2. Sort
        games_df = pd.concat(dfs, ignore_index=True)
        games_df['game_date'] = pd.to_datetime(games_df['game_date'])
        games_df = games_df.sort_values('game_date').drop_duplicates(
            subset=['game_date', 'home_team', 'away_team']
        ).reset_index(drop=True)
        
        # 3. Calculate Advanced Features
        features_list = []
        current_elo = self.elo_ratings.copy()
        
        for idx, game in games_df.iterrows():
            if game['home_score'] == 0 and game['away_score'] == 0: 
                continue
                
            home, away = game['home_team'], game['away_team']
            date = game['game_date']
            
            # Get Past Games for Metrics
            past_games = games_df[games_df.index < idx]
            
            # Basic Features
            home_elo = current_elo.get(home, 1500)
            away_elo = current_elo.get(away, 1500)
            
            # Advanced Metrics Calculation
            home_adv = calculate_advanced_metrics(past_games, home)
            away_adv = calculate_advanced_metrics(past_games, away)
            
            feat = {
                'game_date': date,
                'home_team': home, 'away_team': away,
                'home_score': game['home_score'], 'away_score': game['away_score'],
                'home_won': 1 if game['home_score'] > game['away_score'] else 0,
                
                # ELO
                'home_elo': home_elo, 'away_elo': away_elo,
                'elo_diff': home_elo - away_elo,
                
                # Form & Rest
                'form_home': calculate_form(past_games, home),
                'form_away': calculate_form(past_games, away),
                'form_diff': calculate_form(past_games, home) - calculate_form(past_games, away),
                'rest_home': calculate_rest_days(past_games, home, date),
                'rest_away': calculate_rest_days(past_games, away, date),
                'rest_diff': calculate_rest_days(past_games, home, date) - calculate_rest_days(past_games, away, date),
                
                # FIXED: Home Court Advantage (now calculated properly)
                'home_court_advantage': calculate_home_court_strength(past_games, home),
                
                # Advanced Metrics
                'off_rating_home': home_adv['offensive_rating'],
                'off_rating_away': away_adv['offensive_rating'],
                'def_rating_home': home_adv['defensive_rating'],
                'def_rating_away': away_adv['defensive_rating'],
                'net_rating_home': home_adv['net_rating'],
                'net_rating_away': away_adv['net_rating'],
                'pace_home': home_adv['pace'],
                'pace_away': away_adv['pace'],
                'win_streak_home': home_adv['win_streak'],
                'win_streak_away': away_adv['win_streak'],
                
                # NEW FEATURES
                'back_to_back_home': is_back_to_back(past_games, home, date),
                'back_to_back_away': is_back_to_back(past_games, away, date),
                'h2h_win_rate': get_h2h_record(past_games, home, away),
            }
            
            features_list.append(feat)
            update_elo(current_elo, home, away, game['home_score'], game['away_score'])
            
        features_df = pd.DataFrame(features_list)
        self.elo_ratings = current_elo # Save state
        
        log_info(f"Processed {len(features_df)} games with enhanced features")
        return features_df

    def get_upcoming_games_features(self) -> pd.DataFrame:
        """Generate features for upcoming games"""
        upcoming_file = UPCOMING_DATA_DIR / "upcoming_games.csv"
        if not upcoming_file.exists(): 
            log_warning("No upcoming games file found")
            return pd.DataFrame()
            
        upcoming_df = pd.read_csv(upcoming_file)
        
        # Reload history for accurate "current" stats
        dfs = []
        for d_dir in [HISTORICAL_DATA_DIR, CURRENT_DATA_DIR]:
             for f in d_dir.glob("*.csv"):
                try: dfs.append(pd.read_csv(f))
                except: pass
        
        history_df = pd.DataFrame()
        if dfs:
            history_df = pd.concat(dfs, ignore_index=True)
            history_df['game_date'] = pd.to_datetime(history_df['game_date'])
            history_df = history_df.sort_values('game_date')

        features_list = []
        date_now = datetime.now()
        
        for _, game in upcoming_df.iterrows():
            home, away = game['home_team'], game['away_team']
            
            # ELO
            h_elo = self.elo_ratings.get(home, 1500)
            a_elo = self.elo_ratings.get(away, 1500)
            
            # Advanced Metrics
            h_adv = calculate_advanced_metrics(history_df, home)
            a_adv = calculate_advanced_metrics(history_df, away)
            
            feat = {
                'game_date': game['game_date'],
                'home_team': home, 'away_team': away,
                'home_elo': h_elo, 'away_elo': a_elo,
                'elo_diff': h_elo - a_elo,
                'form_home': calculate_form(history_df, home),
                'form_away': calculate_form(history_df, away),
                'form_diff': calculate_form(history_df, home) - calculate_form(history_df, away),
                'rest_home': calculate_rest_days(history_df, home, date_now),
                'rest_away': calculate_rest_days(history_df, away, date_now),
                'rest_diff': calculate_rest_days(history_df, home, date_now) - calculate_rest_days(history_df, away, date_now),
                
                # FIXED: Home Court Advantage
                'home_court_advantage': calculate_home_court_strength(history_df, home),
                
                # Advanced
                'off_rating_home': h_adv['offensive_rating'],
                'off_rating_away': a_adv['offensive_rating'],
                'def_rating_home': h_adv['defensive_rating'],
                'def_rating_away': a_adv['defensive_rating'],
                'net_rating_home': h_adv['net_rating'],
                'net_rating_away': a_adv['net_rating'],
                'pace_home': h_adv['pace'],
                'pace_away': a_adv['pace'],
                'win_streak_home': h_adv['win_streak'],
                'win_streak_away': a_adv['win_streak'],
                
                # NEW FEATURES
                'back_to_back_home': is_back_to_back(history_df, home, date_now),
                'back_to_back_away': is_back_to_back(history_df, away, date_now),
                'h2h_win_rate': get_h2h_record(history_df, home, away),
            }
            features_list.append(feat)
        
        log_info(f"Generated features for {len(features_list)} upcoming games")
        return pd.DataFrame(features_list)