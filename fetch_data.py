"""
fetch_data.py - UPDATED TO USE 100% FREE APIs
Replace your existing fetch_data.py with this
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import time
from config import *
from nbautils import log_info, log_error, log_warning, normalize_team_name, ensure_directories

class FreeNBADataCollector:
    """NBA Data Collector using 100% FREE APIs"""
    
    def __init__(self):
        ensure_directories()
        self.current_season = self._get_current_season()
        
        # NBA Stats API headers
        self.nba_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://stats.nba.com/',
            'Origin': 'https://stats.nba.com',
        }
        log_info(f"Free NBA Data Collector initialized (Season: {self.current_season})")
    
    def _get_current_season(self) -> str:
        """Calculate current NBA season (e.g., '2024-25')"""
        now = datetime.now()
        if now.month >= 10:  # Season starts in October
            return f"{now.year}-{str(now.year + 1)[2:]}"
        else:
            return f"{now.year - 1}-{str(now.year)[2:]}"
    
    # ------------------------------------------------------------------------
    # 1. Fetch Historical Data (BallDontLie - Most Reliable)
    # ------------------------------------------------------------------------
    def fetch_historical_data(self) -> bool:
        """Fetch last 2+ seasons using BallDontLie API"""
        log_info("Fetching historical NBA data using BallDontLie API...")
        
        try:
            # Calculate dynamic date range
            current_season_start = datetime(datetime.now().year, 10, 1)
            if datetime.now().month < 10:
                current_season_start = datetime(datetime.now().year - 1, 10, 1)
            
            # Get last 2 full seasons + current season
            start_date = (current_season_start - timedelta(days=730)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            log_info(f"Fetching games from {start_date} to {end_date}")
            
            df = self.fetch_balldontlie_games(start_date, end_date)
            
            if not df.empty:
                # Filter out games with 0 scores (unplayed games)
                df = df[(df['home_score'] > 0) & (df['away_score'] > 0)]
                
                filename = f"historical_games_balldontlie.csv"
                filepath = HISTORICAL_DATA_DIR / filename
                df.to_csv(filepath, index=False)
                log_info(f"✅ Saved historical data: {len(df)} games")
                return True
            else:
                log_warning("No historical data retrieved")
                return False
            
        except Exception as e:
            log_error(f"Historical data collection failed: {e}")
            return False
    
    # ------------------------------------------------------------------------
    # 2. Fetch Current Season (ESPN - Most Reliable Free Source)
    # ------------------------------------------------------------------------
    def fetch_current_season_data(self, days_back: int = 60) -> bool:
        """Fetch recent games using ESPN API"""
        log_info(f"Fetching current season data (last {days_back} days via ESPN)...")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            all_games = []
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y%m%d')
                games = self._fetch_espn_scoreboard(date_str)
                
                # Only keep finished games (both scores > 0)
                finished_games = [g for g in games if g['home_score'] > 0 and g['away_score'] > 0]
                
                if finished_games:
                    all_games.extend(finished_games)
                    log_info(f"  {current_date.strftime('%Y-%m-%d')}: {len(finished_games)} games")
                
                current_date += timedelta(days=1)
                time.sleep(0.5)  # Be nice to ESPN's API
            
            if all_games:
                df = pd.DataFrame(all_games)
                df = df.drop_duplicates(subset=['game_id'])  # Remove duplicates
                
                filename = f"current_season_espn_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = CURRENT_DATA_DIR / filename
                df.to_csv(filepath, index=False)
                log_info(f"✅ Current season data saved: {len(df)} games")
                return True
            else:
                log_warning("No current season games found")
                return False
                
        except Exception as e:
            log_error(f"Current season fetch failed: {e}")
            return False

    # ------------------------------------------------------------------------
    # 3. Fetch Upcoming Games (ESPN)
    # ------------------------------------------------------------------------
    def fetch_upcoming_games(self, days_ahead: int = 7) -> bool:
        """Fetch upcoming games for next N days"""
        log_info(f"Fetching upcoming games (next {days_ahead} days via ESPN)...")
        
        try:
            all_games = []
            today = datetime.now()
            
            for days in range(0, days_ahead + 1):  # Include today
                target_date = today + timedelta(days=days)
                date_str = target_date.strftime('%Y%m%d')
                games = self._fetch_espn_scoreboard(date_str)
                
                if games:
                    # For upcoming games, we want ALL games (even unplayed)
                    all_games.extend(games)
                    log_info(f"  {target_date.strftime('%Y-%m-%d')}: {len(games)} games")
                
                time.sleep(1)  # Rate limiting
            
            if all_games:
                df = pd.DataFrame(all_games)
                df = df.drop_duplicates(subset=['game_id'])
                
                filename = "upcoming_games.csv"
                filepath = UPCOMING_DATA_DIR / filename
                df.to_csv(filepath, index=False)
                log_info(f"✅ Upcoming games saved: {len(df)} games")
                return True
            else:
                log_warning("No upcoming games found")
                return False
                
        except Exception as e:
            log_error(f"Upcoming games fetch failed: {e}")
            return False

    # ------------------------------------------------------------------------
    # Helper: ESPN Scoreboard API
    # ------------------------------------------------------------------------
    def _fetch_espn_scoreboard(self, date: str) -> List[Dict]:
        """Fetch games for a specific date from ESPN"""
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        params = {'dates': date}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for event in data.get('events', []):
                try:
                    competition = event['competitions'][0]
                    
                    # Find home and away teams
                    # ESPN API labels are CORRECT - no swap needed
                    home = next(t for t in competition['competitors'] if t['homeAway'] == 'home')
                    away = next(t for t in competition['competitors'] if t['homeAway'] == 'away')
                    
                    # Parse game date (ESPN format: 2024-01-11T02:00Z)
                    game_date_str = event['date'][:10]  # Take YYYY-MM-DD part
                    
                    games.append({
                        'game_id': event['id'],
                        'game_date': game_date_str,
                        'home_team': normalize_team_name(home['team']['abbreviation']),
                        'away_team': normalize_team_name(away['team']['abbreviation']),
                        'home_score': int(home.get('score', 0)),
                        'away_score': int(away.get('score', 0)),
                        'season': self.current_season,
                        'data_source': 'espn'
                    })
                except (KeyError, StopIteration, ValueError) as e:
                    log_warning(f"Skipped malformed game: {e}")
                    continue
            
            return games
            
        except requests.exceptions.RequestException as e:
            log_warning(f"ESPN API error for {date}: {e}")
            return []
        except Exception as e:
            log_error(f"Unexpected error parsing ESPN data: {e}")
            return []

    # ------------------------------------------------------------------------
    # Helper: BallDontLie API (Backup/Historical)
    # ------------------------------------------------------------------------
    def fetch_balldontlie_games(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch games from BallDontLie API with pagination"""
        log_info(f"Fetching from BallDontLie API ({start_date} to {end_date})...")
        
        api_key = BALLDONTLIE_API_KEY
        url = "https://api.balldontlie.io/v1/games"
        headers = {
            'Authorization': api_key,
            'User-Agent': 'NBA-Predictor/1.0'
        }
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'per_page': 100
        }
        
        all_games = []
        next_cursor = None
        page = 1
        
        try:
            while True:
                if next_cursor:
                    params['cursor'] = next_cursor
                
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    log_warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                games = data.get('data', [])
                if not games:
                    log_info(f"No more games found (page {page})")
                    break
                
                log_info(f"Page {page}: Retrieved {len(games)} games")
                
                for game in games:
                    try:
                        all_games.append({
                            'game_id': game['id'],
                            'game_date': game['date'][:10],
                            'home_team': normalize_team_name(game['home_team']['abbreviation']),
                            'away_team': normalize_team_name(game['visitor_team']['abbreviation']),
                            'home_score': game['home_team_score'] or 0,
                            'away_score': game['visitor_team_score'] or 0,
                            'season': game.get('season', self.current_season),
                            'data_source': 'balldontlie'
                        })
                    except (KeyError, TypeError) as e:
                        log_warning(f"Skipped malformed game: {e}")
                        continue
                
                # Check for next page
                meta = data.get('meta', {})
                if meta.get('next_cursor'):
                    next_cursor = meta['next_cursor']
                    page += 1
                    time.sleep(1)  # Rate limiting
                else:
                    log_info(f"All pages retrieved ({page} total)")
                    break
            
            df = pd.DataFrame(all_games)
            log_info(f"✅ Total games retrieved: {len(df)}")
            return df
            
        except requests.exceptions.RequestException as e:
            log_error(f"BallDontLie API error: {e}")
            return pd.DataFrame()
        except Exception as e:
            log_error(f"Unexpected error in BallDontLie fetch: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------------
    # Convenience Method: Fetch All Data
    # ------------------------------------------------------------------------
    def fetch_all_data(self, days_back: int = 60, days_ahead: int = 7) -> Dict[str, bool]:
        """Fetch all data: historical, current, and upcoming"""
        log_info("\n" + "="*70)
        log_info("STARTING COMPREHENSIVE DATA FETCH")
        log_info("="*70 + "\n")
        
        results = {
            'historical': False,
            'current': False,
            'upcoming': False
        }
        
        # 1. Historical data (if not recently fetched)
        historical_file = HISTORICAL_DATA_DIR / "historical_games_balldontlie.csv"
        if not historical_file.exists():
            log_info("📚 Fetching historical data...")
            results['historical'] = self.fetch_historical_data()
        else:
            log_info("📚 Historical data already exists (skipping)")
            results['historical'] = True
        
        # 2. Current season data
        log_info("\n📊 Fetching current season data...")
        results['current'] = self.fetch_current_season_data(days_back)
        
        # 3. Upcoming games
        log_info("\n🔮 Fetching upcoming games...")
        results['upcoming'] = self.fetch_upcoming_games(days_ahead)
        
        # Summary
        log_info("\n" + "="*70)
        log_info("DATA FETCH SUMMARY")
        log_info("="*70)
        log_info(f"Historical Data: {'✅ Success' if results['historical'] else '❌ Failed'}")
        log_info(f"Current Season:  {'✅ Success' if results['current'] else '❌ Failed'}")
        log_info(f"Upcoming Games:  {'✅ Success' if results['upcoming'] else '❌ Failed'}")
        log_info("="*70 + "\n")
        
        return results


# Compatibility Class (for backward compatibility)
class NBADataCollector(FreeNBADataCollector):
    """Alias for backward compatibility"""
    pass


# Test/Demo Function
if __name__ == "__main__":
    collector = FreeNBADataCollector()
    results = collector.fetch_all_data(days_back=30, days_ahead=7)
    
    if all(results.values()):
        print("\n🎉 All data fetched successfully!")
    else:
        print("\n⚠️  Some data fetches failed. Check logs for details.")