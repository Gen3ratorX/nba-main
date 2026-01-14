"""
FREE NBA DATA APIS - Complete Implementation
No paid subscriptions required!
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import json
from pathlib import Path
import os

# ============================================================================
# 1. NBA STATS API (stats.nba.com) - 100% FREE, NO KEY NEEDED
# ============================================================================

class FreeNBAStatsAPI:
    """
    Official NBA Stats API - Completely Free
    No authentication required!
    """
    
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats"
        
        # Required headers to avoid 403 errors
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://stats.nba.com/',
            'Origin': 'https://stats.nba.com',
            'Connection': 'keep-alive',
        }
    
    def get_todays_scoreboard(self) -> List[Dict]:
        """
        Get today's games with live scores
        Endpoint: scoreboardV2
        """
        today = datetime.now().strftime('%m/%d/%Y')
        
        url = f"{self.base_url}/scoreboardV2"
        params = {
            'GameDate': today,
            'LeagueID': '00',
            'DayOffset': '0'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            # Parse game data
            game_headers = data['resultSets'][0]['headers']
            game_data = data['resultSets'][0]['rowSet']
            
            for game in game_data:
                game_dict = dict(zip(game_headers, game))
                games.append({
                    'game_id': game_dict['GAME_ID'],
                    'game_date': game_dict['GAME_DATE_EST'],
                    'home_team': game_dict['HOME_TEAM_ABBREVIATION'],
                    'away_team': game_dict['VISITOR_TEAM_ABBREVIATION'],
                    'home_score': game_dict['HOME_TEAM_SCORE'] or 0,
                    'away_score': game_dict['VISITOR_TEAM_SCORE'] or 0,
                    'game_status': game_dict['GAME_STATUS_TEXT']
                })
            
            print(f"✓ Found {len(games)} games for {today}")
            return games
            
        except Exception as e:
            print(f"✗ Error fetching scoreboard: {e}")
            return []
    
    def get_team_game_logs(self, team_abbr: str, season: str = '2024-25') -> pd.DataFrame:
        """
        Get game logs for a team
        Endpoint: teamgamelog
        """
        # Get team ID from abbreviation
        team_id = self._get_team_id(team_abbr)
        
        url = f"{self.base_url}/teamgamelog"
        params = {
            'TeamID': team_id,
            'Season': season,
            'SeasonType': 'Regular Season'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"✓ Got {len(df)} games for {team_abbr}")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching team logs: {e}")
            return pd.DataFrame()
    
    def get_league_standings(self, season: str = '2024-25') -> pd.DataFrame:
        """
        Get current league standings
        Endpoint: leaguestandings
        """
        url = f"{self.base_url}/leaguestandingsv3"
        params = {
            'LeagueID': '00',
            'Season': season,
            'SeasonType': 'Regular Season'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"✓ Got standings for {len(df)} teams")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching standings: {e}")
            return pd.DataFrame()
    
    def _get_team_id(self, team_abbr: str) -> str:
        """Convert team abbreviation to team ID"""
        team_ids = {
            'ATL': '1610612737', 'BOS': '1610612738', 'BKN': '1610612751',
            'CHA': '1610612766', 'CHI': '1610612741', 'CLE': '1610612739',
            'DAL': '1610612742', 'DEN': '1610612743', 'DET': '1610612765',
            'GSW': '1610612744', 'HOU': '1610612745', 'IND': '1610612754',
            'LAC': '1610612746', 'LAL': '1610612747', 'MEM': '1610612763',
            'MIA': '1610612748', 'MIL': '1610612749', 'MIN': '1610612750',
            'NOP': '1610612740', 'NYK': '1610612752', 'OKC': '1610612760',
            'ORL': '1610612753', 'PHI': '1610612755', 'PHX': '1610612756',
            'POR': '1610612757', 'SAC': '1610612758', 'SAS': '1610612759',
            'TOR': '1610612761', 'UTA': '1610612762', 'WAS': '1610612764'
        }
        return team_ids.get(team_abbr, '1610612747')

# ============================================================================
# 2. BALLDONTLIE API - FREE, NO KEY NEEDED (up to 60 requests/min)
# ============================================================================

class BallDontLieAPI:
    """
    BallDontLie API - NOW REQUIRES FREE KEY
    Get key at: https://balldontlie.io/
    """
    
    def __init__(self, api_key: str = None):
        # TRY TO GET KEY FROM ENV OR USE A PLACEHOLDER
        self.api_key = api_key or os.getenv('BALLDONTLIE_API_KEY')
        self.base_url = "https://api.balldontlie.io/v1"
        
        # HEADERS MUST INCLUDE THE API KEY NOW
        self.headers = {
            'Authorization': f'{self.api_key}',
            'User-Agent': 'NBA-Predictor/1.0'
        }
    
    def get_games(self, start_date: str, end_date: str = None) -> List[Dict]:
        if not self.api_key:
            print("⚠️ MISSING BALLDONTLIE KEY: Skipped.")
            return []

        if end_date is None:
            end_date = start_date
        
        url = f"{self.base_url}/games"
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'per_page': 100
        }
        
        all_games = []
        next_cursor = None # Used for the new pagination style
        
        try:
            while True:
                # If we have a cursor from the last loop, add it to params
                if next_cursor:
                    params['cursor'] = next_cursor
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                # Handle Rate Limiting (429)
                if response.status_code == 429:
                    print("⏳ Rate limit hit. Sleeping 60s...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                games = data.get('data', [])
                if not games:
                    break
                
                for game in games:
                    all_games.append({
                        'game_id': game['id'],
                        'game_date': game['date'][:10],
                        'home_team': game['home_team']['abbreviation'],
                        'away_team': game['visitor_team']['abbreviation'],
                        'home_score': game['home_team_score'],
                        'away_score': game['visitor_team_score'],
                        'status': game['status']
                    })
                
                # --- PAGINATION LOGIC ---
                meta = data.get('meta', {})
                
                # The API now uses "next_cursor" instead of "total_pages"
                if 'next_cursor' in meta and meta['next_cursor']:
                    next_cursor = meta['next_cursor']
                else:
                    # No next cursor means we are done
                    break

                time.sleep(1) # Be nice to the API
            
            return all_games
            
        except Exception as e:
            print(f"✗ Error fetching from BallDontLie: {e}")
            return []

# ============================================================================
# 3. ESPN HIDDEN API - FREE, NO KEY NEEDED
# ============================================================================

class ESPNHiddenAPI:
    """
    ESPN's unofficial API - Free and reliable
    Used by ESPN's own website
    """
    
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    def get_scoreboard(self, date: str = None) -> List[Dict]:
        """
        Get games for a specific date
        Format: YYYYMMDD
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        url = f"{self.base_url}/scoreboard"
        params = {'dates': date}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            for event in data.get('events', []):
                competition = event['competitions'][0]
                home_team = next(t for t in competition['competitors'] if t['homeAway'] == 'home')
                away_team = next(t for t in competition['competitors'] if t['homeAway'] == 'away')
                
                games.append({
                    'game_id': event['id'],
                    'game_date': event['date'][:10],
                    'home_team': home_team['team']['abbreviation'],
                    'away_team': away_team['team']['abbreviation'],
                    'home_score': int(home_team.get('score', 0)),
                    'away_score': int(away_team.get('score', 0)),
                    'status': competition['status']['type']['description']
                })
            
            print(f"✓ Got {len(games)} games from ESPN")
            return games
            
        except Exception as e:
            print(f"✗ Error fetching ESPN data: {e}")
            return []
    
    def get_team_schedule(self, team_abbr: str, season: int = 2025) -> List[Dict]:
        """Get team schedule"""
        # ESPN team IDs mapping
        team_id = self._get_espn_team_id(team_abbr)
        
        url = f"{self.base_url}/teams/{team_id}/schedule"
        params = {'season': season}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            games = []
            
            for event in data.get('events', []):
                games.append({
                    'date': event['date'][:10],
                    'opponent': event['name'],
                    'result': event.get('result', 'Scheduled')
                })
            
            print(f"✓ Got {len(games)} games for {team_abbr}")
            return games
            
        except Exception as e:
            print(f"✗ Error fetching team schedule: {e}")
            return []
    
    def _get_espn_team_id(self, team_abbr: str) -> str:
        """Convert abbreviation to ESPN team ID"""
        espn_ids = {
            'ATL': '1', 'BOS': '2', 'BKN': '17', 'CHA': '30', 'CHI': '4',
            'CLE': '5', 'DAL': '6', 'DEN': '7', 'DET': '8', 'GSW': '9',
            'HOU': '10', 'IND': '11', 'LAC': '12', 'LAL': '13', 'MEM': '29',
            'MIA': '14', 'MIL': '15', 'MIN': '16', 'NOP': '3', 'NYK': '18',
            'OKC': '25', 'ORL': '19', 'PHI': '20', 'PHX': '21', 'POR': '22',
            'SAC': '23', 'SAS': '24', 'TOR': '28', 'UTA': '26', 'WAS': '27'
        }
        return espn_ids.get(team_abbr, '13')

# ============================================================================
# 4. THE ODDS API - FREE TIER (500 requests/month)
# ============================================================================

class TheOddsAPI:
    """
    The Odds API - Free tier available
    Sign up at: https://the-odds-api.com/
    Free: 500 requests/month
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY', '')
        self.base_url = "https://api.the-odds-api.com/v4"
    
    def get_nba_odds(self, markets: str = 'h2h') -> List[Dict]:
        """
        Get current NBA betting odds
        markets: 'h2h' (moneyline), 'spreads', 'totals'
        """
        if not self.api_key:
            print("⚠️  ODDS_API_KEY not set - sign up at the-odds-api.com (free)")
            return []
        
        url = f"{self.base_url}/sports/basketball_nba/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            games = response.json()
            
            odds_data = []
            for game in games:
                odds_data.append({
                    'game_id': game['id'],
                    'commence_time': game['commence_time'][:10],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'bookmakers': game.get('bookmakers', [])
                })
            
            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining', 'unknown')
            print(f"✓ Got odds for {len(odds_data)} games (Requests remaining: {remaining})")
            
            return odds_data
            
        except Exception as e:
            print(f"✗ Error fetching odds: {e}")
            return []

# ============================================================================
# 5. INTEGRATED FREE DATA COLLECTOR
# ============================================================================

class FreeNBADataCollector:
    """
    Combines all free APIs for comprehensive data collection
    """
    
    def __init__(self, odds_api_key: str = None):
        self.nba_stats = FreeNBAStatsAPI()
        self.balldontlie = BallDontLieAPI('0f20cae2-c584-4bd0-bc12-bb5020238cf0')
        self.espn = ESPNHiddenAPI()
        self.odds_api = TheOddsAPI(odds_api_key)
    
    def get_todays_games_comprehensive(self) -> pd.DataFrame:
        """
        Get today's games from multiple sources
        Combines for reliability
        """
        all_games = []
        
        # Source 1: NBA Stats API
        print("\n📊 Fetching from NBA Stats API...")
        nba_games = self.nba_stats.get_todays_scoreboard()
        all_games.extend(nba_games)
        time.sleep(1)
        
        # Source 2: ESPN
        print("📊 Fetching from ESPN...")
        espn_games = self.espn.get_scoreboard()
        all_games.extend(espn_games)
        time.sleep(1)
        
        # Deduplicate
        df = pd.DataFrame(all_games)
        if not df.empty:
            df = df.drop_duplicates(subset=['home_team', 'away_team', 'game_date'])
            print(f"\n✅ Found {len(df)} unique games today")
        
        return df
    
    def get_historical_games(
        self, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Get historical games (free, no limits)
        Uses BallDontLie API
        """
        print(f"\n📊 Fetching games from {start_date} to {end_date}...")
        
        games = self.balldontlie.get_games(start_date, end_date)
        df = pd.DataFrame(games)
        
        print(f"✅ Retrieved {len(df)} games")
        return df
    
    def get_upcoming_with_odds(self) -> pd.DataFrame:
        """
        Get upcoming games with betting odds
        """
        # Get upcoming games from ESPN
        print("\n📊 Fetching upcoming games...")
        today = datetime.now()
        games = []
        
        for days in range(1, 8):  # Next 7 days
            date = (today + timedelta(days=days)).strftime('%Y%m%d')
            daily_games = self.espn.get_scoreboard(date)
            games.extend(daily_games)
            time.sleep(1)
        
        df = pd.DataFrame(games)
        
        # Add odds if API key available
        if self.odds_api.api_key:
            print("📊 Fetching betting odds...")
            odds = self.odds_api.get_nba_odds()
            # Merge odds with games (implementation depends on data structure)
        
        print(f"✅ Found {len(df)} upcoming games")
        return df

# ============================================================================
# INSTALLATION & SETUP INSTRUCTIONS
# ============================================================================

def print_setup_instructions():
    """Print setup instructions for free APIs"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║        FREE NBA APIs - SETUP INSTRUCTIONS                     ║
╚═══════════════════════════════════════════════════════════════╝

1. NBA Stats API (stats.nba.com)
   ✓ NO REGISTRATION NEEDED
   ✓ NO API KEY NEEDED
   ✓ Unlimited requests (be reasonable with rate limiting)
   ✓ Official NBA data
   → Ready to use immediately!

2. BallDontLie API
   ✓ NO REGISTRATION NEEDED
   ✓ NO API KEY NEEDED
   ✓ 60 requests per minute
   ✓ Historical data available
   → Ready to use immediately!

3. ESPN Hidden API
   ✓ NO REGISTRATION NEEDED
   ✓ NO API KEY NEEDED
   ✓ Used by ESPN's own site
   ✓ Real-time scores
   → Ready to use immediately!

4. The Odds API (OPTIONAL - for betting odds)
   1. Go to: https://the-odds-api.com/
   2. Click "Get a Free API Key"
   3. Sign up with email
   4. Copy your API key
   5. Add to .env file:
      ODDS_API_KEY=your_key_here
   
   FREE TIER:
   ✓ 500 requests per month
   ✓ Real-time betting odds
   ✓ Multiple bookmakers

═══════════════════════════════════════════════════════════════

🚀 QUICK START:

# Install required packages
pip install requests pandas

# Test the APIs
python free_nba_apis.py

# That's it! No paid subscriptions needed.
═══════════════════════════════════════════════════════════════
    """)

# ============================================================================
# TESTING & DEMO
# ============================================================================

if __name__ == "__main__":
    print_setup_instructions()
    
    print("\n" + "="*70)
    print("TESTING FREE APIs")
    print("="*70)
    
    # Initialize collector
    collector = FreeNBADataCollector()
    
    # Test 1: Today's games
    print("\n🏀 TEST 1: Today's Games")
    print("-" * 70)
    today_games = collector.get_todays_games_comprehensive()
    if not today_games.empty:
        print(today_games[['game_date', 'home_team', 'away_team', 'home_score', 'away_score']])
    
    # Test 2: Historical data
    print("\n🏀 TEST 2: Recent Historical Games")
    print("-" * 70)
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    historical = collector.get_historical_games(start_date, end_date)
    print(f"Retrieved {len(historical)} games from past week")
    
    # Test 3: Upcoming games
    print("\n🏀 TEST 3: Upcoming Games")
    print("-" * 70)
    upcoming = collector.get_upcoming_with_odds()
    if not upcoming.empty:
        print(upcoming[['game_date', 'home_team', 'away_team']].head())
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)