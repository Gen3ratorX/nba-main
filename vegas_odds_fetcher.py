"""
vegas_odds_fetcher.py - The Odds API Integration
Fetches real-time NBA odds (moneyline, spreads, totals)

API Docs: https://the-odds-api.com/liveapi/guides/v4/

Free Tier Limits:
- 500 requests per month
- Updates every 5 minutes
- Covers major US sportsbooks

Setup:
1. Get free API key: https://the-odds-api.com/
2. Add to .env file: ODDS_API_KEY=your_key_here
3. Or set environment variable
"""
import os
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VegasOddsFetcher:
    """Fetch NBA odds from The Odds API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        
        if not self.api_key:
            print("⚠️  WARNING: No ODDS_API_KEY found!")
            print("   Get one free at: https://the-odds-api.com/")
            print("   Add to .env file: ODDS_API_KEY=your_key_here")
        
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"
        
        # Sportsbooks to query (in order of preference)
        self.preferred_books = [
            'fanduel',      # FanDuel (most popular)
            'draftkings',   # DraftKings
            'betmgm',       # BetMGM
            'caesars',      # Caesars
            'pinnacle'      # Pinnacle (sharpest lines)
        ]
        
        # Cache to avoid burning API requests
        self.cache_file = Path('data/vegas_odds_cache.json')
        self.cache_duration = 300  # 5 minutes (API updates every 5 min)
        self.odds_history_file = Path('data/odds_history/odds_history.csv')
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached odds if fresh enough"""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            # Check if cache is still fresh
            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                print(f"✅ Using cached odds (fetched {int((datetime.now() - cache_time).total_seconds())}s ago)")
                return cache
            
            return {}
        except:
            return {}
    
    def _save_cache(self, data: Dict):
        """Save odds to cache"""
        cache = {
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'requests_remaining': data.get('requests_remaining', 'unknown')
        }
        
        self.cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

    def _load_stale_cache_data(self) -> Optional[Dict]:
        """Load cached odds data regardless of freshness (offline fallback)."""
        if not self.cache_file.exists():
            return None
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            return cache.get('data')
        except Exception:
            return None
    
    def get_nba_odds(self, markets: List[str] = None) -> Optional[Dict]:
        """
        Fetch NBA odds from The Odds API - FIXED VERSION
        
        Args:
            markets: List of markets to fetch. Options:
                    'h2h' = moneyline (1 credit)
                    'spreads' = point spreads (1 credit)
                    'totals' = over/under (1 credit)
                    
        Returns:
            Dict with odds data or None if error
        """
        stale_cache_data = self._load_stale_cache_data()

        # Check cache first
        if self.cache.get('data'):
            return self.cache['data']

        if not self.api_key:
            if stale_cache_data:
                print("⚠️  No API key; using stale cached odds data")
                return stale_cache_data
            print("❌ Cannot fetch odds: No API key configured")
            return None
        
        # Default to all markets (costs 3 credits per request)
        if markets is None:
            markets = ['h2h', 'spreads', 'totals']
        
        markets_str = ','.join(markets)
        
        # CORRECT ENDPOINT: /v4/sports/basketball_nba/odds
        url = f"{self.base_url}/sports/{self.sport}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',              # US sportsbooks (FanDuel, DraftKings, etc.)
            'markets': markets_str,        # h2h,spreads,totals
            'oddsFormat': 'american',      # American odds format (-110, +150, etc.)
            'dateFormat': 'iso'            # ISO 8601 date format
        }
        
        try:
            print(f"📡 Fetching live odds from The Odds API...")
            print(f"   Endpoint: {url}")
            print(f"   Markets: {markets_str} (costs {len(markets)} credits)")
            
            response = requests.get(url, params=params, timeout=10)
            
            # Check rate limit headers
            requests_remaining = response.headers.get('x-requests-remaining', 'unknown')
            requests_used = response.headers.get('x-requests-used', 'unknown')
            
            print(f"   API Credits: {requests_used} used, {requests_remaining} remaining this month")
            
            if response.status_code == 200:
                data = response.json()
                
                if not data:
                    print("⚠️  API returned empty data - no games available")
                    return None
                
                # Add metadata
                result = {
                    'games': data,
                    'fetched_at': datetime.now().isoformat(),
                    'requests_remaining': requests_remaining,
                    'requests_used': requests_used
                }
                
                # Save to cache
                self._save_cache(result)
                
                print(f"✅ Fetched odds for {len(data)} NBA games")
                return result
            
            elif response.status_code == 401:
                print("❌ Invalid API key")
                return stale_cache_data
            
            elif response.status_code == 422:
                print("❌ Invalid parameters (check market names or sport key)")
                return stale_cache_data
            
            elif response.status_code == 429:
                print("❌ Rate limit exceeded - wait until next month or upgrade plan")
                return stale_cache_data
            
            else:
                print(f"❌ API error: {response.status_code} - {response.text[:200]}")
                return stale_cache_data
        
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return stale_cache_data
        
        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
            return stale_cache_data
    
    def parse_game_odds(self, game_data: Dict) -> Dict:
        """
        Parse odds for a single game
        
        Returns:
            Dict with cleaned odds data
        """
        home_team = game_data.get('home_team', '')
        away_team = game_data.get('away_team', '')
        commence_time = game_data.get('commence_time', '')
        
        bookmakers = game_data.get('bookmakers', [])
        
        # Find best/preferred sportsbook
        best_book = None
        for book_name in self.preferred_books:
            best_book = next((b for b in bookmakers if b['key'] == book_name), None)
            if best_book:
                break
        
        # Fallback to first available book
        if not best_book and bookmakers:
            best_book = bookmakers[0]
        
        if not best_book:
            return {
                'home_team': home_team,
                'away_team': away_team,
                'game_time': commence_time,
                'available': False
            }
        
        # Extract markets
        markets = best_book.get('markets', [])
        
        # Parse moneyline (h2h)
        h2h_market = next((m for m in markets if m['key'] == 'h2h'), None)
        moneyline = self._parse_moneyline(h2h_market, home_team, away_team)
        
        # Parse spreads
        spreads_market = next((m for m in markets if m['key'] == 'spreads'), None)
        spreads = self._parse_spreads(spreads_market, home_team, away_team)
        
        # Parse totals
        totals_market = next((m for m in markets if m['key'] == 'totals'), None)
        totals = self._parse_totals(totals_market)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'game_time': commence_time,
            'sportsbook': best_book.get('title', 'Unknown'),
            'available': True,
            'moneyline': moneyline,
            'spreads': spreads,
            'totals': totals,
            'last_update': best_book.get('last_update', '')
        }
    
    def _parse_moneyline(self, market: Optional[Dict], home: str, away: str) -> Dict:
        """Parse moneyline odds"""
        if not market or 'outcomes' not in market:
            return {'home': None, 'away': None}
        
        outcomes = market['outcomes']
        home_odds = next((o['price'] for o in outcomes if o['name'] == home), None)
        away_odds = next((o['price'] for o in outcomes if o['name'] == away), None)
        
        return {
            'home': home_odds,
            'away': away_odds
        }
    
    def _parse_spreads(self, market: Optional[Dict], home: str, away: str) -> Dict:
        """Parse spread odds"""
        if not market or 'outcomes' not in market:
            return {'home_line': None, 'home_odds': None, 'away_line': None, 'away_odds': None}
        
        outcomes = market['outcomes']
        home_outcome = next((o for o in outcomes if o['name'] == home), None)
        away_outcome = next((o for o in outcomes if o['name'] == away), None)
        
        return {
            'home_line': home_outcome.get('point') if home_outcome else None,
            'home_odds': home_outcome.get('price') if home_outcome else None,
            'away_line': away_outcome.get('point') if away_outcome else None,
            'away_odds': away_outcome.get('price') if away_outcome else None
        }
    
    def _parse_totals(self, market: Optional[Dict]) -> Dict:
        """Parse totals (over/under)"""
        if not market or 'outcomes' not in market:
            return {'line': None, 'over_odds': None, 'under_odds': None}
        
        outcomes = market['outcomes']
        over = next((o for o in outcomes if o['name'] == 'Over'), None)
        under = next((o for o in outcomes if o['name'] == 'Under'), None)
        
        # Line should be the same for both
        line = over.get('point') if over else (under.get('point') if under else None)
        
        return {
            'line': line,
            'over_odds': over.get('price') if over else None,
            'under_odds': under.get('price') if under else None
        }
    
    def get_odds_for_today(self) -> pd.DataFrame:
        """
        Get all odds for today's games as a DataFrame
        
        Returns:
            DataFrame with columns: home_team, away_team, ml_home, ml_away, 
                                   total_line, spread_home, etc.
        """
        odds_data = self.get_nba_odds()
        
        if not odds_data or 'games' not in odds_data:
            print("⚠️  No odds data available")
            return pd.DataFrame()
        
        games = odds_data['games']
        parsed_games = []
        
        for game in games:
            parsed = self.parse_game_odds(game)
            
            if parsed['available']:
                row = {
                    'home_team': self._normalize_team_name(parsed['home_team']),
                    'away_team': self._normalize_team_name(parsed['away_team']),
                    'game_time': parsed['game_time'],
                    'sportsbook': parsed['sportsbook'],
                    
                    # Moneyline
                    'ml_home': parsed['moneyline']['home'],
                    'ml_away': parsed['moneyline']['away'],
                    
                    # Spreads
                    'spread_home': parsed['spreads']['home_line'],
                    'spread_home_odds': parsed['spreads']['home_odds'],
                    'spread_away': parsed['spreads']['away_line'],
                    'spread_away_odds': parsed['spreads']['away_odds'],
                    
                    # Totals
                    'total_line': parsed['totals']['line'],
                    'over_odds': parsed['totals']['over_odds'],
                    'under_odds': parsed['totals']['under_odds'],
                }
                
                parsed_games.append(row)
        
        df = pd.DataFrame(parsed_games)
        
        if not df.empty:
            # Sort by game time
            df['game_time'] = pd.to_datetime(df['game_time'])
            df = df.sort_values('game_time')
            df['game_date'] = df['game_time'].dt.date.astype(str)
            df['fetched_at'] = odds_data.get('fetched_at', datetime.now().isoformat())
            self._archive_odds_snapshot(df)
        
        return df

    def _archive_odds_snapshot(self, df: pd.DataFrame):
        """
        Append parsed odds to long-term local history for backtesting.
        Keeps latest snapshot per (game_date, home_team, away_team, fetched_at).
        """
        if df.empty:
            return

        cols = [
            'game_date', 'game_time', 'fetched_at', 'sportsbook',
            'home_team', 'away_team',
            'ml_home', 'ml_away',
            'spread_home', 'spread_home_odds', 'spread_away', 'spread_away_odds',
            'total_line', 'over_odds', 'under_odds',
        ]

        snapshot = df.copy()
        for c in cols:
            if c not in snapshot.columns:
                snapshot[c] = None
        snapshot = snapshot[cols]

        self.odds_history_file.parent.mkdir(parents=True, exist_ok=True)

        if self.odds_history_file.exists():
            try:
                existing = pd.read_csv(self.odds_history_file)
                merged = pd.concat([existing, snapshot], ignore_index=True)
            except Exception:
                merged = snapshot
        else:
            merged = snapshot

        merged = merged.drop_duplicates(
            subset=['game_date', 'home_team', 'away_team', 'fetched_at'],
            keep='last'
        )
        merged.to_csv(self.odds_history_file, index=False)
    
    def _normalize_team_name(self, name: str) -> str:
        """Convert The Odds API team names to our 3-letter codes"""
        # The Odds API uses full team names like "Boston Celtics"
        # We need to convert to our codes like "BOS"
        
        mapping = {
            'Atlanta Hawks': 'ATL',
            'Boston Celtics': 'BOS',
            'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA',
            'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL',
            'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW',
            'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND',
            'Los Angeles Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL',
            'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA',
            'Milwaukee Bucks': 'MIL',
            'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK',
            'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI',
            'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC',
            'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA',
            'Washington Wizards': 'WAS'
        }
        
        return mapping.get(name, name)
    
    def get_total_for_game(self, home_team: str, away_team: str) -> Optional[float]:
        """
        Get the total line for a specific game
        
        Args:
            home_team: 3-letter code (e.g., 'BOS')
            away_team: 3-letter code (e.g., 'LAL')
            
        Returns:
            Total line (float) or None
        """
        df = self.get_odds_for_today()
        
        if df.empty:
            return None
        
        # Try to find matching game
        game = df[
            (df['home_team'] == home_team) & 
            (df['away_team'] == away_team)
        ]
        
        if game.empty:
            return None
        
        return game.iloc[0]['total_line']
    
    def display_odds_summary(self):
        """Print a nice summary of today's odds"""
        df = self.get_odds_for_today()
        
        if df.empty:
            print("No odds available")
            return
        
        print("\n" + "="*100)
        print("📊 TODAY'S VEGAS ODDS")
        print("="*100)
        print(f"{'TIME':<8} | {'MATCHUP':<35} | {'ML':<15} | {'SPREAD':<15} | {'TOTAL'}")
        print("-"*100)
        
        for _, game in df.iterrows():
            time_str = game['game_time'].strftime('%I:%M %p')
            matchup = f"{game['away_team']} @ {game['home_team']}"
            
            ml = f"{int(game['ml_away']):+d} / {int(game['ml_home']):+d}" if pd.notna(game.get('ml_away')) and pd.notna(game.get('ml_home')) else "N/A"
            spread = f"{game['spread_home']:+.1f} ({int(game['spread_home_odds']):+d})" if pd.notna(game.get('spread_home')) and game['spread_home'] else "N/A"
            total = f"{game['total_line']:.1f}" if game['total_line'] else "N/A"
            
            print(f"{time_str:<8} | {matchup:<35} | {ml:<15} | {spread:<15} | {total}")
        
        print("-"*100)
        print(f"Source: {df.iloc[0]['sportsbook']}")
        print("="*100 + "\n")


def test_api():
    """Test the odds fetcher"""
    print("="*70)
    print("TESTING THE ODDS API INTEGRATION")
    print("="*70 + "\n")
    
    fetcher = VegasOddsFetcher()
    
    # Test 1: Fetch odds
    print("Test 1: Fetching NBA odds...")
    odds = fetcher.get_nba_odds()
    
    if odds:
        print(f"✅ Successfully fetched {len(odds.get('games', []))} games")
        print(f"   API requests remaining: {odds.get('requests_remaining', 'unknown')}")
    else:
        print("❌ Failed to fetch odds")
        return
    
    # Test 2: Display formatted odds
    print("\nTest 2: Displaying formatted odds...")
    fetcher.display_odds_summary()
    
    # Test 3: Get specific game total
    print("\nTest 3: Getting specific game total...")
    df = fetcher.get_odds_for_today()
    
    if not df.empty:
        first_game = df.iloc[0]
        home = first_game['home_team']
        away = first_game['away_team']
        
        total = fetcher.get_total_for_game(home, away)
        print(f"   {away} @ {home}")
        print(f"   Total line: {total}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_api()
