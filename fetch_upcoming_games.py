"""
Fetch Upcoming NBA Games using the Free Data Collector
Wrapper for compatibility with main.py
"""
from fetch_data import FreeNBADataCollector

def fetch_upcoming_games(days_ahead: int = 7) -> bool:
    """
    Fetch upcoming games using the unified FreeNBADataCollector.
    This replaces the old RapidAPI logic.
    """
    collector = FreeNBADataCollector()
    return collector.fetch_upcoming_games(days_ahead=days_ahead)

if __name__ == "__main__":
    fetch_upcoming_games(days_ahead=7)