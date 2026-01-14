"""
update_results.py - Automatically fetch game results and update performance tracker
Run this daily to update your predictions with actual outcomes
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from performance_tracker import PerformanceTracker
from fetch_data import NBADataCollector
from nbautils import log_info, log_warning, normalize_team_name
import argparse

class ResultsUpdater:
    """Automatically fetch game results and update performance tracker"""
    
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.collector = NBADataCollector()
        self.current_data_dir = Path('data/current')
    
    def fetch_recent_results(self, days_back: int = 3) -> pd.DataFrame:
        """
        Fetch game results from last N days
        
        Args:
            days_back: How many days back to check for results
            
        Returns:
            DataFrame with game results
        """
        log_info(f"📥 Fetching results from last {days_back} days...")
        
        # Fetch fresh data
        success = self.collector.fetch_current_season_data(days_back=days_back)
        
        if not success:
            log_warning("Failed to fetch current season data")
            return pd.DataFrame()
        
        # Load the most recent results file
        result_files = list(self.current_data_dir.glob("current_season_*.csv"))
        
        if not result_files:
            log_warning("No result files found")
            return pd.DataFrame()
        
        # Get the most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        log_info(f"📄 Loading results from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        
        # Filter for completed games only (both scores > 0)
        completed = df[(df['home_score'] > 0) & (df['away_score'] > 0)].copy()
        
        log_info(f"✅ Found {len(completed)} completed games")
        
        return completed
    
    def update_tracker_with_results(self, results_df: pd.DataFrame) -> dict:
        """
        Update performance tracker with game results
        
        Returns:
            Dictionary with update statistics
        """
        if results_df.empty:
            return {'updated': 0, 'not_found': 0, 'already_recorded': 0}
        
        stats = {
            'updated': 0,
            'not_found': 0,
            'already_recorded': 0,
            'errors': 0
        }
        
        # Get pending predictions
        pending_predictions = [p for p in self.tracker.predictions if not p['result_recorded']]
        
        if not pending_predictions:
            log_info("ℹ️  No pending predictions to update")
            return stats
        
        log_info(f"\n🔄 Updating {len(pending_predictions)} pending predictions...")
        
        for pred in pending_predictions:
            try:
                # Normalize team names for matching
                pred_home = normalize_team_name(pred['home_team'])
                pred_away = normalize_team_name(pred['away_team'])
                pred_date = pred['game_date']
                
                # Try to find matching game in results
                # Match by teams and date
                match = results_df[
                    (results_df['home_team'].apply(normalize_team_name) == pred_home) &
                    (results_df['away_team'].apply(normalize_team_name) == pred_away) &
                    (results_df['game_date'] == pred_date)
                ]
                
                if match.empty:
                    # Try without date (in case date format differs)
                    match = results_df[
                        (results_df['home_team'].apply(normalize_team_name) == pred_home) &
                        (results_df['away_team'].apply(normalize_team_name) == pred_away)
                    ]
                
                if not match.empty:
                    game = match.iloc[0]
                    
                    # Determine actual winner
                    if game['home_score'] > game['away_score']:
                        actual_winner = game['home_team']
                    else:
                        actual_winner = game['away_team']
                    
                    # Update tracker
                    self.tracker.record_result(
                        prediction_id=pred['prediction_id'],
                        actual_winner=actual_winner,
                        home_score=int(game['home_score']),
                        away_score=int(game['away_score'])
                    )
                    
                    # Show result
                    result_emoji = "✅" if pred['predicted_winner'] == actual_winner else "❌"
                    print(f"{result_emoji} {pred['home_team']} vs {pred['away_team']}: "
                          f"{int(game['home_score'])}-{int(game['away_score'])} "
                          f"({'WIN' if pred['predicted_winner'] == actual_winner else 'LOSS'})")
                    
                    stats['updated'] += 1
                else:
                    stats['not_found'] += 1
                    
            except Exception as e:
                log_warning(f"Error updating {pred['prediction_id']}: {e}")
                stats['errors'] += 1
        
        return stats
    
    def run_update(self, days_back: int = 3, show_report: bool = True) -> dict:
        """
        Complete update process: fetch results and update tracker
        
        Args:
            days_back: How many days back to check
            show_report: Whether to display performance report
            
        Returns:
            Update statistics
        """
        print("\n" + "="*70)
        print("🔄 NBA RESULTS UPDATER")
        print("="*70)
        
        # Step 1: Fetch results
        results_df = self.fetch_recent_results(days_back)
        
        if results_df.empty:
            print("\n⚠️  No results to process")
            return {'updated': 0}
        
        # Step 2: Update tracker
        print()
        stats = self.update_tracker_with_results(results_df)
        
        # Step 3: Display summary
        print("\n" + "="*70)
        print("📊 UPDATE SUMMARY")
        print("="*70)
        print(f"✅ Updated: {stats['updated']}")
        print(f"🔍 Not found: {stats['not_found']}")
        print(f"❌ Errors: {stats['errors']}")
        
        if stats['not_found'] > 0:
            print(f"\nℹ️  {stats['not_found']} predictions not yet played or data not available")
        
        print("="*70)
        
        # Step 4: Show performance report if requested
        if show_report and stats['updated'] > 0:
            print("\n")
            print(self.tracker.get_performance_report())
            
            # Show alerts
            alerts = self.tracker.check_for_alerts()
            if alerts:
                print("\n" + "="*70)
                print("⚠️  PERFORMANCE ALERTS")
                print("="*70)
                for alert in alerts:
                    print(f"   {alert}")
                print("="*70 + "\n")
        
        return stats


def main():
    """Command line interface for results updater"""
    parser = argparse.ArgumentParser(
        description='Update performance tracker with game results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_results.py                    # Update with last 3 days
  python update_results.py --days 7           # Check last week
  python update_results.py --no-report        # Skip performance report
  python update_results.py --export           # Export results to CSV
        """
    )
    
    parser.add_argument('--days', type=int, default=3,
                       help='Days back to check for results (default: 3)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip performance report display')
    parser.add_argument('--export', action='store_true',
                       help='Export results to CSV after update')
    
    args = parser.parse_args()
    
    # Run updater
    updater = ResultsUpdater()
    stats = updater.run_update(
        days_back=args.days,
        show_report=not args.no_report
    )
    
    # Optional: Export to CSV
    if args.export:
        print("\n📊 Exporting results to CSV...")
        updater.tracker.export_to_csv('data/prediction_results.csv')
    
    # Exit with appropriate code
    if stats['updated'] > 0:
        print("\n✅ Update completed successfully!\n")
        return 0
    else:
        print("\n⚠️  No predictions were updated\n")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())