"""
performance_tracker.py - Real-time Performance Tracking & Validation
Tracks actual results vs predictions to validate model performance
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from time_utils import get_user_timezone, now_in_tz

class PerformanceTracker:
    """
    Track actual betting results vs model predictions
    Provides real-time feedback on model accuracy and profitability
    """
    
    def __init__(self, filepath: str = 'data/prediction_history.json'):
        self.filepath = Path(filepath)
        self.user_tz = get_user_timezone()
        self.predictions: List[Dict] = []
        self.load_history()
    
    def record_prediction(
        self,
        game_date: str,
        home_team: str,
        away_team: str,
        predicted_winner: str,
        win_probability: float,
        confidence_level: str,
        edge: float,
        bet_amount: Optional[float] = None,
        odds: int = -110,
        game_id: Optional[str] = None,
        bet_type: str = 'moneyline',
        direction: Optional[str] = None,
        predicted_total: Optional[float] = None,
        vegas_line: Optional[float] = None
    ) -> str:
        """
        Record a prediction for a future game (moneyline or totals)

        Returns:
            prediction_id for later result recording
        """
        suffix = f"_{bet_type}" if bet_type != 'moneyline' else ""
        prediction_id = f"{game_date}_{home_team}_{away_team}{suffix}".replace(' ', '_')

        # Skip duplicates
        for p in self.predictions:
            if p['prediction_id'] == prediction_id:
                return prediction_id

        prediction = {
            'prediction_id': prediction_id,
            'game_id': game_id,
            'timestamp': now_in_tz(self.user_tz).isoformat(),
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'bet_type': bet_type,
            'predicted_winner': predicted_winner,
            'win_probability': win_probability,
            'confidence_level': confidence_level,
            'edge': edge,
            'bet_amount': bet_amount,
            'odds': odds,
            'direction': direction,
            'predicted_total': predicted_total,
            'vegas_line': vegas_line,
            'actual_winner': None,
            'actual_total': None,
            'correct': None,
            'profit': None,
            'result_recorded': False
        }

        self.predictions.append(prediction)
        self.save_history()

        return prediction_id
    
    def record_result(
        self,
        prediction_id: str,
        actual_winner: str,
        home_score: Optional[int] = None,
        away_score: Optional[int] = None
    ) -> Dict:
        """
        Record actual game result and calculate performance metrics
        
        Returns:
            Updated prediction with results
        """
        # Find prediction
        pred = None
        for p in self.predictions:
            if p['prediction_id'] == prediction_id:
                pred = p
                break
        
        if not pred:
            raise ValueError(f"Prediction ID not found: {prediction_id}")
        
        # Record result
        pred['actual_winner'] = actual_winner
        pred['result_recorded'] = True
        pred['result_timestamp'] = now_in_tz(self.user_tz).isoformat()

        if home_score is not None:
            pred['home_score'] = home_score
            pred['away_score'] = away_score

        # Determine correctness based on bet type
        if pred.get('bet_type', 'moneyline') == 'totals' and home_score is not None and away_score is not None:
            actual_total = home_score + away_score
            pred['actual_total'] = actual_total
            if pred.get('direction') == 'OVER':
                pred['correct'] = actual_total > pred['vegas_line']
            else:
                pred['correct'] = actual_total < pred['vegas_line']
        else:
            pred['correct'] = (pred['predicted_winner'] == actual_winner)
        
        # Calculate profit if bet was placed
        if pred['bet_amount']:
            if pred['correct']:
                # Win: get back bet + profit
                if pred['odds'] < 0:
                    profit = pred['bet_amount'] * (100 / abs(pred['odds']))
                else:
                    profit = pred['bet_amount'] * (pred['odds'] / 100)
                pred['profit'] = profit
            else:
                # Loss: lose bet amount
                pred['profit'] = -pred['bet_amount']
        
        self.save_history()
        
        return pred
    
    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics"""
        completed = [p for p in self.predictions if p['result_recorded']]
        
        if not completed:
            return {
                'total_predictions': len(self.predictions),
                'completed': 0,
                'pending': len(self.predictions),
                'message': 'No completed predictions yet'
            }
        
        bets_placed = [p for p in completed if p['bet_amount'] is not None]
        
        correct = sum(1 for p in completed if p['correct'])
        total_wagered = sum(p['bet_amount'] for p in bets_placed if p['bet_amount'])
        total_profit = sum(p.get('profit', 0) for p in bets_placed)
        
        return {
            'total_predictions': len(self.predictions),
            'completed': len(completed),
            'pending': len(self.predictions) - len(completed),
            'correct': correct,
            'incorrect': len(completed) - correct,
            'accuracy': correct / len(completed) if completed else 0,
            'bets_placed': len(bets_placed),
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            'avg_edge': np.mean([p['edge'] for p in bets_placed]) if bets_placed else 0
        }
    
    def analyze_calibration(self) -> pd.DataFrame:
        """
        Analyze calibration: Do X% predictions actually win X% of the time?
        
        Returns:
            DataFrame with calibration analysis by probability bins
        """
        completed = [p for p in self.predictions if p['result_recorded']]
        
        if not completed:
            return pd.DataFrame()
        
        df = pd.DataFrame(completed)
        
        # Define probability bins
        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        df['prob_bin'] = pd.cut(df['win_probability'], bins=bins, include_lowest=True)
        
        # Calculate actual win rate per bin
        calibration = df.groupby('prob_bin', observed=True).agg({
            'win_probability': ['mean', 'count'],
            'correct': 'mean'
        }).round(3)
        
        calibration.columns = ['predicted_prob', 'count', 'actual_win_rate']
        calibration['calibration_error'] = abs(
            calibration['actual_win_rate'] - calibration['predicted_prob']
        )
        
        return calibration.reset_index()
    
    def get_recent_performance(self, n_games: int = 20) -> Dict:
        """Get performance over last N games"""
        completed = [p for p in self.predictions if p['result_recorded']]
        
        if not completed:
            return {'message': 'No completed predictions'}
        
        recent = sorted(completed, key=lambda x: x['result_timestamp'], reverse=True)[:n_games]
        
        correct = sum(1 for p in recent if p['correct'])
        bets = [p for p in recent if p['bet_amount'] is not None]
        
        return {
            'games': len(recent),
            'correct': correct,
            'accuracy': correct / len(recent),
            'bets_placed': len(bets),
            'profit': sum(p.get('profit', 0) for p in bets)
        }
    
    def check_for_alerts(self) -> List[str]:
        """
        Check for performance issues and return alerts
        
        Returns:
            List of warning messages
        """
        alerts = []
        stats = self.get_overall_stats()
        
        if stats['completed'] < 30:
            return alerts  # Need more data
        
        # Alert 1: Accuracy below breakeven
        if stats['accuracy'] < 0.524:
            alerts.append(
                f"🚨 CRITICAL: Win rate {stats['accuracy']:.1%} is below breakeven (52.4%)"
            )
        
        # Alert 2: Recent performance decline
        recent = self.get_recent_performance(20)
        if recent['accuracy'] < stats['accuracy'] - 0.10:
            alerts.append(
                f"⚠️  WARNING: Recent performance ({recent['accuracy']:.1%}) is 10%+ below overall"
            )
        
        # Alert 3: Negative ROI
        if stats['roi'] < 0 and stats['bets_placed'] > 20:
            alerts.append(
                f"⚠️  WARNING: Negative ROI of {stats['roi']:.1f}% over {stats['bets_placed']} bets"
            )
        
        # Alert 4: Check calibration
        calibration = self.analyze_calibration()
        if not calibration.empty:
            avg_cal_error = calibration['calibration_error'].mean()
            if avg_cal_error > 0.10:
                alerts.append(
                    f"⚠️  WARNING: Poor calibration (avg error {avg_cal_error:.1%})"
                )
        
        return alerts
    
    def get_performance_by_confidence(self) -> pd.DataFrame:
        """Analyze performance by confidence level"""
        completed = [p for p in self.predictions if p['result_recorded']]
        
        if not completed:
            return pd.DataFrame()
        
        df = pd.DataFrame(completed)
        
        stats = df.groupby('confidence_level').agg({
            'correct': ['count', 'sum', 'mean'],
            'win_probability': 'mean',
            'edge': 'mean'
        }).round(3)
        
        stats.columns = ['total', 'correct', 'accuracy', 'avg_prob', 'avg_edge']
        
        return stats.reset_index()
    
    def get_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        stats = self.get_overall_stats()
        
        if stats['completed'] == 0:
            return "No completed predictions to analyze."
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    PERFORMANCE TRACKING REPORT                    ║
╚══════════════════════════════════════════════════════════════════╝

📊 OVERALL STATISTICS
{'='*70}
Total Predictions:     {stats['total_predictions']}
Completed:             {stats['completed']}
Pending:               {stats['pending']}

Correct Predictions:   {stats['correct']}
Incorrect Predictions: {stats['incorrect']}
Accuracy:              {stats['accuracy']:.1%}

🎯 BENCHMARK COMPARISON:
   • Your Accuracy:    {stats['accuracy']:.1%}
   • Breakeven (-110): 52.4%
   • Status:           {'✅ PROFITABLE' if stats['accuracy'] > 0.524 else '❌ UNPROFITABLE'}

{'='*70}

💰 BETTING PERFORMANCE
{'='*70}
Bets Placed:          {stats['bets_placed']}
Total Wagered:        ${stats['total_wagered']:,.2f}
Total Profit:         ${stats['total_profit']:,.2f}
ROI:                  {stats['roi']:.1f}%
Average Edge:         {stats['avg_edge']:.1f}%

{'='*70}
"""
        
        # Add recent performance
        recent = self.get_recent_performance(20)
        if recent.get('games', 0) > 0:
            report += f"""
📈 RECENT PERFORMANCE (Last {recent['games']} Games)
{'='*70}
Accuracy:             {recent['accuracy']:.1%}
Profit:               ${recent['profit']:,.2f}

"""
        
        # Add calibration analysis
        calibration = self.analyze_calibration()
        if not calibration.empty:
            report += f"""
🎯 CALIBRATION ANALYSIS
{'='*70}
{'Predicted Range':<20} {'Count':<10} {'Actual Win %':<15} {'Error'}
{'-'*70}
"""
            for _, row in calibration.iterrows():
                prob_range = str(row['prob_bin'])
                report += f"{prob_range:<20} {int(row['count']):<10} {row['actual_win_rate']:>6.1%}          {row['calibration_error']:>5.1%}\n"
            
            avg_error = calibration['calibration_error'].mean()
            report += f"\nAverage Calibration Error: {avg_error:.1%}\n"
            
            if avg_error < 0.05:
                report += "✅ Excellent calibration - probabilities are trustworthy\n"
            elif avg_error < 0.10:
                report += "⚠️  Fair calibration - use probabilities with caution\n"
            else:
                report += "❌ Poor calibration - probabilities may be unreliable\n"
        
        # Add performance by confidence level
        conf_stats = self.get_performance_by_confidence()
        if not conf_stats.empty:
            report += f"""
{'='*70}

📊 PERFORMANCE BY CONFIDENCE LEVEL
{'='*70}
{'Level':<15} {'Count':<10} {'Accuracy':<12} {'Avg Prob':<12} {'Avg Edge'}
{'-'*70}
"""
            for _, row in conf_stats.iterrows():
                report += f"{row['confidence_level']:<15} {int(row['total']):<10} {row['accuracy']:>6.1%}       {row['avg_prob']:>6.1%}       {row['avg_edge']:>5.1f}%\n"
        
        # Add alerts
        alerts = self.check_for_alerts()
        if alerts:
            report += f"""
{'='*70}

⚠️  ALERTS
{'='*70}
"""
            for alert in alerts:
                report += f"{alert}\n"
        
        report += f"""
{'='*70}

💡 RECOMMENDATIONS:
"""
        
        # Intelligent recommendations
        if stats['accuracy'] > 0.60:
            report += "   ✅ Excellent accuracy! Continue current strategy.\n"
        elif stats['accuracy'] > 0.55:
            report += "   ✅ Good accuracy. Monitor calibration and ROI.\n"
        elif stats['accuracy'] > 0.524:
            report += "   ⚠️  Barely profitable. Consider being more selective.\n"
        else:
            report += "   ❌ Below breakeven. Stop betting and review model.\n"
        
        if stats['completed'] < 50:
            report += "   ℹ️  Sample size is small - need 50+ games for confidence.\n"
        
        if stats['roi'] < 0 and stats['bets_placed'] > 30:
            report += "   ⚠️  Negative ROI despite predictions. Check bet sizing.\n"
        
        report += f"""
{'='*70}
"""
        
        return report
    
    def export_to_csv(self, filepath: str = 'data/prediction_results.csv'):
        """Export all predictions to CSV for analysis"""
        if not self.predictions:
            print("No predictions to export")
            return
        
        df = pd.DataFrame(self.predictions)
        df.to_csv(filepath, index=False)
        print(f"✅ Exported {len(df)} predictions to {filepath}")
    
    def save_history(self):
        """Save predictions to JSON file"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump({
                'predictions': self.predictions,
                'last_updated': now_in_tz(self.user_tz).isoformat()
            }, f, indent=2)
    
    def load_history(self):
        """Load predictions from JSON file"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
                print(f"✅ Loaded {len(self.predictions)} predictions from history")
            except Exception as e:
                print(f"⚠️  Could not load history: {e}")
                self.predictions = []
        else:
            self.predictions = []
    
    def clear_history(self):
        """Clear all prediction history (use with caution!)"""
        self.predictions = []
        self.save_history()
        print("✅ Prediction history cleared")


def main():
    """Demo/test the performance tracker"""
    print("="*70)
    print("PERFORMANCE TRACKER - DEMO")
    print("="*70)
    
    tracker = PerformanceTracker('data/test_performance.json')
    
    # Simulate some predictions and results
    print("\n📝 Recording test predictions...")
    
    # Test case 1: High confidence correct
    pred_id_1 = tracker.record_prediction(
        game_date='2026-01-13',
        home_team='Cleveland Cavaliers',
        away_team='Utah Jazz',
        predicted_winner='Cleveland Cavaliers',
        win_probability=0.796,
        confidence_level='High',
        edge=0.272,
        bet_amount=30.0,
        odds=-110
    )
    
    # Test case 2: Medium confidence incorrect
    pred_id_2 = tracker.record_prediction(
        game_date='2026-01-13',
        home_team='Boston Celtics',
        away_team='Toronto Raptors',
        predicted_winner='Boston Celtics',
        win_probability=0.623,
        confidence_level='Medium',
        edge=0.099,
        bet_amount=25.0,
        odds=-110
    )
    
    print(f"✅ Recorded 2 predictions")
    
    # Simulate results coming in
    print("\n📊 Recording results...")
    
    tracker.record_result(pred_id_1, 'Cleveland Cavaliers', 115, 98)
    print("✅ Result 1 recorded: WIN")
    
    tracker.record_result(pred_id_2, 'Toronto Raptors', 102, 105)
    print("✅ Result 2 recorded: LOSS")
    
    # Show performance report
    print("\n" + tracker.get_performance_report())
    
    # Export
    tracker.export_to_csv('data/test_predictions.csv')
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
