"""
betting_strategy_unified.py - UNIFIED BETTING SYSTEM V2.0
Integrates Moneyline + Totals with Kelly Criterion
Author: Enhanced NBA Prediction System
Date: 2026-02-07

Key Features:
- Separate Kelly calculations for ML and Totals
- Uncertainty-aware totals betting (uses V4 quantile model)
- Portfolio management across bet types
- Anti-overfitting safeguards
- Reality-based edge thresholds
"""
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import json
from pathlib import Path
import numpy as np

class UnifiedBettingSystem:
    """Professional betting system for NBA - handles both ML and Totals"""
    
    def __init__(self, bankroll: float = 1000.0):
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        
        # Conservative Kelly settings
        self.kelly_fraction = 0.25  # Quarter-Kelly
        self.max_bet_pct = 0.05     # Max 5% per bet
        self.min_bet_amount = 5.0
        
        # Minimum edges (strict)
        self.min_edge_ml = 0.05      # 5% for moneylines
        self.min_edge_totals = 0.03  # 3% for totals
        
        # Model calibration (adjust based on your backtest results)
        # CRITICAL: Your ML model showed 25% accuracy on Feb 8 vs 75% test accuracy
        # This requires aggressive discounting until model is fixed
        self.ml_uncertainty_discount = 0.80  # Discount ML predictions by 20%
        
        # Portfolio limits
        self.max_total_exposure = 0.15  # Max 15% of bankroll across all bets
        self.max_bets_per_day = 3
        
        # Tracking
        self.bet_history: List[Dict] = []
        self.daily_exposure = 0.0
        self.pending_bets: List[Dict] = []
        
    def calculate_moneyline_bet(
        self,
        win_probability: float,
        odds: int,
        confidence: str,
        model_test_accuracy: float = 0.55,
        game_info: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Calculate moneyline bet using Kelly criterion
        
        Args:
            win_probability: Model's predicted win probability (0-1)
            odds: American odds (e.g., -110, +150)
            confidence: 'High', 'Medium', or 'Low'
            model_test_accuracy: Actual test accuracy from backtest
            game_info: Optional dict with game details
            
        Returns:
            Dict with bet recommendation or None
        """
        # Keep Low confidence plays available, but make them much smaller.
        if confidence == 'Low':
            kelly_confidence_multiplier = 0.35
        else:
            kelly_confidence_multiplier = 1.0
        
        # Apply model calibration discount
        adjusted_prob = win_probability * self.ml_uncertainty_discount
        adjusted_prob = max(0.52, min(0.70, adjusted_prob))  # Cap at 70% — model is 66.8% accurate
        
        # Convert odds to decimal
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            decimal_odds = 1 + (odds / 100)
            implied_prob = 100 / (odds + 100)
        
        # Calculate edge against the raw implied probability (vig included)
        # This is conservative: we need to beat the line INCLUDING the vig
        true_breakeven = implied_prob

        edge = adjusted_prob - true_breakeven
        
        # Minimum edge filter
        if edge < self.min_edge_ml:
            return None
        
        # Kelly Criterion
        b = decimal_odds - 1
        kelly_pct = (b * adjusted_prob - (1 - adjusted_prob)) / b
        
        # Apply fractional Kelly + confidence multiplier
        if confidence == 'High':
            kelly_multiplier = self.kelly_fraction
        elif confidence == 'Medium':
            kelly_multiplier = self.kelly_fraction * 0.6
        else:
            kelly_multiplier = self.kelly_fraction * 0.35

        kelly_multiplier *= kelly_confidence_multiplier
        
        bet_fraction = kelly_pct * kelly_multiplier
        bet_fraction = max(0, min(bet_fraction, self.max_bet_pct))
        
        if bet_fraction < 0.005:  # Less than 0.5%? Skip
            return None
        
        # Check portfolio limits
        if self.daily_exposure + bet_fraction > self.max_total_exposure:
            return None
        
        bet_amount = self.current_bankroll * bet_fraction
        
        if bet_amount < self.min_bet_amount:
            return None
        
        expected_value = bet_amount * edge
        
        return {
            'bet_type': 'moneyline',
            'bet_amount': round(bet_amount, 2),
            'pct_of_bankroll': round(bet_fraction * 100, 1),
            'edge': round(edge * 100, 1),
            'expected_value': round(expected_value, 2),
            'confidence': confidence,
            'win_probability': round(win_probability * 100, 1),
            'adjusted_win_probability': round(adjusted_prob * 100, 1),
            'true_breakeven': round(true_breakeven * 100, 1),
            'odds': odds,
            'game_info': game_info
        }
    
    def calculate_totals_bet(
        self,
        predicted_total: float,
        vegas_line: float,
        uncertainty: float,
        pred_lower: float,
        pred_upper: float,
        totals_mae: float = 15.5,  # From latest CV (avg MAE across folds)
        odds: int = -110,
        game_info: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Calculate totals bet using uncertainty-adjusted Kelly
        
        Args:
            predicted_total: Model's median prediction
            vegas_line: Current O/U line
            uncertainty: 90th - 10th percentile range
            pred_lower: 10th percentile prediction
            pred_upper: 90th percentile prediction
            totals_mae: Your model's MAE on totals (from backtest)
            odds: American odds (default -110)
            game_info: Optional game details
            
        Returns:
            Dict with bet recommendation or None
        """
        # CRITICAL: Skip high uncertainty bets
        if uncertainty > 40:
            return None
        
        # Calculate difference from Vegas line
        diff = abs(predicted_total - vegas_line)
        
        # SANITY CHECK: Skip if model disagrees too much with Vegas
        if diff > 12.0:
            return None  # Model prediction too far from market consensus

        # Tier-based minimum differences
        if uncertainty < 25:  # Diamond tier
            min_diff = 3.0
            confidence = 'High'
        elif uncertainty < 35:  # Gold tier
            min_diff = 5.0
            confidence = 'Medium'
        else:  # 35-40 range
            min_diff = 7.0
            confidence = 'Low'
        
        # Filter by minimum difference
        if diff < min_diff:
            return None
        
        # Determine direction
        if predicted_total > vegas_line:
            direction = 'OVER'
        else:
            direction = 'UNDER'
        
        # Estimate win probability using normal distribution
        # Use the LARGER of model's self-reported uncertainty OR empirical MAE
        # This prevents the model from overestimating its own precision
        model_std = uncertainty / (2 * 1.28)  # 80% interval ≈ 1.28 std devs each side
        empirical_std = totals_mae * 1.25  # MAE → std dev (assuming ~normal errors)
        std_dev = max(model_std, empirical_std)

        # Z-score for how far Vegas is from prediction
        z_score = diff / std_dev

        # Convert to win probability (using cumulative normal distribution)
        from scipy.stats import norm
        win_prob = norm.cdf(z_score)

        # Clamp to reasonable range — lower floor to avoid fake edges
        win_prob = max(0.50, min(0.75, win_prob))

        # Convert odds
        if odds < 0:
            decimal_odds = 1 + (100 / abs(odds))
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            decimal_odds = 1 + (odds / 100)
            implied_prob = 100 / (odds + 100)

        # True breakeven — use implied prob with vig still in (conservative)
        true_breakeven = implied_prob

        # Calculate edge
        edge = win_prob - true_breakeven
        
        # Minimum edge filter
        if edge < self.min_edge_totals:
            return None
        
        # Kelly Criterion
        b = decimal_odds - 1
        kelly_pct = (b * win_prob - (1 - win_prob)) / b
        
        # Apply fractional Kelly with uncertainty discount
        uncertainty_factor = 1 - (uncertainty / 100)  # Lower factor for high uncertainty
        kelly_multiplier = self.kelly_fraction * uncertainty_factor
        
        bet_fraction = kelly_pct * kelly_multiplier
        bet_fraction = max(0, min(bet_fraction, self.max_bet_pct))
        
        if bet_fraction < 0.005:
            return None
        
        # Check portfolio limits
        if self.daily_exposure + bet_fraction > self.max_total_exposure:
            return None
        
        bet_amount = self.current_bankroll * bet_fraction
        
        if bet_amount < self.min_bet_amount:
            return None
        
        expected_value = bet_amount * edge
        
        return {
            'bet_type': 'total',
            'direction': direction,
            'vegas_line': vegas_line,
            'predicted_total': round(predicted_total, 1),
            'difference': round(diff, 1),
            'bet_amount': round(bet_amount, 2),
            'pct_of_bankroll': round(bet_fraction * 100, 1),
            'edge': round(edge * 100, 1),
            'expected_value': round(expected_value, 2),
            'confidence': confidence,
            'win_probability': round(win_prob * 100, 1),
            'uncertainty': round(uncertainty, 1),
            'range': f"{pred_lower:.0f}-{pred_upper:.0f}",
            'odds': odds,
            'game_info': game_info
        }
    
    def evaluate_game(
        self,
        # Moneyline inputs
        ml_win_prob: float,
        ml_odds: int,
        ml_confidence: str,
        ml_test_accuracy: float,
        
        # Totals inputs
        predicted_total: float,
        vegas_total: float,
        total_uncertainty: float,
        total_lower: float,
        total_upper: float,
        totals_mae: float,
        total_odds: int,
        
        # Game info
        game_info: Dict
    ) -> Dict:
        """
        Evaluate a single game for BOTH moneyline and totals opportunities
        
        Returns:
            Dict with both recommendations + combined analysis
        """
        ml_bet = self.calculate_moneyline_bet(
            win_probability=ml_win_prob,
            odds=ml_odds,
            confidence=ml_confidence,
            model_test_accuracy=ml_test_accuracy,
            game_info=game_info
        )
        
        totals_bet = self.calculate_totals_bet(
            predicted_total=predicted_total,
            vegas_line=vegas_total,
            uncertainty=total_uncertainty,
            pred_lower=total_lower,
            pred_upper=total_upper,
            totals_mae=totals_mae,
            odds=total_odds,
            game_info=game_info
        )
        
        # Determine best opportunity
        recommendations = []
        
        if ml_bet:
            recommendations.append({
                **ml_bet,
                'priority_score': ml_bet['edge'] * ml_bet['win_probability']
            })
        
        if totals_bet:
            recommendations.append({
                **totals_bet,
                'priority_score': totals_bet['edge'] * totals_bet['win_probability']
            })
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return {
            'game_info': game_info,
            'moneyline_bet': ml_bet,
            'totals_bet': totals_bet,
            'recommendations': recommendations,
            'best_bet': recommendations[0] if recommendations else None
        }
    
    def get_daily_recommendations(
        self,
        game_evaluations: List[Dict],
        max_bets: Optional[int] = None
    ) -> List[Dict]:
        """
        Select best bets across all games for the day
        
        Args:
            game_evaluations: List of outputs from evaluate_game()
            max_bets: Maximum bets to recommend (default uses self.max_bets_per_day)
            
        Returns:
            Sorted list of best bets
        """
        max_bets = max_bets or self.max_bets_per_day
        
        # Collect all recommendations
        all_recs = []
        for game_eval in game_evaluations:
            for rec in game_eval['recommendations']:
                all_recs.append(rec)
        
        # Sort by priority score
        all_recs.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Select top bets within exposure limits
        selected = []
        total_exposure = 0.0
        
        for rec in all_recs:
            if len(selected) >= max_bets:
                break
            
            bet_pct = rec['pct_of_bankroll'] / 100
            
            if total_exposure + bet_pct <= self.max_total_exposure:
                selected.append(rec)
                total_exposure += bet_pct
            
        self.daily_exposure = total_exposure
        
        return selected
    
    def record_bet(self, bet_info: Dict):
        """Record a bet in history"""
        bet_record = {
            'timestamp': datetime.now().isoformat(),
            **bet_info,
            'bankroll_at_bet': self.current_bankroll
        }
        self.bet_history.append(bet_record)
        self.pending_bets.append(bet_record)
    
    def record_result(
        self,
        bet_id: int,
        won: bool,
        actual_result: Optional[Dict] = None
    ):
        """Record the outcome of a bet"""
        if bet_id >= len(self.pending_bets):
            return
        
        bet = self.pending_bets[bet_id]
        bet['result'] = 'win' if won else 'loss'
        bet['settled_at'] = datetime.now().isoformat()
        
        if won:
            if bet['odds'] < 0:
                profit = bet['bet_amount'] * (100 / abs(bet['odds']))
            else:
                profit = bet['bet_amount'] * (bet['odds'] / 100)
        else:
            profit = -bet['bet_amount']
        
        bet['profit'] = profit
        self.current_bankroll += profit
        
        # Move from pending to history
        self.pending_bets.pop(bet_id)
    
    def get_stats(self) -> Dict:
        """Get betting statistics"""
        settled = [b for b in self.bet_history if 'result' in b]
        
        if not settled:
            return {
                'total_bets': 0,
                'pending': len(self.pending_bets),
                'message': 'No settled bets yet'
            }
        
        wins = sum(1 for b in settled if b['result'] == 'win')
        total_wagered = sum(b['bet_amount'] for b in settled)
        total_profit = sum(b.get('profit', 0) for b in settled)
        
        ml_bets = [b for b in settled if b['bet_type'] == 'moneyline']
        totals_bets = [b for b in settled if b['bet_type'] == 'total']
        
        return {
            'total_bets': len(settled),
            'pending': len(self.pending_bets),
            'wins': wins,
            'losses': len(settled) - wins,
            'win_rate': wins / len(settled) if settled else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'bankroll_change_pct': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100),
            'ml_bets': len(ml_bets),
            'totals_bets': len(totals_bets),
            'ml_win_rate': sum(1 for b in ml_bets if b['result'] == 'win') / len(ml_bets) if ml_bets else 0,
            'totals_win_rate': sum(1 for b in totals_bets if b['result'] == 'win') / len(totals_bets) if totals_bets else 0
        }
    
    def print_summary(self):
        """Print betting system summary"""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("💰 UNIFIED BETTING SYSTEM - PERFORMANCE SUMMARY")
        print("="*80)
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Bets:        {stats.get('total_bets', 0)}")
        print(f"   Pending:           {stats.get('pending', 0)}")
        print(f"   Win Rate:          {stats.get('win_rate', 0):.1%}")
        print(f"   ROI:               {stats.get('roi', 0):.1f}%")
        
        print(f"\n💵 BANKROLL:")
        print(f"   Initial:           ${stats.get('initial_bankroll', 0):,.2f}")
        print(f"   Current:           ${stats.get('current_bankroll', 0):,.2f}")
        print(f"   Change:            {stats.get('bankroll_change_pct', 0):+.1f}%")
        
        if stats.get('ml_bets', 0) > 0 or stats.get('totals_bets', 0) > 0:
            print(f"\n🎯 BY BET TYPE:")
            print(f"   Moneyline:         {stats.get('ml_bets', 0)} bets ({stats.get('ml_win_rate', 0):.1%} win rate)")
            print(f"   Totals:            {stats.get('totals_bets', 0)} bets ({stats.get('totals_win_rate', 0):.1%} win rate)")
        
        print("="*80 + "\n")


# Backward compatibility wrapper
class KellyBetting(UnifiedBettingSystem):
    """Wrapper for backward compatibility with main.py"""
    
    def calculate_bet_size(
        self,
        win_probability: float,
        american_odds: int = -110,
        implied_probability: Optional[float] = None,
        confidence_level: str = 'Medium'
    ) -> Optional[Dict]:
        """Legacy interface - calls moneyline calculation"""
        return self.calculate_moneyline_bet(
            win_probability=win_probability,
            odds=american_odds,
            confidence=confidence_level,
            model_test_accuracy=0.55  # Default - should be overridden
        )
