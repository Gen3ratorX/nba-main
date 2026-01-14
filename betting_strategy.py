"""
betting_strategy.py - FIXED VERSION
Key fixes:
1. Proper edge calculation accounting for vig
2. Reality checks for edges that are too good to be true
3. Uncertainty discount for model predictions
4. Better breakeven calculation
"""
from typing import Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path

class KellyBetting:
    def __init__(self, bankroll: float = 1000.0):
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.kelly_fraction = 0.25      # Use 1/4 Kelly for safety
        self.min_edge = 0.03            # FIXED: 3% minimum edge (realistic)
        self.max_bet_pct = 0.03         # Never bet more than 3% per game
        self.min_bet_amount = 5.0       # Don't bet less than $5
        self.uncertainty_discount = 0.95  # NEW: Discount probabilities by 5%
        
        # Bet tracking
        self.bet_history: List[Dict] = []
        self.total_bets = 0
        self.total_profit = 0.0
    
    def calculate_bet_size(
        self, 
        win_probability: float,
        american_odds: int = -110,
        implied_probability: Optional[float] = None,
        confidence_level: str = 'Medium'
    ) -> Optional[Dict]:
        """
        FIXED: Calculate optimal bet size using Kelly Criterion with proper edge calculation
        
        Key fixes:
        1. Account for vig properly (true breakeven at -110 is 52.38%, not 50%)
        2. Apply uncertainty discount to model probabilities
        3. Reality check for edges >10%
        
        Args:
            win_probability: Model's predicted win probability (0-1)
            american_odds: American odds format (e.g., -110, +150)
            implied_probability: Market's implied probability (optional)
            confidence_level: Model confidence level
            
        Returns:
            Dictionary with bet recommendation or None if no value
        """
        # Validate input
        if not 0 < win_probability < 1:
            return None
        
        # ==================================================================
        # FIX #1: Apply uncertainty discount to model predictions
        # Even good models aren't perfect - discount by 5%
        # ==================================================================
        adjusted_win_prob = win_probability * self.uncertainty_discount
        
        # Convert American odds to decimal
        if american_odds > 0:
            decimal_odds = (american_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(american_odds)) + 1
        
        # ==================================================================
        # FIX #2: Calculate TRUE breakeven accounting for vig
        # -110 on both sides means you need 52.38% to break even, not 50%
        # ==================================================================
        if american_odds == -110:
            # Standard two-way market with vig
            # Both sides at -110 = 110/210 = 52.38% each = 104.76% total
            true_breakeven = 0.5238
        else:
            # Calculate from odds (simplified for other odds)
            if american_odds > 0:
                true_breakeven = 100 / (american_odds + 100)
            else:
                true_breakeven = abs(american_odds) / (abs(american_odds) + 100)
            
            # Add typical 2.5% vig
            true_breakeven += 0.025
        
        # ==================================================================
        # FIX #3: Calculate REAL edge
        # Edge = (Adjusted Win Prob) - (True Breakeven)
        # ==================================================================
        edge = adjusted_win_prob - true_breakeven
        
        # ==================================================================
        # FIX #4: Reality check - warn if edge seems too high
        # ==================================================================
        if edge > 0.15:
            # Professional bettors rarely find 15%+ edges
            # This suggests model overconfidence
            pass  # We'll warn in the bet recommendation
        
        # If no edge or edge too small, do not bet
        if edge < self.min_edge:
            return None
        
        # Kelly Formula: f = (bp - q) / b
        b = decimal_odds - 1  # Net odds
        p = adjusted_win_prob  # Use adjusted probability
        q = 1 - p
        
        # Calculate raw Kelly percentage
        raw_kelly = (b * p - q) / b
        
        # Apply safety fraction (1/4 Kelly)
        safe_kelly = raw_kelly * self.kelly_fraction
        
        # Apply maximum bet cap (3% of bankroll)
        final_pct = min(safe_kelly, self.max_bet_pct)
        
        # If percentage is negative or zero, don't bet
        if final_pct <= 0:
            return None
            
        # Calculate bet amount
        bet_amount = self.current_bankroll * final_pct
        
        # Don't recommend bets under minimum amount
        if bet_amount < self.min_bet_amount:
            return None
        
        # Calculate expected profit
        expected_profit = bet_amount * edge
        
        # Determine bet confidence based on edge size
        if edge >= 0.10:
            bet_confidence = 'High'
            warning = "⚠️ CAUTION: 10%+ edge is rare - verify model calibration"
        elif edge >= 0.05:
            bet_confidence = 'Medium'
            warning = None
        else:
            bet_confidence = 'Low'
            warning = None
        
        result = {
            'bet_amount': round(bet_amount, 2),
            'pct_of_bankroll': round(final_pct * 100, 1),
            'edge': round(edge * 100, 1),
            'expected_profit': round(expected_profit, 2),
            'confidence': bet_confidence,
            'win_probability': round(win_probability * 100, 1),  # Original
            'adjusted_win_probability': round(adjusted_win_prob * 100, 1),  # After discount
            'true_breakeven': round(true_breakeven * 100, 1),  # What you need to beat
            'american_odds': american_odds,
            'decimal_odds': round(decimal_odds, 2),
            'kelly_percentage': round(raw_kelly * 100, 1),
            'fractional_kelly': round(safe_kelly * 100, 1),
            'uncertainty_discount': round(self.uncertainty_discount * 100, 1)
        }
        
        if warning:
            result['warning'] = warning
        
        return result
    
    def record_bet(
        self, 
        game_info: Dict,
        bet_info: Dict,
        result: Optional[str] = None,
        profit: Optional[float] = None
    ):
        """Record a bet for tracking purposes"""
        bet_record = {
            'timestamp': datetime.now().isoformat(),
            'game_date': game_info.get('game_date'),
            'home_team': game_info.get('home_team'),
            'away_team': game_info.get('away_team'),
            'predicted_winner': game_info.get('predicted_winner'),
            'bet_amount': bet_info['bet_amount'],
            'edge': bet_info['edge'],
            'win_probability': bet_info['win_probability'],
            'adjusted_win_probability': bet_info.get('adjusted_win_probability'),
            'odds': bet_info['american_odds'],
            'result': result,
            'profit': profit,
            'bankroll_before': self.current_bankroll
        }
        
        self.bet_history.append(bet_record)
        self.total_bets += 1
        
        # Update bankroll if result is known
        if profit is not None:
            self.current_bankroll += profit
            self.total_profit += profit
    
    def get_betting_stats(self) -> Dict:
        """Get betting performance statistics"""
        if not self.bet_history:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'roi': 0.0,
                'current_bankroll': self.current_bankroll
            }
        
        completed_bets = [b for b in self.bet_history if b['result'] is not None]
        
        if not completed_bets:
            return {
                'total_bets': len(self.bet_history),
                'pending_bets': len(self.bet_history),
                'current_bankroll': self.current_bankroll
            }
        
        wins = sum(1 for b in completed_bets if b['result'] == 'win')
        losses = sum(1 for b in completed_bets if b['result'] == 'loss')
        total_wagered = sum(b['bet_amount'] for b in completed_bets)
        total_profit = sum(b.get('profit', 0) for b in completed_bets)
        
        return {
            'total_bets': len(completed_bets),
            'pending_bets': len(self.bet_history) - len(completed_bets),
            'wins': wins,
            'losses': losses,
            'win_rate': round(wins / len(completed_bets) * 100, 1) if completed_bets else 0.0,
            'total_wagered': round(total_wagered, 2),
            'total_profit': round(total_profit, 2),
            'roi': round((total_profit / total_wagered * 100), 1) if total_wagered > 0 else 0.0,
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': round(self.current_bankroll, 2),
            'bankroll_change': round(self.current_bankroll - self.initial_bankroll, 2),
            'bankroll_change_pct': round((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100, 1)
        }
    
    def reality_check(self):
        """
        NEW: Reality check for betting expectations
        Warns users if their expectations are unrealistic
        """
        print("\n" + "="*70)
        print("📊 BETTING REALITY CHECK")
        print("="*70)
        
        print(f"\n💰 BANKROLL SETTINGS:")
        print(f"   • Starting Bankroll: ${self.initial_bankroll:,.2f}")
        print(f"   • Current Bankroll: ${self.current_bankroll:,.2f}")
        print(f"   • Min Edge Threshold: {self.min_edge*100:.1f}%")
        print(f"   • Uncertainty Discount: {self.uncertainty_discount*100:.0f}%")
        print(f"   • Max Bet per Game: {self.max_bet_pct*100:.1f}%")
        
        print(f"\n🎯 PROFESSIONAL BENCHMARKS:")
        print(f"   • Breakeven at -110 odds: 52.38%")
        print(f"   • Good professional win rate: 54-56%")
        print(f"   • Elite professional win rate: 57-60%")
        print(f"   • Typical edge found by pros: 2-5%")
        
        print(f"\n⚠️  REALITY CHECKS:")
        if self.min_edge < 0.03:
            print(f"   ⚠️  Min edge <3% is aggressive - expect high variance")
        else:
            print(f"   ✅ Min edge of {self.min_edge*100:.1f}% is reasonable")
        
        if self.kelly_fraction > 0.25:
            print(f"   ⚠️  Kelly fraction >{0.25} increases risk of ruin")
        else:
            print(f"   ✅ Using conservative {self.kelly_fraction} Kelly")
        
        if self.max_bet_pct > 0.05:
            print(f"   ⚠️  Max bet >{5}% is very aggressive")
        else:
            print(f"   ✅ Max bet of {self.max_bet_pct*100:.1f}% is conservative")
        
        print("\n💡 REMEMBER:")
        print("   • Even 60% win rate can have 10+ game losing streaks")
        print("   • Variance is real - need 100+ bets to see true edge")
        print("   • Past performance ≠ future results")
        print("   • Never bet money you can't afford to lose")
        
        print("="*70 + "\n")
    
    def save_bet_history(self, filepath: str = 'bet_history.json'):
        """Save bet history to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'bets': self.bet_history,
                    'stats': self.get_betting_stats()
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving bet history: {e}")
            return False
    
    def load_bet_history(self, filepath: str = 'bet_history.json'):
        """Load bet history from file"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.bet_history = data.get('bets', [])
                    stats = data.get('stats', {})
                    self.current_bankroll = stats.get('current_bankroll', self.initial_bankroll)
                    return True
        except Exception as e:
            print(f"Error loading bet history: {e}")
        return False
    
    def reset_bankroll(self, new_bankroll: Optional[float] = None):
        """Reset bankroll to initial or new amount"""
        if new_bankroll is not None:
            self.initial_bankroll = new_bankroll
            self.current_bankroll = new_bankroll
        else:
            self.current_bankroll = self.initial_bankroll
        
        self.bet_history = []
        self.total_bets = 0
        self.total_profit = 0.0
    
    def get_betting_summary(self) -> str:
        """Get formatted betting summary for display"""
        stats = self.get_betting_stats()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BETTING PERFORMANCE                        ║
╚══════════════════════════════════════════════════════════════╝

📊 STATISTICS:
  • Total Bets: {stats.get('total_bets', 0)}
  • Wins: {stats.get('wins', 0)}
  • Losses: {stats.get('losses', 0)}
  • Win Rate: {stats.get('win_rate', 0.0)}%
  • Pending: {stats.get('pending_bets', 0)}

💰 FINANCIAL:
  • Initial Bankroll: ${stats.get('initial_bankroll', 0):,.2f}
  • Current Bankroll: ${stats.get('current_bankroll', 0):,.2f}
  • Total Profit: ${stats.get('total_profit', 0):,.2f}
  • ROI: {stats.get('roi', 0.0)}%
  • Bankroll Change: {stats.get('bankroll_change_pct', 0.0):+.1f}%

═══════════════════════════════════════════════════════════════
"""
        return summary


# Utility function for quick bet calculation
def calculate_bet(
    win_prob: float,
    odds: int = -110,
    bankroll: float = 1000.0,
    min_edge: float = 0.03
) -> Optional[Dict]:
    """
    Quick utility function to calculate bet without creating class instance
    """
    kelly = KellyBetting(bankroll=bankroll)
    kelly.min_edge = min_edge
    return kelly.calculate_bet_size(win_prob, odds)


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("KELLY CRITERION BETTING SYSTEM - FIXED VERSION TEST")
    print("="*70)
    
    # Initialize betting system
    kelly = KellyBetting(bankroll=1000.0)
    
    # Show reality check
    kelly.reality_check()
    
    # Test scenarios with REALISTIC probabilities
    test_scenarios = [
        {'win_prob': 0.65, 'odds': -110, 'name': 'Strong edge (65% vs 52.4% breakeven)'},
        {'win_prob': 0.60, 'odds': -110, 'name': 'Good edge (60%)'},
        {'win_prob': 0.56, 'odds': -110, 'name': 'Moderate edge (56%)'},
        {'win_prob': 0.54, 'odds': -110, 'name': 'Small edge (54%)'},
        {'win_prob': 0.52, 'odds': -110, 'name': 'Below threshold (52%)'},
    ]
    
    print(f"\n💰 Initial Bankroll: ${kelly.initial_bankroll:,.2f}")
    print(f"📊 Min Edge Threshold: {kelly.min_edge*100}%")
    print(f"🎯 Max Bet Size: {kelly.max_bet_pct*100}%")
    print(f"🔻 Uncertainty Discount: {kelly.uncertainty_discount*100}%\n")
    
    for scenario in test_scenarios:
        bet = kelly.calculate_bet_size(
            win_probability=scenario['win_prob'],
            american_odds=scenario['odds']
        )
        
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"Model Prediction: {scenario['win_prob']*100:.1f}%")
        print(f"Odds: {scenario['odds']}")
        
        if bet:
            print(f"\n✅ BET RECOMMENDED:")
            print(f"  • Bet Amount: ${bet['bet_amount']:.2f} ({bet['pct_of_bankroll']}% of bankroll)")
            print(f"  • Original Probability: {bet['win_probability']}%")
            print(f"  • Adjusted Probability: {bet['adjusted_win_probability']}% (after {bet['uncertainty_discount']}% discount)")
            print(f"  • True Breakeven: {bet['true_breakeven']}%")
            print(f"  • Edge: {bet['edge']}%")
            print(f"  • Expected Profit: ${bet['expected_profit']:.2f}")
            print(f"  • Confidence: {bet['confidence']}")
            if 'warning' in bet:
                print(f"\n  {bet['warning']}")
        else:
            print(f"\n🚫 NO BET - Edge below {kelly.min_edge*100}% threshold")
    
    print(f"\n{'='*70}\n")