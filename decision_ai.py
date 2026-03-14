"""
decision_ai.py

"Final decision" layer for betting recommendations.

Today it wraps the existing UnifiedBettingSystem selection logic (which is already
backtestable) and adds persistence hooks + a future-proof interface for a learned
meta-model.

When enough labeled history exists in the warehouse, you can train a meta-model
to re-rank candidate bets, but the default behavior stays conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DecisionConfig:
    max_bets: int = 3
    # Future: path to a trained re-ranker model.
    model_path: str = "models/decision_ai.pkl"


class DecisionAI:
    def __init__(self, betting_system, config: Optional[DecisionConfig] = None):
        self.betting = betting_system
        self.config = config or DecisionConfig(max_bets=getattr(betting_system, "max_bets_per_day", 3))

    def decide(self, game_evaluations: List[Dict[str, Any]], max_bets: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Returns the final bet list for the day.

        For now this uses the existing portfolio logic. The interface exists so we can
        later add a learned ranking layer without changing main pipeline code.
        """
        max_bets = int(max_bets or self.config.max_bets)
        return self.betting.get_daily_recommendations(game_evaluations, max_bets)

