"""
data_processor.py - FIXED VERSION with Complete Totals Features & V4 Sniper Upgrades
PATCH APPLIED: Added back_to_back features to training data
Key improvements:
1. Added actual scoring averages (PPG, points allowed)
2. Added Volatility (StdDev) and Momentum (L3 vs L10)
3. Added Pace Interaction and Fatigue Index
4. 🔥 FIXED: back_to_back_home and back_to_back_away now in training data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from nbautils import (
    log_info, log_error, log_warning, update_elo, calculate_form, 
    calculate_rest_days, calculate_advanced_metrics,
    init_elo, save_json, is_back_to_back, get_h2h_record, calculate_home_court_strength,
    normalize_team_name
)
from datetime import datetime
from player_impact import PlayerImpactTracker

class NBADataProcessor:
    def __init__(self):
        ensure_directories()
        self.nba_teams = list(NBA_TEAMS.keys())
        self.elo_ratings = init_elo(self.nba_teams, base=1500)
        self.odds_history_file = Path('data/odds_history/odds_history.csv')
        self.test_predictions_file = Path('data/test_predictions.csv')
        self.test_performance_file = Path('data/test_performance.json')
        log_info("NBA Data Processor initialized (Fixed with V4 Sniper Features)")

    def _iter_canonical_training_files(self):
        """
        Use canonical files only to prevent duplicate/backfill noise from *_backup or dated snapshots.
        """
        return [
            HISTORICAL_DATA_DIR / 'historical_games_balldontlie.csv',
            CURRENT_DATA_DIR / 'current_season_espn.csv',
        ]

    def _normalize_team_code_quiet(self, team_name):
        """Quiet normalization: return None for non-NBA teams without warning logs."""
        if pd.isna(team_name) or team_name is None:
            return None

        code = str(team_name).strip().upper()
        if code in self.nba_teams:
            return code

        espn_mappings = {
            'WSH': 'WAS',
            'UT': 'UTA',
            'NO': 'NOP',
            'NY': 'NYK',
            'GS': 'GSW',
            'SA': 'SAS',
            'BRK': 'BKN',
        }
        mapped = espn_mappings.get(code)
        if mapped in self.nba_teams:
            return mapped

        # Full-name match
        name_lower = str(team_name).strip().lower()
        for abbr, full_name in NBA_TEAMS.items():
            if name_lower == full_name.lower():
                return abbr

        return None

    def _filter_valid_nba_games(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        """
        Keep only rows where both teams are recognized NBA teams.
        This removes all-star/exhibition/non-NBA rows (e.g., STARS, WORLD).
        """
        if df.empty:
            return df

        valid_codes = set(self.nba_teams)

        out = df.copy()
        out['home_team_raw'] = out['home_team']
        out['away_team_raw'] = out['away_team']
        out['home_team'] = out['home_team'].apply(self._normalize_team_code_quiet)
        out['away_team'] = out['away_team'].apply(self._normalize_team_code_quiet)

        valid_mask = out['home_team'].notna() & out['away_team'].notna()
        valid_mask &= out['home_team'].isin(valid_codes) & out['away_team'].isin(valid_codes)
        dropped = int((~valid_mask).sum())
        if dropped > 0:
            dropped_rows = out.loc[
                ~valid_mask, ['game_date', 'home_team_raw', 'away_team_raw']
            ].head(5)
            sample = "; ".join(
                f"{r['game_date']}:{r['away_team_raw']}@{r['home_team_raw']}"
                for _, r in dropped_rows.iterrows()
            )
            log_warning(
                f"Dropped {dropped} non-NBA rows during {context} filter. "
                f"Samples: {sample}"
            )

        return out.loc[valid_mask].drop(
            columns=['home_team_raw', 'away_team_raw'], errors='ignore'
        ).copy()

    def _last_non_null(self, s):
        s = s.dropna()
        return s.iloc[-1] if len(s) else np.nan

    def _load_historical_odds(self) -> pd.DataFrame:
        """
        Build game-level odds history from local snapshots and tracker exports.
        Returns one row per (game_date, home_team, away_team) when available.
        """
        odds_frames = []

        # 1) Preferred source: archived market snapshots from vegas_odds_fetcher
        if self.odds_history_file.exists():
            try:
                df = pd.read_csv(self.odds_history_file)
                if {'game_date', 'home_team', 'away_team'}.issubset(df.columns):
                    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date.astype(str)
                    df['home_team'] = df['home_team'].apply(self._normalize_team_code_quiet)
                    df['away_team'] = df['away_team'].apply(self._normalize_team_code_quiet)
                    df['snapshot_ts'] = pd.to_datetime(df.get('fetched_at'), errors='coerce')
                    df['source'] = 'market_snapshot'
                    df['source_priority'] = 0
                    odds_frames.append(df[[
                        'game_date', 'home_team', 'away_team',
                        'ml_home', 'ml_away', 'total_line',
                        'snapshot_ts', 'source', 'source_priority'
                    ]])
            except Exception as e:
                log_warning(f"Could not load odds history snapshots: {e}")

        # 2) Fallback source: tracked picks (single-side odds if available)
        if self.test_predictions_file.exists():
            try:
                df = pd.read_csv(self.test_predictions_file)
                required = {'game_date', 'home_team', 'away_team', 'predicted_winner', 'odds'}
                if required.issubset(df.columns):
                    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date.astype(str)
                    df['home_team'] = df['home_team'].apply(self._normalize_team_code_quiet)
                    df['away_team'] = df['away_team'].apply(self._normalize_team_code_quiet)
                    pred_norm = df['predicted_winner'].apply(self._normalize_team_code_quiet)
                    df['ml_home'] = np.where(pred_norm == df['home_team'], df['odds'], np.nan)
                    df['ml_away'] = np.where(pred_norm == df['away_team'], df['odds'], np.nan)
                    df['total_line'] = np.nan
                    df['snapshot_ts'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
                    df['source'] = 'tracked_pick'
                    df['source_priority'] = 1
                    odds_frames.append(df[[
                        'game_date', 'home_team', 'away_team',
                        'ml_home', 'ml_away', 'total_line',
                        'snapshot_ts', 'source', 'source_priority'
                    ]])
            except Exception as e:
                log_warning(f"Could not load tracked test_predictions odds: {e}")

        # 3) Fallback source: test performance JSON (same side-only structure)
        if self.test_performance_file.exists():
            try:
                import json
                payload = json.loads(self.test_performance_file.read_text())
                preds = payload.get('predictions', [])
                if preds:
                    df = pd.DataFrame(preds)
                    required = {'game_date', 'home_team', 'away_team', 'predicted_winner', 'odds'}
                    if required.issubset(df.columns):
                        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.date.astype(str)
                        df['home_team'] = df['home_team'].apply(self._normalize_team_code_quiet)
                        df['away_team'] = df['away_team'].apply(self._normalize_team_code_quiet)
                        pred_norm = df['predicted_winner'].apply(self._normalize_team_code_quiet)
                        df['ml_home'] = np.where(pred_norm == df['home_team'], df['odds'], np.nan)
                        df['ml_away'] = np.where(pred_norm == df['away_team'], df['odds'], np.nan)
                        df['total_line'] = np.nan
                        df['snapshot_ts'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
                        df['source'] = 'tracked_result'
                        df['source_priority'] = 2
                        odds_frames.append(df[[
                            'game_date', 'home_team', 'away_team',
                            'ml_home', 'ml_away', 'total_line',
                            'snapshot_ts', 'source', 'source_priority'
                        ]])
            except Exception as e:
                log_warning(f"Could not load test_performance odds: {e}")

        if not odds_frames:
            return pd.DataFrame()

        odds = pd.concat(odds_frames, ignore_index=True)
        odds = odds.dropna(subset=['game_date', 'home_team', 'away_team'])
        odds = odds.sort_values(['source_priority', 'snapshot_ts'])

        grouped = odds.groupby(['game_date', 'home_team', 'away_team'], as_index=False).agg({
            'ml_home': self._last_non_null,
            'ml_away': self._last_non_null,
            'total_line': self._last_non_null,
            'snapshot_ts': self._last_non_null,
            'source': self._last_non_null
        })
        return grouped

    def _merge_historical_odds(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge local historical odds onto feature rows for backtesting/reporting.
        """
        if features_df.empty:
            return features_df

        odds_df = self._load_historical_odds()
        out = features_df.copy()

        if odds_df.empty:
            out['ml_home'] = np.nan
            out['ml_away'] = np.nan
            out['total_line'] = np.nan
            out['odds_source'] = None
            out['odds_snapshot_ts'] = None
            out['market_odds_available'] = False
            return out

        out['game_date_key'] = pd.to_datetime(out['game_date'], errors='coerce').dt.date.astype(str)
        out['home_team_key'] = out['home_team'].apply(normalize_team_name)
        out['away_team_key'] = out['away_team'].apply(normalize_team_name)

        merged = out.merge(
            odds_df,
            left_on=['game_date_key', 'home_team_key', 'away_team_key'],
            right_on=['game_date', 'home_team', 'away_team'],
            how='left',
            suffixes=('', '_odds')
        )

        merged['odds_source'] = merged.get('source')
        merged['odds_snapshot_ts'] = merged.get('snapshot_ts')
        merged['market_odds_available'] = merged[['ml_home', 'ml_away']].notna().any(axis=1)

        drop_cols = [
            'game_date_key', 'home_team_key', 'away_team_key',
            'game_date_odds', 'home_team_odds', 'away_team_odds',
            'source', 'snapshot_ts'
        ]
        merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
        return merged

    @staticmethod
    def _american_to_implied(odds):
        """Convert American odds to implied probability."""
        if pd.isna(odds) or odds == 0:
            return np.nan
        odds = float(odds)
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        else:
            return 100.0 / (odds + 100.0)

    def _compute_vegas_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert ml_home/ml_away American odds into implied probability features.
        Missing odds are imputed with neutral values.
        """
        if df.empty:
            return df

        out = df.copy()

        # Compute raw implied probabilities (NaN where odds are missing)
        raw_home = out['ml_home'].apply(self._american_to_implied) if 'ml_home' in out.columns else pd.Series(np.nan, index=out.index)
        raw_away = out['ml_away'].apply(self._american_to_implied) if 'ml_away' in out.columns else pd.Series(np.nan, index=out.index)

        has_odds = raw_home.notna() & raw_away.notna()

        # Impute missing with neutral values
        out['vegas_implied_home'] = raw_home.fillna(0.5)
        out['vegas_implied_away'] = raw_away.fillna(0.5)
        out['vegas_implied_diff'] = out['vegas_implied_home'] - out['vegas_implied_away']
        out['vegas_total_line'] = out['total_line'].fillna(220.0) if 'total_line' in out.columns else 220.0
        out['has_vegas_odds'] = has_odds.astype(float)

        return out

    def _compute_line_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute line movement features from multiple odds snapshots per game.
        Features: ml_movement_home, ml_movement_magnitude, total_line_movement, odds_snapshot_count
        """
        movement_cols = ['ml_movement_home', 'ml_movement_magnitude', 'total_line_movement', 'odds_snapshot_count']
        if df.empty:
            for col in movement_cols:
                df[col] = 0.0
            return df

        out = df.copy()
        for col in movement_cols:
            out[col] = 0.0

        odds_path = Path("data/odds_history/odds_history.csv")
        if not odds_path.exists():
            return out

        try:
            all_odds = pd.read_csv(odds_path)
        except Exception:
            return out

        if all_odds.empty or 'fetched_at' not in all_odds.columns:
            return out

        all_odds['game_date'] = pd.to_datetime(all_odds['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        all_odds = all_odds.sort_values('fetched_at')

        def _to_implied(odds_val):
            try:
                odds_val = float(odds_val)
                if pd.isna(odds_val) or odds_val == 0:
                    return np.nan
                return abs(odds_val) / (abs(odds_val) + 100.0) if odds_val < 0 else 100.0 / (odds_val + 100.0)
            except (ValueError, TypeError):
                return np.nan

        for idx, row in out.iterrows():
            gd = row.get('game_date')
            game_date = pd.to_datetime(gd).strftime('%Y-%m-%d') if pd.notna(gd) else None
            home = str(row.get('home_team', ''))
            away = str(row.get('away_team', ''))

            if not game_date or not home or not away:
                continue

            mask = (
                (all_odds['game_date'] == game_date) &
                (all_odds['home_team'] == home) &
                (all_odds['away_team'] == away)
            )
            game_odds = all_odds[mask]

            if len(game_odds) < 1:
                continue

            out.at[idx, 'odds_snapshot_count'] = float(len(game_odds))

            if len(game_odds) < 2:
                continue

            first = game_odds.iloc[0]
            last = game_odds.iloc[-1]

            first_imp = _to_implied(first.get('ml_home'))
            last_imp = _to_implied(last.get('ml_home'))

            if not np.isnan(first_imp) and not np.isnan(last_imp):
                movement = last_imp - first_imp
                out.at[idx, 'ml_movement_home'] = float(movement)
                out.at[idx, 'ml_movement_magnitude'] = float(abs(movement))

            first_total = pd.to_numeric(first.get('total_line'), errors='coerce')
            last_total = pd.to_numeric(last.get('total_line'), errors='coerce')
            if not np.isnan(first_total) and not np.isnan(last_total):
                out.at[idx, 'total_line_movement'] = float(last_total - first_total)

        return out

    def _add_player_impact_features(self, df: pd.DataFrame, is_upcoming: bool = False) -> pd.DataFrame:
        """
        Batch-compute player availability features for all rows in df.
        Uses PlayerImpactTracker with persistent caching to avoid redundant API calls.
        """
        if df.empty:
            # Set default columns
            for col in ['minutes_missing_home', 'minutes_missing_away',
                        'star_missing_home', 'star_missing_away',
                        'roster_strength_home', 'roster_strength_away']:
                df[col] = 0.0 if 'missing' in col or 'star' in col else 1.0
            return df

        try:
            tracker = PlayerImpactTracker()
            features_list = []

            for _, row in df.iterrows():
                home = str(row.get('home_team', ''))
                away = str(row.get('away_team', ''))
                game_date = str(row.get('game_date', ''))[:10]

                feats = tracker.compute_impact_features(
                    home_team=home,
                    away_team=away,
                    game_date=game_date,
                    is_upcoming=is_upcoming,
                )
                features_list.append(feats)

            tracker.flush_cache()

            impact_df = pd.DataFrame(features_list, index=df.index)
            for col in impact_df.columns:
                df[col] = impact_df[col]

            player_coverage = (df['minutes_missing_home'] + df['minutes_missing_away']).gt(0).mean()
            log_info(f"Player impact: {len(df)} games processed "
                     f"({'upcoming' if is_upcoming else 'historical'}, "
                     f"{player_coverage:.1%} have missing players)")

        except Exception as e:
            log_warning(f"Player impact computation failed ({e}); using neutral defaults")
            df['minutes_missing_home'] = 0.0
            df['minutes_missing_away'] = 0.0
            df['star_missing_home'] = 0.0
            df['star_missing_away'] = 0.0
            df['roster_strength_home'] = 1.0
            df['roster_strength_away'] = 1.0

        return df

    def _get_rolling_stats(self, df, team, window, date_col='game_date'):
        """
        FIXED: Get comprehensive stats including Volatility and Momentum.
        Returns:
            (metrics_dict, consistency, avg_total, avg_ppg, avg_papg, home_ppg, away_ppg, volatility, momentum)
        """
        team_games = df[((df['home_team'] == team) | (df['away_team'] == team))]
        
        if len(team_games) < 1:
            # Return defaults
            return None, None, None, None, None, None, None, 10.0, 0.0
            
        # Select last N games
        recent = team_games.tail(window)
        
        # Calculate Advanced Metrics (pace, ratings, etc.)
        metrics = calculate_advanced_metrics(recent, team)
        
        # NEW: Calculate actual scoring statistics
        team_points = []
        opp_points = []
        game_totals = []
        home_points = []  # When playing at home
        away_points = []  # When playing away
        
        for _, game in recent.iterrows():
            is_home = (game['home_team'] == team)
            
            if is_home:
                pts = game['home_score']
                opp_pts = game['away_score']
                home_points.append(pts)
            else:
                pts = game['away_score']
                opp_pts = game['home_score']
                away_points.append(pts)
            
            team_points.append(pts)
            opp_points.append(opp_pts)
            game_totals.append(game['home_score'] + game['away_score'])
        
        # Calculate averages
        avg_points_scored = np.mean(team_points) if team_points else 110.0
        avg_points_allowed = np.mean(opp_points) if opp_points else 110.0
        avg_game_total = np.mean(game_totals) if game_totals else 220.0
        avg_home_ppg = np.mean(home_points) if home_points else 110.0
        avg_away_ppg = np.mean(away_points) if away_points else 110.0
        
        # 1. Volatility (Standard Deviation of Points Scored)
        if len(team_points) > 1:
            volatility = np.std(team_points)
        else:
            volatility = 10.0 

        # 2. Momentum (L3 vs L10 trend)
        # Positive = Heating up, Negative = Cooling down
        if len(team_games) >= 3:
            recent_3 = team_games.tail(3)
            pts_L3 = []
            for _, g in recent_3.iterrows():
                pts_L3.append(g['home_score'] if g['home_team'] == team else g['away_score'])
            avg_L3 = np.mean(pts_L3)
            momentum = avg_L3 - avg_points_scored
        else:
            momentum = 0.0
        
        # Consistency (of Totals)
        if len(game_totals) > 1:
            consistency = np.std(game_totals)
        else:
            consistency = 15.0
            
        return metrics, consistency, avg_game_total, avg_points_scored, avg_points_allowed, avg_home_ppg, avg_away_ppg, volatility, momentum

    def _get_ewma_stats(self, df, team, span=10):
        """
        Compute exponentially weighted moving averages for key stats.
        Recent games weighted more heavily than older ones.
        Returns dict with ewma features, or defaults if insufficient data.
        """
        defaults = {
            'ewma_ppg': 110.0,
            'ewma_papg': 110.0,
            'ewma_net': 0.0,
            'ewma_total': 220.0,
        }

        team_games = df[((df['home_team'] == team) | (df['away_team'] == team))]
        if len(team_games) < 2:
            return defaults

        recent = team_games.tail(span * 2)  # Take wider window, let EWMA weight it

        pts_scored = []
        pts_allowed = []
        for _, game in recent.iterrows():
            if game['home_team'] == team:
                pts_scored.append(float(game['home_score']))
                pts_allowed.append(float(game['away_score']))
            else:
                pts_scored.append(float(game['away_score']))
                pts_allowed.append(float(game['home_score']))

        s_scored = pd.Series(pts_scored)
        s_allowed = pd.Series(pts_allowed)
        s_total = s_scored + s_allowed

        ewma_ppg = float(s_scored.ewm(span=span, min_periods=1).mean().iloc[-1])
        ewma_papg = float(s_allowed.ewm(span=span, min_periods=1).mean().iloc[-1])
        ewma_total = float(s_total.ewm(span=span, min_periods=1).mean().iloc[-1])

        return {
            'ewma_ppg': ewma_ppg,
            'ewma_papg': ewma_papg,
            'ewma_net': ewma_ppg - ewma_papg,
            'ewma_total': ewma_total,
        }

    def _opponent_strength(self, history_df: pd.DataFrame, team: str, window: int = 10) -> float:
        """
        Estimate schedule strength as average opponent recent form over last `window` games.
        Range is approximately [0, 1].
        """
        if history_df.empty:
            return 0.5

        team_games = history_df[(history_df['home_team'] == team) | (history_df['away_team'] == team)].tail(window)
        if team_games.empty:
            return 0.5

        opp_forms = []
        for _, game in team_games.iterrows():
            opp = game['away_team'] if game['home_team'] == team else game['home_team']
            opp_forms.append(calculate_form(history_df, opp))

        if not opp_forms:
            return 0.5
        return float(np.clip(np.mean(opp_forms), 0.0, 1.0))

    def process_all_data(self) -> pd.DataFrame:
        log_info("Starting data processing...")
        
        # 1. Load Data (canonical files only)
        dfs = []
        for f in self._iter_canonical_training_files():
            try:
                if f.exists():
                    dfs.append(pd.read_csv(f))
            except Exception:
                pass
        
        if not dfs: return pd.DataFrame()
            
        games_df = pd.concat(dfs, ignore_index=True)
        games_df['game_date'] = pd.to_datetime(games_df['game_date'])
        games_df = self._filter_valid_nba_games(games_df, context='historical/current ingest')
        games_df = games_df.sort_values('game_date').drop_duplicates(
            subset=['game_date', 'home_team', 'away_team']
        ).reset_index(drop=True)
        
        features_list = []
        current_elo = self.elo_ratings.copy()
        
        # 2. Iterate and Calculate Features
        for idx, game in games_df.iterrows():
            if game['home_score'] == 0 and game['away_score'] == 0: 
                continue
                
            home, away = game['home_team'], game['away_team']
            date = game['game_date']
            
            # Slice history (strictly past games - prevents leakage)
            history = games_df[games_df.index < idx]
            
            # --- Calculate Rolling Stats (L5, L10) with NEW V4 features ---
            # Using L10 for broad stats + volatility/momentum
            h_L10, h_std10, h_total10, h_ppg10, h_papg10, h_home_ppg10, _, h_vol, h_mom = self._get_rolling_stats(history, home, 10)
            a_L10, a_std10, a_total10, a_ppg10, a_papg10, _, a_away_ppg10, a_vol, a_mom = self._get_rolling_stats(history, away, 10)
            
            # Using L5 for Pace (react faster)
            h_L5, _, h_total5, h_ppg5, h_papg5, h_home_ppg5, _, _, _ = self._get_rolling_stats(history, home, 5)
            a_L5, _, a_total5, a_ppg5, a_papg5, _, a_away_ppg5, _, _ = self._get_rolling_stats(history, away, 5)
            
            # Fallbacks for new teams or early season
            if not h_L10: 
                h_L10 = calculate_advanced_metrics(history, home)
                h_ppg10 = h_papg10 = h_total10 = 110.0
                h_home_ppg10 = 110.0
                h_vol = 10.0; h_mom = 0.0
            if not a_L10: 
                a_L10 = calculate_advanced_metrics(history, away)
                a_ppg10 = a_papg10 = a_total10 = 110.0
                a_away_ppg10 = 110.0
                a_vol = 10.0; a_mom = 0.0
            if not h_L5: 
                h_L5 = h_L10
                h_ppg5 = h_ppg10
                h_papg5 = h_papg10
                h_total5 = h_total10
                h_home_ppg5 = h_home_ppg10
            if not a_L5: 
                a_L5 = a_L10
                a_ppg5 = a_ppg10
                a_papg5 = a_papg10
                a_total5 = a_total10
                a_away_ppg5 = a_away_ppg10
            
            # EWMA stats (recency-weighted)
            h_ewma = self._get_ewma_stats(history, home, span=10)
            a_ewma = self._get_ewma_stats(history, away, span=10)

            # Basic ELO
            h_elo = current_elo.get(home, 1500)
            a_elo = current_elo.get(away, 1500)

            feat = {
                'game_date': date,
                'home_team': home, 'away_team': away,
                'home_score': game['home_score'], 'away_score': game['away_score'],

                # Target Variable
                'home_won': 1 if game['home_score'] > game['away_score'] else 0,

                # Existing Rolling Features
                'pace_home_L5': h_L5.get('pace', 99),
                'pace_away_L5': a_L5.get('pace', 99),
                'pace_home_L10': h_L10.get('pace', 99),
                'pace_away_L10': a_L10.get('pace', 99),
                
                'off_rating_home_L10': h_L10.get('offensive_rating', 110),
                'off_rating_away_L10': a_L10.get('offensive_rating', 110),
                'def_rating_home_L10': h_L10.get('defensive_rating', 110),
                'def_rating_away_L10': a_L10.get('defensive_rating', 110),
                'net_rating_home_L10': h_L10.get('net_rating', 0),
                'net_rating_away_L10': a_L10.get('net_rating', 0),
                
                'consistency_home': h_std10 if h_std10 else 15.0,
                'consistency_away': a_std10 if a_std10 else 15.0,
                
                # Scoring Features
                'home_ppg_L5': h_ppg5,
                'away_ppg_L5': a_ppg5,
                'home_ppg_L10': h_ppg10,
                'away_ppg_L10': a_ppg10,
                'home_papg_L10': h_papg10,
                'away_papg_L10': a_papg10,
                'home_total_avg_L5': h_total5,
                'away_total_avg_L5': a_total5,
                'home_home_ppg_L10': h_home_ppg10,
                'away_away_ppg_L10': a_away_ppg10,
                
                # NEW V4: Volatility & Momentum
                'home_volatility': h_vol,
                'away_volatility': a_vol,
                'matchup_volatility': h_vol + a_vol,
                
                'home_momentum': h_mom,
                'away_momentum': a_mom,
                'total_momentum': h_mom + a_mom,

                # EWMA features (recency-weighted scoring)
                'ewma_ppg_home': h_ewma['ewma_ppg'],
                'ewma_ppg_away': a_ewma['ewma_ppg'],
                'ewma_papg_home': h_ewma['ewma_papg'],
                'ewma_papg_away': a_ewma['ewma_papg'],
                'ewma_net_home': h_ewma['ewma_net'],
                'ewma_net_away': a_ewma['ewma_net'],
                'ewma_net_diff': h_ewma['ewma_net'] - a_ewma['ewma_net'],

                # Standard Features
                'elo_diff': h_elo - a_elo,
                'rest_home': calculate_rest_days(history, home, date),
                'rest_away': calculate_rest_days(history, away, date),
                'home_court_advantage': calculate_home_court_strength(history, home),
                
                # For Moneyline Training
                'form_home': calculate_form(history, home),
                'form_away': calculate_form(history, away),
                'form_diff': calculate_form(history, home) - calculate_form(history, away),
                'rest_diff': calculate_rest_days(history, home, date) - calculate_rest_days(history, away, date),
                
                # 🔥 FIXED: Added back_to_back features to training data
                'back_to_back_home': is_back_to_back(history, home, date),
                'back_to_back_away': is_back_to_back(history, away, date),
                # Head-to-head matchup features
                'h2h_win_rate': get_h2h_record(history, home, away),
                'h2h_games_played': len(history[
                    ((history['home_team'] == home) & (history['away_team'] == away)) |
                    ((history['home_team'] == away) & (history['away_team'] == home))
                ]),
            }
            
            # --- Advanced Feature Calculation ---
            # Pace Interaction (Fast * Fast = Explosion)
            pace_h = feat['pace_home_L5']
            pace_a = feat['pace_away_L5']
            feat['pace_interaction'] = (pace_h * pace_a) / 10000.0
            
            # Fatigue Index (Weighted Rest)
            rh = feat['rest_home']
            ra = feat['rest_away']
            h_fatigue = 1.0 if rh == 0 else (0.0 if rh == 1 else -0.5)
            a_fatigue = 1.0 if ra == 0 else (0.0 if ra == 1 else -0.5)
            feat['fatigue_index'] = h_fatigue + a_fatigue
            
            # Derived
            feat['combined_pace'] = (feat['pace_home_L5'] + feat['pace_away_L5']) / 2
            feat['pace_diff'] = abs(feat['pace_home_L5'] - feat['pace_away_L5'])
            feat['total_consistency'] = (feat['consistency_home'] + feat['consistency_away']) / 2
            feat['efficiency_ratio_home'] = feat['off_rating_home_L10'] / (feat['def_rating_home_L10'] + 1)
            feat['efficiency_ratio_away'] = feat['off_rating_away_L10'] / (feat['def_rating_away_L10'] + 1)
            feat['combined_ppg'] = feat['home_ppg_L5'] + feat['away_ppg_L5']
            feat['scoring_differential'] = feat['home_ppg_L10'] - feat['away_ppg_L10']
            feat['defensive_total'] = feat['home_papg_L10'] + feat['away_papg_L10']
            feat['net_rating_diff'] = feat['net_rating_home_L10'] - feat['net_rating_away_L10']
            feat['def_rating_diff'] = feat['def_rating_away_L10'] - feat['def_rating_home_L10']
            feat['opp_strength_home_L10'] = self._opponent_strength(history, home, window=10)
            feat['opp_strength_away_L10'] = self._opponent_strength(history, away, window=10)
            feat['schedule_strength_diff'] = feat['opp_strength_home_L10'] - feat['opp_strength_away_L10']
            
            features_list.append(feat)
            update_elo(current_elo, home, away, game['home_score'], game['away_score'])
            
        result = pd.DataFrame(features_list)
        result = self._merge_historical_odds(result)
        result = self._compute_vegas_features(result)

        # Compute player availability features for historical games
        result = self._add_player_impact_features(result, is_upcoming=False)

        # Compute line movement features from odds snapshot history
        result = self._compute_line_movement_features(result)

        odds_coverage = result['market_odds_available'].mean() if not result.empty else 0
        log_info(
            f"Processed {len(features_list)} games with Complete Scoring Features & V4 Upgrades "
            f"(odds coverage: {odds_coverage:.1%})"
        )
        return result

    def get_upcoming_games_features(self) -> pd.DataFrame:
        """Upcoming games logic with complete features"""
        upcoming_file = UPCOMING_DATA_DIR / "upcoming_games.csv"
        if not upcoming_file.exists(): 
            return pd.DataFrame()
        upcoming_df = pd.read_csv(upcoming_file)
        upcoming_df = self._filter_valid_nba_games(upcoming_df, context='upcoming ingest')
        
        # Load history (canonical files only)
        dfs = []
        for f in self._iter_canonical_training_files():
            try:
                if f.exists():
                    dfs.append(pd.read_csv(f))
            except Exception:
                pass
        history_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not history_df.empty:
            history_df['game_date'] = pd.to_datetime(history_df['game_date'])
            history_df = self._filter_valid_nba_games(history_df, context='upcoming history')
            history_df = history_df.sort_values('game_date')

        features_list = []
        date_now = datetime.now()
        
        for _, game in upcoming_df.iterrows():
            home, away = game['home_team'], game['away_team']
            
            # Rolling Stats with NEW features
            h_L10, h_std10, h_total10, h_ppg10, h_papg10, h_home_ppg10, _, h_vol, h_mom = self._get_rolling_stats(history_df, home, 10)
            a_L10, a_std10, a_total10, a_ppg10, a_papg10, _, a_away_ppg10, a_vol, a_mom = self._get_rolling_stats(history_df, away, 10)
            
            h_L5, _, h_total5, h_ppg5, h_papg5, h_home_ppg5, _, _, _ = self._get_rolling_stats(history_df, home, 5)
            a_L5, _, a_total5, a_ppg5, a_papg5, _, a_away_ppg5, _, _ = self._get_rolling_stats(history_df, away, 5)
            
            # Fallbacks
            if not h_L10: 
                h_L10 = calculate_advanced_metrics(history_df, home)
                h_ppg10 = h_papg10 = h_total10 = h_home_ppg10 = 110.0
                h_vol = 10.0; h_mom = 0.0
            if not a_L10: 
                a_L10 = calculate_advanced_metrics(history_df, away)
                a_ppg10 = a_papg10 = a_total10 = a_away_ppg10 = 110.0
                a_vol = 10.0; a_mom = 0.0
            if not h_L5: 
                h_L5 = h_L10
                h_ppg5 = h_ppg10
                h_papg5 = h_papg10
                h_total5 = h_total10
                h_home_ppg5 = h_home_ppg10
            if not a_L5: 
                a_L5 = a_L10
                a_ppg5 = a_ppg10
                a_papg5 = a_papg10
                a_total5 = a_total10
                a_away_ppg5 = a_away_ppg10

            # EWMA stats (recency-weighted)
            h_ewma = self._get_ewma_stats(history_df, home, span=10)
            a_ewma = self._get_ewma_stats(history_df, away, span=10)

            rh = calculate_rest_days(history_df, home, date_now)
            ra = calculate_rest_days(history_df, away, date_now)

            feat = {
                'game_date': game['game_date'],
                'home_team': home, 'away_team': away,

                # Pace & Efficiency
                'pace_home_L5': h_L5.get('pace', 99),
                'pace_away_L5': a_L5.get('pace', 99),
                'pace_home_L10': h_L10.get('pace', 99),
                'pace_away_L10': a_L10.get('pace', 99),
                
                'off_rating_home_L10': h_L10.get('offensive_rating', 110),
                'off_rating_away_L10': a_L10.get('offensive_rating', 110),
                'def_rating_home_L10': h_L10.get('defensive_rating', 110),
                'def_rating_away_L10': a_L10.get('defensive_rating', 110),
                'net_rating_home_L10': h_L10.get('net_rating', 0),
                'net_rating_away_L10': a_L10.get('net_rating', 0),
                
                'consistency_home': h_std10 if h_std10 else 15.0,
                'consistency_away': a_std10 if a_std10 else 15.0,
                
                # NEW: Scoring Features
                'home_ppg_L5': h_ppg5,
                'away_ppg_L5': a_ppg5,
                'home_ppg_L10': h_ppg10,
                'away_ppg_L10': a_ppg10,
                'home_papg_L10': h_papg10,
                'away_papg_L10': a_papg10,
                'home_total_avg_L5': h_total5,
                'away_total_avg_L5': a_total5,
                'home_home_ppg_L10': h_home_ppg10,
                'away_away_ppg_L10': a_away_ppg10,
                
                # NEW V4: Volatility
                'home_volatility': h_vol,
                'away_volatility': a_vol,
                'matchup_volatility': h_vol + a_vol,
                'home_momentum': h_mom,
                'away_momentum': a_mom,
                'total_momentum': h_mom + a_mom,

                # EWMA features (recency-weighted scoring)
                'ewma_ppg_home': h_ewma['ewma_ppg'],
                'ewma_ppg_away': a_ewma['ewma_ppg'],
                'ewma_papg_home': h_ewma['ewma_papg'],
                'ewma_papg_away': a_ewma['ewma_papg'],
                'ewma_net_home': h_ewma['ewma_net'],
                'ewma_net_away': a_ewma['ewma_net'],
                'ewma_net_diff': h_ewma['ewma_net'] - a_ewma['ewma_net'],

                # Standard
                'elo_diff': self.elo_ratings.get(home, 1500) - self.elo_ratings.get(away, 1500),
                'rest_home': rh,
                'rest_away': ra,
                'home_court_advantage': calculate_home_court_strength(history_df, home),
                
                # For ML
                'form_diff': calculate_form(history_df, home) - calculate_form(history_df, away),
                'rest_diff': calculate_rest_days(history_df, home, date_now) - calculate_rest_days(history_df, away, date_now),
                
                # Basic
                'net_rating_home': h_L10.get('net_rating', 0),
                'net_rating_away': a_L10.get('net_rating', 0),
                
                # Extra
                'back_to_back_home': is_back_to_back(history_df, home, date_now),
                'back_to_back_away': is_back_to_back(history_df, away, date_now),
                'h2h_win_rate': get_h2h_record(history_df, home, away),
                'h2h_games_played': len(history_df[
                    ((history_df['home_team'] == home) & (history_df['away_team'] == away)) |
                    ((history_df['home_team'] == away) & (history_df['away_team'] == home))
                ]),
            }
            
            # Interaction
            pace_h = feat['pace_home_L5']
            pace_a = feat['pace_away_L5']
            feat['pace_interaction'] = (pace_h * pace_a) / 10000.0
            
            # Fatigue
            h_fatigue = 1.0 if rh == 0 else (0.0 if rh == 1 else -0.5)
            a_fatigue = 1.0 if ra == 0 else (0.0 if ra == 1 else -0.5)
            feat['fatigue_index'] = h_fatigue + a_fatigue
            
            # Advanced
            feat['combined_pace'] = (feat['pace_home_L5'] + feat['pace_away_L5']) / 2
            feat['pace_diff'] = abs(feat['pace_home_L5'] - feat['pace_away_L5'])
            feat['total_consistency'] = (feat['consistency_home'] + feat['consistency_away']) / 2
            feat['efficiency_ratio_home'] = feat['off_rating_home_L10'] / (feat['def_rating_home_L10'] + 1)
            feat['efficiency_ratio_away'] = feat['off_rating_away_L10'] / (feat['def_rating_away_L10'] + 1)
            feat['combined_ppg'] = feat['home_ppg_L5'] + feat['away_ppg_L5']
            feat['scoring_differential'] = feat['home_ppg_L10'] - feat['away_ppg_L10']
            feat['defensive_total'] = feat['home_papg_L10'] + feat['away_papg_L10']
            feat['net_rating_diff'] = feat['net_rating_home_L10'] - feat['net_rating_away_L10']
            feat['def_rating_diff'] = feat['def_rating_away_L10'] - feat['def_rating_home_L10']
            feat['opp_strength_home_L10'] = self._opponent_strength(history_df, home, window=10)
            feat['opp_strength_away_L10'] = self._opponent_strength(history_df, away, window=10)
            feat['schedule_strength_diff'] = feat['opp_strength_home_L10'] - feat['opp_strength_away_L10']
            
            features_list.append(feat)
            
        result = pd.DataFrame(features_list)
        result = self._merge_historical_odds(result)
        result = self._compute_vegas_features(result)

        # Compute player availability features for upcoming games
        result = self._add_player_impact_features(result, is_upcoming=True)

        # Compute line movement features
        result = self._compute_line_movement_features(result)

        return result
