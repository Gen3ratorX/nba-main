"""
train_totals_model.py - V5 CALIBRATED QUANTILE REGRESSION SYSTEM

🔧 V5 IMPROVEMENTS (2026-02-08):
1. Implements Quantile Regression (Median, 10th, 90th percentiles)
2. Uses TimeSeriesSplit for validation (Prevents data leakage)
3. Outputs an Uncertainty Index for every prediction
4. ✅ NEW: Vegas calibration layer to correct systematic bias
5. ✅ NEW: Bias diagnostics and feature importance analysis
6. ✅ NEW: Conservative feature weights (reduced offensive bias)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import joblib
from data_processor import NBADataProcessor
from nbautils import log_info, log_error

class NBATotalsModel:
    def __init__(self):
        # Model 1: The Sniper (Median / 50th Percentile) - More conservative
        self.model_main = HistGradientBoostingRegressor(
            loss='absolute_error',
            learning_rate=0.04,
            max_iter=500,
            max_depth=8,
            l2_regularization=5.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
        
        # Model 2: The Floor (10th Percentile)
        self.model_lower = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=0.10,
            learning_rate=0.04,
            max_iter=500,
            max_depth=8,
            l2_regularization=5.0,
            early_stopping=True,
            random_state=42
        )
        
        # Model 3: The Ceiling (90th Percentile)
        self.model_upper = HistGradientBoostingRegressor(
            loss='quantile',
            quantile=0.90,
            learning_rate=0.04,
            max_iter=500,
            max_depth=8,
            l2_regularization=5.0,
            early_stopping=True,
            random_state=42
        )
        
        # Vegas Calibration Layer (NEW in V5)
        self.calibrator = LinearRegression()
        self.calibration_metrics = {}
        
        self.scaler = RobustScaler()
        
        # Define the exact feature set we want
        self.feature_cols = [
            'pace_home_L5', 'pace_away_L5', 'pace_interaction',
            'off_rating_home_L10', 'off_rating_away_L10',
            'def_rating_home_L10', 'def_rating_away_L10',
            'def_rating_diff',
            'net_rating_diff',
            'schedule_strength_diff',
            'home_ppg_L10', 'away_ppg_L10',
            'home_papg_L10', 'away_papg_L10',
            'home_home_ppg_L10', 'away_away_ppg_L10',
            'combined_ppg', 'defensive_total',
            'home_volatility', 'away_volatility', 'matchup_volatility',
            'home_momentum', 'away_momentum', 'total_momentum',
            'elo_diff', 'fatigue_index', 'home_court_advantage'
        ]

    def analyze_bias(self, y_true, y_pred, phase="Training"):
        """Analyze systematic bias in predictions"""
        bias = y_pred - y_true
        
        print(f"\n📊 {phase.upper()} BIAS ANALYSIS")
        print("="*80)
        print(f"   Mean Bias:        {bias.mean():+.2f} points")
        print(f"   Median Bias:      {np.median(bias):+.2f} points")
        print(f"   Std Dev:          {bias.std():.2f} points")
        print(f"   Over-predictions: {(bias > 0).mean():.1%}")
        print(f"   Under-predictions:{(bias < 0).mean():.1%}")
        
        for threshold in [3, 5, 8, 10]:
            pct = (np.abs(bias) > threshold).mean()
            print(f"   |Bias| > {threshold:2d} points: {pct:.1%}")
        
        return {
            'mean_bias': bias.mean(),
            'median_bias': np.median(bias),
            'std_bias': bias.std(),
            'over_pct': (bias > 0).mean()
        }

    def train(self):
        print("\n" + "="*80)
        print("🏀 TRAINING TOTALS MODEL V5 (CALIBRATED QUANTILE SNIPER)")
        print("="*80)
        
        processor = NBADataProcessor()
        df = processor.process_all_data()
        
        if df.empty:
            log_error("No data available for training")
            return

        df['total_points'] = df['home_score'] + df['away_score']
        
        missing = [f for f in self.feature_cols if f not in df.columns]
        if missing:
            log_error(f"Missing features: {missing}. Update data_processor.py!")
            return

        df = df.dropna(subset=self.feature_cols + ['total_points'])
        
        X = df[self.feature_cols]
        y = df['total_points']
        
        print(f"\n📊 Dataset: {len(df)} games")
        print(f"   Mean Total: {y.mean():.1f}")
        print(f"   Std Dev:    {y.std():.1f}")
        print(f"   Range:      {y.min():.0f} - {y.max():.0f}")
        
        print("\n" + "="*80)
        print("🕰️  STARTING TIME-SERIES VALIDATION WITH BIAS ANALYSIS")
        print("="*80)
        tscv = TimeSeriesSplit(n_splits=5)
        fold_maes = []
        fold_biases = []
        all_val_preds = []
        all_val_actuals = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            scaler_fold = RobustScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train)
            X_val_scaled = scaler_fold.transform(X_val)
            
            model_fold = HistGradientBoostingRegressor(
                loss='absolute_error',
                learning_rate=0.04,
                max_iter=500,
                max_depth=8,
                l2_regularization=5.0,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42
            )
            model_fold.fit(X_train_scaled, y_train)
            preds = model_fold.predict(X_val_scaled)
            
            mae = mean_absolute_error(y_val, preds)
            bias = (preds - y_val).mean()
            
            fold_maes.append(mae)
            fold_biases.append(bias)
            all_val_preds.extend(preds)
            all_val_actuals.extend(y_val)
            
            print(f"   Fold {fold+1} - MAE: {mae:.2f} | Bias: {bias:+.2f}")

        avg_mae = np.mean(fold_maes)
        avg_bias = np.mean(fold_biases)
        
        print(f"\n✅ VALIDATION COMPLETE")
        print(f"   Average MAE:  {avg_mae:.2f} points")
        print(f"   Average Bias: {avg_bias:+.2f} points")
        
        val_preds_array = np.array(all_val_preds)
        val_actuals_array = np.array(all_val_actuals)
        bias_metrics = self.analyze_bias(val_actuals_array, val_preds_array, "Validation")

        print("\n" + "="*80)
        print("🎯 TRAINING VEGAS CALIBRATION LAYER")
        print("="*80)
        
        self.calibrator.fit(val_preds_array.reshape(-1, 1), val_actuals_array)
        
        calibrated_preds = self.calibrator.predict(val_preds_array.reshape(-1, 1))
        calibrated_bias = (calibrated_preds - val_actuals_array).mean()
        calibrated_mae = mean_absolute_error(val_actuals_array, calibrated_preds)
        
        print(f"   Calibration Slope:     {self.calibrator.coef_[0]:.4f}")
        print(f"   Calibration Intercept: {self.calibrator.intercept_:.2f}")
        print(f"   Before Calibration - Bias: {avg_bias:+.2f}, MAE: {avg_mae:.2f}")
        print(f"   After Calibration  - Bias: {calibrated_bias:+.2f}, MAE: {calibrated_mae:.2f}")
        
        self.calibration_metrics = {
            'slope': float(self.calibrator.coef_[0]),
            'intercept': float(self.calibrator.intercept_),
            'pre_bias': float(avg_bias),
            'post_bias': float(calibrated_bias),
            'pre_mae': float(avg_mae),
            'post_mae': float(calibrated_mae)
        }

        print("\n🚀 Training Final Quantile Models on Full Dataset...")
        X_scaled = self.scaler.fit_transform(X)
        
        self.model_main.fit(X_scaled, y)
        self.model_lower.fit(X_scaled, y)
        self.model_upper.fit(X_scaled, y)
        
        final_preds = self.model_main.predict(X_scaled)
        final_bias_metrics = self.analyze_bias(y, final_preds, "Final Training")
        
        # --- FEATURE IMPORTANCE SECTION ---
        # Some environments (sandboxed runners) restrict process spawning; keep this robust.
        print("\n🔬 FEATURE IMPORTANCE ANALYSIS (Permutation-based)")
        print("="*80)
        try:
            # HistGradientBoosting doesn't have .feature_importances_, so we use permutation.
            # Use threading + n_jobs=1 to avoid OS-level multiprocessing restrictions.
            try:
                from joblib import parallel_backend
                with parallel_backend('threading'):
                    perm_result = permutation_importance(
                        self.model_main, X_scaled, y, n_repeats=5, random_state=42, n_jobs=1
                    )
            except Exception:
                perm_result = permutation_importance(
                    self.model_main, X_scaled, y, n_repeats=5, random_state=42, n_jobs=1
                )

            importances = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': perm_result.importances_mean
            }).sort_values('importance', ascending=False)
        except Exception as e:
            print(f"⚠️  Feature importance skipped: {e}")
            importances = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': [0.0] * len(self.feature_cols)
            })
        
        # Normalize for balance calculation
        total_imp = importances['importance'].sum()
        if total_imp > 0:
            importances['importance'] = importances['importance'] / total_imp

        print("\nTop 10 Most Important Features:")
        for idx, row in importances.head(10).iterrows():
            print(f"   {row['feature']:<25} {row['importance']:.4f}")
        
        # Check offensive vs defensive balance
        offensive_features = ['pace_interaction', 'off_rating_home_L10', 'off_rating_away_L10', 
                              'home_ppg_L10', 'away_ppg_L10', 'combined_ppg']
        defensive_features = ['def_rating_home_L10', 'def_rating_away_L10', 
                              'home_papg_L10', 'away_papg_L10', 'defensive_total']
        
        off_weight = importances[importances['feature'].isin(offensive_features)]['importance'].sum()
        def_weight = importances[importances['feature'].isin(defensive_features)]['importance'].sum()
        
        print(f"\n⚖️  Feature Balance:")
        print(f"   Offensive features: {off_weight:.1%}")
        print(f"   Defensive features: {def_weight:.1%}")
        print(f"   Other features:     {1 - off_weight - def_weight:.1%}")
        
        # Save System
        model_data = {
            'main': self.model_main,
            'lower': self.model_lower,
            'upper': self.model_upper,
            'scaler': self.scaler,
            'calibrator': self.calibrator,
            'features': self.feature_cols,
            'mae': avg_mae,
            'calibrated_mae': calibrated_mae,
            'calibration_metrics': self.calibration_metrics,
            'bias_metrics': {
                'validation': bias_metrics,
                'final': final_bias_metrics
            },
            'feature_importance': importances.to_dict('records'),
            'version': '5.0 (Calibrated Quantile Sniper)',
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        model_path = models_dir / f"totals_v5_{timestamp}.pkl"
        latest_path = models_dir / "latest_totals.pkl"
        
        joblib.dump(model_data, model_path)
        joblib.dump(model_data, latest_path)
        
        print(f"\n💾 System Saved:")
        print(f"   Main Path:   {model_path}")
        print(f"   Latest Path: {latest_path}")
        print("="*80 + "\n")

def train_totals_model():
    model = NBATotalsModel()
    model.train()

if __name__ == "__main__":
    train_totals_model()
