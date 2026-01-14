"""
model_utils.py - FIXED VERSION
Key fixes:
1. Proper train/test split that doesn't contaminate test data
2. Walk-forward validation for realistic performance estimates
3. Calibration analysis to check probability accuracy
4. Reality checks for suspicious predictions
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Scikit-Learn Imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
from sklearn.calibration import CalibratedClassifierCV

# XGBoost Import (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

from config import MODELS_DIR, FEATURES
from nbautils import log_info, log_error, log_warning, save_json, load_json, ensure_directories

class NBAModelTrainer:
    """Enhanced NBA prediction model trainer with proper time-series handling"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        ensure_directories()
    
    def create_model_ensemble(self, tune_hyperparameters: bool = False) -> CalibratedClassifierCV:
        """Create an ensemble of different models with calibration"""
        
        # Define base models with better parameters
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        
        lr = LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        )
        
        # Create ensemble
        if HAS_XGBOOST:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('xgb', xgb), ('lr', lr)],
                voting='soft',
                weights=[2, 2, 2, 1]
            )
            log_info("Using 4-model ensemble: RF + GB + XGBoost + LogReg")
        else:
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft',
                weights=[2, 2, 1]
            )
            log_warning("XGBoost not available - using 3-model ensemble")
        
        # Wrap with calibration to fix overconfident probabilities
        calibrated_ensemble = CalibratedClassifierCV(
            ensemble,
            method='isotonic',
            cv=5
        )
        
        log_info("Model ensemble created with probability calibration")
        return calibrated_ensemble
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'home_won') -> Tuple:
        """Clean and prepare features for training"""
        if df.empty:
            log_error("Empty dataframe provided for feature preparation")
            return None, None, None
        
        # Filter for only the columns we defined in config.py
        available_features = [col for col in FEATURES if col in df.columns]
        
        if not available_features:
            log_error("No valid features found in dataframe")
            return None, None, None
        
        # Drop rows with NaN in features
        df_clean = df.dropna(subset=available_features)
        
        X = df_clean[available_features].copy()
        y = df_clean[target_column].copy() if target_column in df_clean.columns else None
        
        return X, y, available_features

    def train_model_with_timeseries_cv(self, X, y, feature_names, tune_hyperparameters=False):
        """
        FIXED: Train with proper time-series split that doesn't contaminate test data
        
        Key changes:
        1. Use chronological split for final train/test
        2. Never train on test data
        3. Report BOTH CV and holdout test accuracy
        4. Scaler fitted only on training data
        """
        log_info("Training NBA prediction model with Time-Series Split...")
        
        # =================================================================
        # STEP 1: Cross-validation for model selection (80% of data)
        # =================================================================
        train_size = int(len(X) * 0.8)
        X_train_full = X.iloc[:train_size]
        y_train_full = y.iloc[:train_size]
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        # Cross-validation loop ON TRAINING DATA ONLY
        for train_idx, val_idx in tscv.split(X_train_full):
            X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
            
            # Scale (fit on train, transform on val)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = self.create_model_ensemble(tune_hyperparameters)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on validation fold
            score = model.score(X_val_scaled, y_val)
            cv_scores.append(score)
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        log_info(f"Time-Series CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        
        # =================================================================
        # STEP 2: Train final model on 80% training data
        # =================================================================
        final_scaler = StandardScaler()
        X_train_scaled = final_scaler.fit_transform(X_train_full)
        
        final_model = self.create_model_ensemble(tune_hyperparameters)
        final_model.fit(X_train_scaled, y_train_full)
        
        # =================================================================
        # STEP 3: Evaluate on truly unseen 20% holdout test set
        # =================================================================
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        
        X_test_scaled = final_scaler.transform(X_test)  # ← ONLY transform, never fit
        
        y_pred = final_model.predict(X_test_scaled)
        y_proba = final_model.predict_proba(X_test_scaled)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_proba)
        test_f1 = f1_score(y_test, y_pred)
        
        log_info(f"Holdout Test Accuracy: {test_accuracy:.4f}, ROC-AUC: {test_roc_auc:.4f}")
        
        # =================================================================
        # REALITY CHECK: Warn if test >> CV (overfitting indicator)
        # =================================================================
        if test_accuracy - cv_mean > 0.10:
            log_warning("⚠️  WARNING: Test accuracy is 10%+ higher than CV!")
            log_warning("   This suggests potential overfitting or data leakage")
            log_warning(f"   CV: {cv_mean:.1%} vs Test: {test_accuracy:.1%}")
        
        # =================================================================
        # STEP 4: Analyze calibration
        # =================================================================
        self.analyze_calibration(y_test, y_proba)
        
        metrics = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_accuracy,
            'f1_score': test_f1,
            'roc_auc': test_roc_auc,
            'log_loss': log_loss(y_test, y_proba),
            'train_size': train_size,
            'test_size': len(X_test)
        }
        
        # Log feature importance
        self._log_feature_importance(final_model, feature_names)
        
        return {
            'model': final_model,
            'scaler': final_scaler,
            'feature_names': feature_names,
            'evaluation': metrics,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_accuracy,  # Add for compatibility
            'roc_auc': test_roc_auc,  # Add for compatibility
            'f1_score': test_f1  # Add for compatibility
        }

    def analyze_calibration(self, y_true, y_proba):
        """
        NEW: Analyze if predicted probabilities match actual outcomes
        This reveals if your 88% predictions actually win 88% of the time
        """
        log_info("\n" + "="*70)
        log_info("📊 CALIBRATION ANALYSIS")
        log_info("="*70)
        
        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        
        log_info(f"{'Predicted Range':<20} {'Count':<10} {'Actual Win %':<15} {'Status'}")
        log_info("-" * 70)
        
        total_calibration_error = 0
        total_predictions = 0
        
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            mask = (y_proba >= lower) & (y_proba < upper)
            count = mask.sum()
            
            if count > 0:
                actual_win_rate = y_true[mask].mean()
                predicted_avg = y_proba[mask].mean()
                calibration_error = abs(actual_win_rate - predicted_avg)
                
                total_calibration_error += calibration_error * count
                total_predictions += count
                
                # Determine calibration quality
                if calibration_error < 0.05:
                    status = "✅ Good"
                elif calibration_error < 0.10:
                    status = "⚠️  Fair"
                else:
                    status = "❌ Poor"
                
                log_info(f"{lower:.0%}-{upper:.0%}          {count:<10} "
                        f"{actual_win_rate:>6.1%} (pred: {predicted_avg:.1%})  {status}")
            else:
                log_info(f"{lower:.0%}-{upper:.0%}          {count:<10} {'N/A':<15}")
        
        # Overall calibration metric
        mean_calibration_error = total_calibration_error / total_predictions if total_predictions > 0 else 0
        
        log_info("="*70)
        log_info(f"Mean Calibration Error: {mean_calibration_error:.1%}")
        if mean_calibration_error < 0.05:
            log_info("✅ Model is well-calibrated - probabilities are trustworthy")
        elif mean_calibration_error < 0.10:
            log_info("⚠️  Model calibration is fair - use probabilities with caution")
        else:
            log_info("❌ Model is poorly calibrated - probabilities may be unreliable")
        log_info("="*70 + "\n")
        
        return mean_calibration_error

    def walk_forward_validation(self, df: pd.DataFrame, feature_names: List[str], n_splits: int = 10):
        """
        NEW: Walk-forward validation - simulates real-world usage
        Train on past data, predict future games (never seeing future in training)
        
        This is the GOLD STANDARD for time-series model evaluation
        """
        log_info("\n" + "="*70)
        log_info("🚶 WALK-FORWARD VALIDATION (Real-world simulation)")
        log_info("="*70)
        
        X, y, _ = self.prepare_features(df, target_column='home_won')
        if X is None:
            return {}
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        all_predictions = []
        fold_accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train new model for this fold
            model = self.create_model_ensemble()
            model.fit(X_train_scaled, y_train)
            
            # Predict future
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)
            
            # Record predictions
            for i, (prob, pred, actual) in enumerate(zip(y_proba, y_pred, y_test)):
                all_predictions.append({
                    'fold': fold,
                    'predicted_prob': prob,
                    'predicted_outcome': pred,
                    'actual_outcome': actual,
                    'correct': pred == actual
                })
            
            fold_acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(fold_acc)
            
            log_info(f"Fold {fold}/{n_splits}: {len(y_test)} games, Accuracy: {fold_acc:.1%}")
        
        # Analyze results
        df_preds = pd.DataFrame(all_predictions)
        
        overall_accuracy = df_preds['correct'].mean()
        
        log_info("\n" + "="*70)
        log_info("📊 WALK-FORWARD RESULTS")
        log_info("="*70)
        log_info(f"Overall Accuracy: {overall_accuracy:.1%}")
        log_info(f"Average per fold: {np.mean(fold_accuracies):.1%} (±{np.std(fold_accuracies):.1%})")
        log_info(f"Best fold: {max(fold_accuracies):.1%}")
        log_info(f"Worst fold: {min(fold_accuracies):.1%}")
        
        # Calibration analysis
        log_info("\n📊 Calibration by probability bins:")
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            mask = (df_preds['predicted_prob'] >= bins[i]) & (df_preds['predicted_prob'] < bins[i+1])
            if mask.sum() > 5:  # Only show if we have enough samples
                actual_rate = df_preds.loc[mask, 'actual_outcome'].mean()
                predicted_avg = df_preds.loc[mask, 'predicted_prob'].mean()
                count = mask.sum()
                error = abs(actual_rate - predicted_avg)
                log_info(f"  {bins[i]:.0%}-{bins[i+1]:.0%}: {count:3d} games, "
                        f"Predicted={predicted_avg:.1%}, Actual={actual_rate:.1%}, "
                        f"Error={error:.1%}")
        
        log_info("="*70 + "\n")
        
        # Reality check
        if overall_accuracy > 0.60:
            log_info("✅ Model shows strong predictive power (>60% accuracy)")
            log_info("   This could translate to profitable betting if well-calibrated")
        elif overall_accuracy > 0.55:
            log_info("⚠️  Model shows moderate predictive power (55-60%)")
            log_info("   May be profitable with careful bet selection")
        else:
            log_info("❌ Model accuracy is near random (≤55%)")
            log_info("   Not recommended for actual betting without improvements")
        
        return {
            'overall_accuracy': overall_accuracy,
            'fold_accuracies': fold_accuracies,
            'predictions': df_preds
        }

    def backtest_model(
        self, 
        df: pd.DataFrame, 
        model, 
        scaler, 
        feature_names: List[str],
        n_splits: int = 10
    ) -> Dict:
        """Backtest model on historical data using rolling time-series validation"""
        log_info(f"Starting backtest with {n_splits} time-series splits...")
        
        X, y, _ = self.prepare_features(df, target_column='home_won')
        if X is None:
            return {}
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        backtest_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)
            
            # Train new model for this fold
            fold_model = self.create_model_ensemble()
            fold_model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = fold_model.predict(X_test_scaled)
            y_proba = fold_model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            backtest_results.append({
                'fold': fold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            })
            
            log_info(f"Fold {fold}/{n_splits} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Aggregate results
        df_results = pd.DataFrame(backtest_results)
        
        summary = {
            'mean_accuracy': float(df_results['accuracy'].mean()),
            'std_accuracy': float(df_results['accuracy'].std()),
            'mean_precision': float(df_results['precision'].mean()),
            'mean_recall': float(df_results['recall'].mean()),
            'mean_f1': float(df_results['f1_score'].mean()),
            'mean_roc_auc': float(df_results['roc_auc'].mean())
        }
        
        log_info("\n" + "="*50)
        log_info("BACKTEST SUMMARY")
        log_info("="*50)
        log_info(f"Mean Accuracy: {summary['mean_accuracy']:.4f} (+/- {summary['std_accuracy']:.4f})")
        log_info(f"Mean Precision: {summary['mean_precision']:.4f}")
        log_info(f"Mean Recall: {summary['mean_recall']:.4f}")
        log_info(f"Mean F1: {summary['mean_f1']:.4f}")
        log_info(f"Mean ROC-AUC: {summary['mean_roc_auc']:.4f}")
        log_info("="*50)
        
        return summary

    def save_model(self, model, scaler, feature_names, metrics):
        """Save the model assets"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        joblib.dump(model, self.models_dir / f'nba_model_{timestamp}.pkl')
        joblib.dump(scaler, self.models_dir / f'scaler_{timestamp}.pkl')
        joblib.dump(feature_names, self.models_dir / f'features_{timestamp}.pkl')
        save_json(metrics, self.models_dir / f'metrics_{timestamp}.json')
        log_info(f"Model saved with timestamp {timestamp}")
    
    def _log_feature_importance(self, model, feature_names: List[str]):
        """Log feature importance from the trained model"""
        try:
            # Access the base estimator from the calibrated classifier
            if hasattr(model, 'calibrated_classifiers_'):
                base_estimator = model.calibrated_classifiers_[0].estimator
                
                # For VotingClassifier, get importance from RandomForest
                if hasattr(base_estimator, 'estimators_'):
                    rf_model = base_estimator.estimators_[0][1]
                    if hasattr(rf_model, 'feature_importances_'):
                        importances = rf_model.feature_importances_
                        feature_importance = dict(zip(feature_names, importances))
                        
                        # Sort by importance
                        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                        
                        log_info("\n" + "="*50)
                        log_info("TOP 10 MOST IMPORTANT FEATURES")
                        log_info("="*50)
                        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                            log_info(f"{i:2d}. {feature:25s} {importance:.4f}")
                        log_info("="*50)
        except Exception as e:
            log_warning(f"Could not extract feature importance: {e}")


# Wrapper functions for compatibility
def create_enhanced_model(df: pd.DataFrame, target_column: str = 'home_won', **kwargs):
    """
    FIXED: Wrapper function with proper validation
    """
    trainer = NBAModelTrainer()
    X, y, features = trainer.prepare_features(df, target_column)
    
    if X is None or y is None: 
        return None
    
    # Train with fixed methodology
    results = trainer.train_model_with_timeseries_cv(X, y, features)
    
    # Optional: Run walk-forward validation for additional confidence
    if len(df) > 1000:  # Only if we have enough data
        log_info("\n🚶 Running walk-forward validation...")
        wf_results = trainer.walk_forward_validation(df, features, n_splits=10)
        results['walk_forward_accuracy'] = wf_results.get('overall_accuracy', 0)
    
    # Save model
    trainer.save_model(results['model'], results['scaler'], features, results['evaluation'])
    
    return results

def predict_upcoming_games_enhanced(model, scaler, upcoming_df, feature_names):
    """Predict upcoming games with reality checks"""
    if upcoming_df.empty: 
        return pd.DataFrame()
    
    # Ensure all features exist
    X = upcoming_df.copy()
    for col in feature_names:
        if col not in X.columns: 
            X[col] = 0
            
    X = X[feature_names]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)
    
    preds = upcoming_df.copy()
    preds['home_win_probability'] = probs[:, 1]
    preds['away_win_probability'] = probs[:, 0]
    preds['predicted_home_win'] = probs[:, 1] > 0.5
    preds['confidence'] = preds[['home_win_probability', 'away_win_probability']].max(axis=1)
    
    # Add confidence levels
    preds['confidence_level'] = preds['confidence'].apply(
        lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.6 else 'Low')
    )
    
    # NEW: Reality check warnings
    high_confidence_count = (preds['confidence'] > 0.85).sum()
    if high_confidence_count > len(preds) * 0.3:
        log_warning(f"⚠️  WARNING: {high_confidence_count}/{len(preds)} predictions >85% confidence")
        log_warning("   Model may be overconfident. Use caution with betting.")
    
    return preds