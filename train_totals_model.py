"""
train_totals_model.py - Train model specifically for predicting totals
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from data_processor import NBADataProcessor
from nbautils import log_info

def train_totals_model():
    """Train regression model to predict total points"""
    
    print("\n" + "="*80)
    print("🏀 TRAINING TOTALS PREDICTION MODEL")
    print("="*80)
    
    # Load historical data
    processor = NBADataProcessor()
    features_df = processor.process_all_data()
    
    if features_df.empty:
        print("❌ No data to train on")
        return
    
    # Create target: total points
    features_df['total_points'] = features_df['home_score'] + features_df['away_score']
    
    # Select features relevant to scoring
    totals_features = [
        'pace_home', 'pace_away',
        'off_rating_home', 'off_rating_away',
        'def_rating_home', 'def_rating_away',
        'net_rating_home', 'net_rating_away',
        'form_home', 'form_away',
        'rest_home', 'rest_away',
        'back_to_back_home', 'back_to_back_away'
    ]
    
    # Prepare data
    X = features_df[totals_features].copy()
    y = features_df['total_points'].copy()
    
    # Remove games with missing scores
    valid_idx = (y > 0)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"📊 Training on {len(X)} games")
    print(f"   Mean total: {y.mean():.1f}")
    print(f"   Std dev: {y.std():.1f}")
    
    # Time-series split
    train_size = int(len(X) * 0.8)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble
    print("\n🤖 Training ensemble model...")
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    
    # Train models
    rf.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)
    
    # Ensemble predictions (average)
    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_gb = gb.predict(X_test_scaled)
    y_pred = (y_pred_rf + y_pred_gb) / 2
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"   Mean Absolute Error: {mae:.2f} points")
    print(f"   Root Mean Squared Error: {rmse:.2f} points")
    print(f"   Baseline (mean): {y_train.mean():.1f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': totals_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 TOP 5 FEATURES:")
    for idx, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    joblib.dump(rf, models_dir / f'totals_rf_{timestamp}.pkl')
    joblib.dump(gb, models_dir / f'totals_gb_{timestamp}.pkl')
    joblib.dump(scaler, models_dir / f'totals_scaler_{timestamp}.pkl')
    joblib.dump(totals_features, models_dir / f'totals_features_{timestamp}.pkl')
    
    print(f"\n✅ Model saved with timestamp: {timestamp}")
    print("="*80 + "\n")
    
    return {
        'rf': rf,
        'gb': gb,
        'scaler': scaler,
        'features': totals_features,
        'mae': mae,
        'rmse': rmse
    }

if __name__ == "__main__":
    train_totals_model()