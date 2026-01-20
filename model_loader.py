"""
model_loader.py - Centralized Model Loading
All model loading logic in one place to avoid duplication
"""
import joblib
from pathlib import Path
from typing import Optional, Dict
from nbautils import log_info, log_warning

MODELS_DIR = Path('models')

def load_moneyline_model() -> Optional[Dict]:
    """
    Load the latest trained moneyline (win/loss) model
    
    Returns:
        Dictionary with 'model', 'scaler', 'features', 'timestamp' or None
    """
    model_files = list(MODELS_DIR.glob('nba_model_*.pkl'))
    
    if not model_files:
        log_warning("No moneyline model found. Run: python main.py --days-back 30")
        return None
    
    latest = max(model_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest.stem.replace('nba_model_', '')
    
    try:
        model = joblib.load(latest)
        scaler = joblib.load(MODELS_DIR / f'scaler_{timestamp}.pkl')
        features = joblib.load(MODELS_DIR / f'features_{timestamp}.pkl')
        
        log_info(f"✅ Moneyline model loaded ({timestamp})")
        return {
            'model': model,
            'scaler': scaler,
            'features': features,
            'timestamp': timestamp
        }
    except Exception as e:
        log_warning(f"Error loading moneyline model: {e}")
        return None


def load_totals_model() -> Optional[Dict]:
    """
    Load the latest trained totals (over/under) regression model
    
    Returns:
        Dictionary with 'rf', 'gb', 'scaler', 'features', 'timestamp' or None
    """
    rf_files = list(MODELS_DIR.glob('totals_rf_*.pkl'))
    gb_files = list(MODELS_DIR.glob('totals_gb_*.pkl'))
    
    if not rf_files or not gb_files:
        log_warning("No totals model found. Run: python train_totals_model.py")
        return None
    
    latest_rf = max(rf_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest_rf.stem.replace('totals_rf_', '')
    
    try:
        rf = joblib.load(latest_rf)
        gb = joblib.load(MODELS_DIR / f'totals_gb_{timestamp}.pkl')
        scaler = joblib.load(MODELS_DIR / f'totals_scaler_{timestamp}.pkl')
        features = joblib.load(MODELS_DIR / f'totals_features_{timestamp}.pkl')
        
        log_info(f"✅ Totals model loaded ({timestamp})")
        return {
            'rf': rf,
            'gb': gb,
            'scaler': scaler,
            'features': features,
            'timestamp': timestamp
        }
    except Exception as e:
        log_warning(f"Error loading totals model: {e}")
        return None


def get_model_info() -> Dict:
    """Get information about available models"""
    ml_model = load_moneyline_model()
    totals_model = load_totals_model()
    
    return {
        'moneyline': ml_model is not None,
        'moneyline_timestamp': ml_model['timestamp'] if ml_model else None,
        'totals': totals_model is not None,
        'totals_timestamp': totals_model['timestamp'] if totals_model else None
    }
