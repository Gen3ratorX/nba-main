"""
model_loader.py - Centralized model loading with backward compatibility
"""
import joblib
from pathlib import Path
from typing import Optional, Dict
from nbautils import log_info, log_warning

MODELS_DIR = Path('models')

def load_moneyline_model():
    """Legacy ML Loader"""
    try:
        files = list(MODELS_DIR.glob('nba_model_*.pkl'))
        if not files: return None
        latest = max(files, key=lambda x: x.stat().st_mtime)
        ts = latest.stem.replace('nba_model_', '')
        return {
            'model': joblib.load(latest),
            'scaler': joblib.load(MODELS_DIR / f'scaler_{ts}.pkl'),
            'features': joblib.load(MODELS_DIR / f'features_{ts}.pkl'),
            'timestamp': ts
        }
    except: return None

def load_totals_model() -> Optional[Dict]:
    """Load latest totals model across V5/V3/V2 formats."""
    try:
        # Prefer explicit latest pointer used by V5 training
        latest_path = MODELS_DIR / 'latest_totals.pkl'
        if latest_path.exists():
            model_data = joblib.load(latest_path)
            if isinstance(model_data, dict):
                model_data.setdefault('timestamp', model_data.get('training_date', 'latest'))
            log_info("✅ Loading latest totals model (latest_totals.pkl)")
            return model_data

        # Fallback: direct V5 artifacts
        v5_files = sorted(MODELS_DIR.glob('totals_v5_*.pkl'))
        if v5_files:
            latest = v5_files[-1]
            ts = latest.stem.replace('totals_v5_', '')
            model_data = joblib.load(latest)
            model_data['timestamp'] = ts
            log_info(f"✅ Loading V5 Calibrated Totals Model ({ts})")
            return model_data

        # Check for V3 files first
        v3_files = sorted(MODELS_DIR.glob('totals_v3_*.pkl'))
        
        if v3_files:
            latest = v3_files[-1]
            ts = latest.stem.replace('totals_v3_', '')
            log_info(f"✅ Loading V3 Sniper Model ({ts})")
            
            # LOAD AND INJECT TIMESTAMP
            model_data = joblib.load(latest)
            model_data['timestamp'] = ts # <--- FIX: Add timestamp manually
            return model_data
            
        # Fallback to V2 (Ensemble)
        et_files = sorted(MODELS_DIR.glob('totals_et_*.pkl'))
        if et_files:
            latest_et = et_files[-1]
            ts = latest_et.stem.replace('totals_et_', '')
            log_info(f"⚠️ Loading V2 Ensemble ({ts})")
            return {
                'version': 2,
                'rf': joblib.load(MODELS_DIR / f'totals_rf_{ts}.pkl'),
                'gb': joblib.load(MODELS_DIR / f'totals_gb_{ts}.pkl'),
                'et': joblib.load(latest_et),
                'weights': joblib.load(MODELS_DIR / f'totals_weights_{ts}.pkl'),
                'scaler': joblib.load(MODELS_DIR / f'totals_scaler_{ts}.pkl'),
                'features': joblib.load(MODELS_DIR / f'totals_features_{ts}.pkl'),
                'timestamp': ts
            }
            
        return None
    except Exception as e:
        log_warning(f"Error loading model: {e}")
        return None
