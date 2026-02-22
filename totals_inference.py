"""
Shared totals inference helpers.
Keeps prediction logic consistent across entrypoints.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _safe_features_frame(game_row: Any, feature_names) -> pd.DataFrame:
    row_dict = game_row.to_dict() if hasattr(game_row, "to_dict") else dict(game_row)
    for feature in feature_names:
        if feature not in row_dict:
            row_dict[feature] = 0
    frame = pd.DataFrame([row_dict], columns=feature_names)
    return frame.replace([np.inf, -np.inf], 0).fillna(0)


def predict_totals_for_game(game_row: Any, model_data: Optional[Dict]) -> Optional[Dict]:
    """
    Predict totals for a single game row across supported model versions.
    Returns a normalized dictionary with:
      predicted_total, lower, upper, uncertainty
    """
    if not model_data:
        return None

    try:
        # V5 quantile model (main/lower/upper) + optional calibrator
        if all(k in model_data for k in ("main", "lower", "upper", "scaler", "features")):
            features = model_data["features"]
            X = _safe_features_frame(game_row, features)
            X_scaled = model_data["scaler"].transform(X)

            pred_main = float(model_data["main"].predict(X_scaled)[0])
            pred_lower = float(model_data["lower"].predict(X_scaled)[0])
            pred_upper = float(model_data["upper"].predict(X_scaled)[0])

            calibrator = model_data.get("calibrator")
            if calibrator is not None:
                pred_main = float(calibrator.predict(np.array([[pred_main]]))[0])
                pred_lower = float(calibrator.predict(np.array([[pred_lower]]))[0])
                pred_upper = float(calibrator.predict(np.array([[pred_upper]]))[0])

            if pred_lower > pred_upper:
                pred_lower, pred_upper = pred_upper, pred_lower

            return {
                "predicted_total": pred_main,
                "lower": pred_lower,
                "upper": pred_upper,
                "uncertainty": pred_upper - pred_lower,
                "model_version": model_data.get("version", "v5"),
            }

        # V3 sniper ensemble with selector/weighted models
        if "hgb" in model_data and "scaler" in model_data:
            features = model_data.get("all_features") or model_data.get("features")
            if not features:
                return None

            X = _safe_features_frame(game_row, features)
            X_scaled = model_data["scaler"].transform(X)

            selector = model_data.get("selector")
            X_in = selector.transform(X_scaled) if selector is not None else X_scaled

            p_hgb = float(model_data["hgb"].predict(X_in)[0])
            p_et = float(model_data["et"].predict(X_in)[0])
            p_rf = float(model_data["rf"].predict(X_in)[0])

            weights = model_data.get("weights", {})
            wh = float(weights.get("w_hgb", 1 / 3))
            we = float(weights.get("w_et", 1 / 3))
            wr = float(weights.get("w_rf", 1 / 3))
            total_w = wh + we + wr or 1.0

            pred_main = (p_hgb * wh + p_et * we + p_rf * wr) / total_w
            low = min(p_hgb, p_et, p_rf)
            high = max(p_hgb, p_et, p_rf)

            return {
                "predicted_total": pred_main,
                "lower": low,
                "upper": high,
                "uncertainty": high - low,
                "model_version": model_data.get("version", "v3"),
            }

        # Legacy V2 two-model average
        if ("rf" in model_data and "gb" in model_data and
                "scaler" in model_data and "features" in model_data):
            features = model_data["features"]
            X = _safe_features_frame(game_row, features)
            X_scaled = model_data["scaler"].transform(X)

            p_rf = float(model_data["rf"].predict(X_scaled)[0])
            p_gb = float(model_data["gb"].predict(X_scaled)[0])
            pred_main = (p_rf + p_gb) / 2.0
            spread = abs(p_rf - p_gb)

            return {
                "predicted_total": pred_main,
                "lower": pred_main - (spread / 2.0),
                "upper": pred_main + (spread / 2.0),
                "uncertainty": spread,
                "model_version": model_data.get("version", "v2"),
            }

        return None
    except Exception:
        return None
