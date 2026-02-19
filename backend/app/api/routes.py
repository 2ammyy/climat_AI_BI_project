# backend/app/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime
from ..utils.mlflow_loader import best_model, model_type
import numpy as np

router = APIRouter(prefix="/api", tags=["predictions"])

class PredictRequest(BaseModel):
    city: str
    date: Optional[str] = None  # YYYY-MM-DD, default today
    temp_max: Optional[float] = None
    precip_daily: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    humidity_percent: Optional[float] = None
    # Add more features if needed

class PredictResponse(BaseModel):
    risk_level: str          # green / yellow / orange / red
    probability: float       # 0-1
    recommendation: str
    model_used: str
    details: Dict

@router.get("/health")
def health_check():
    return {"status": "healthy", "model": model_type}

@router.get("/regions")
def get_regions():
    # Hardcoded for now - later load from data
    return ["Tunis", "Ariana", "Sousse", "Sfax", "Bizerte", "Gabes", "Monastir", "Nabeul"]

@router.post("/predict", response_model=PredictResponse)
def predict_risk(request: PredictRequest):
    try:
        # Default to today if no date
        pred_date = datetime.strptime(request.date, "%Y-%m-%d") if request.date else datetime.now()

        # Prepare feature vector (must match training order)
        # Update this list to match your training features
        features_order = [
            'temp_max', 'temp_min', 'temp_hist', 'temp_scraped',
            'feels_like_c_hist', 'feels_like_c_scraped',
            'humidity_percent_hist', 'humidity_percent_scraped',
            'precip_daily', 'precipprob_hist', 'precipprob_scraped',
            'wind_speed_kmh_hist', 'wind_speed_kmh_scraped', 'wind_speed_mps',
            'city_encoded', 'temp_max_lag1', 'precip_daily_lag1'
        ]

        # Dummy values if not provided (in production, fetch real forecast)
        input_data = {
            'temp_max': request.temp_max or 25.0,
            'precip_daily': request.precip_daily or 0.0,
            'wind_speed_kmh': request.wind_speed_kmh or 15.0,
            'humidity_percent_hist': 60.0,
            # ... fill defaults for all required features
        }

        # Encode city (use same LabelEncoder as training or load from pickle)
        # For simplicity, assume city_encoded = hash or fixed mapping
        input_data['city_encoded'] = hash(request.city) % 100  # placeholder!

        # Create array in correct order
        X = np.array([[input_data.get(f, 0.0) for f in features_order]])

        # Predict
        if model_type == "xgboost" or model_type == "randomforest":
            prob = best_model.predict_proba(X)[0][1]  # probability of danger (class 1)
            pred = 1 if prob > 0.5 else 0
        elif model_type == "prophet":
            # Prophet needs special handling - here we assume precip forecast
            prob = 0.5  # placeholder - implement proper prophet forecast
            pred = 1 if prob > 0.5 else 0
        else:
            raise ValueError("Unknown model type")

        # Map to color levels
        if prob < 0.2:
            level = "green"
            rec = "Safe conditions - normal activities"
        elif prob < 0.5:
            level = "yellow"
            rec = "Be aware - minor precautions"
        elif prob < 0.8:
            level = "orange"
            rec = "High risk - avoid outdoor activities, prepare"
        else:
            level = "red"
            rec = "Extreme danger - stay home, follow authorities"

        return PredictResponse(
            risk_level=level,
            probability=round(float(prob), 3),
            recommendation=rec,
            model_used=model_type,
            details={
                "city": request.city,
                "date": str(pred_date.date()),
                "raw_prob": prob,
                "prediction": pred
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))