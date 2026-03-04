from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
from ..utils.mlflow_loader import best_model, model_type
import numpy as np
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["predictions"])

class PredictRequest(BaseModel):
    city: str
    date: Optional[str] = None
    temp_max_fb: Optional[float] = None  
    temp_min_fb: Optional[float] = None  
    humidity_percent_hist_fb: Optional[float] = None  
    wind_speed_kmh_hist_fb: Optional[float] = None  
    
class PredictResponse(BaseModel):
    risk_level: str
    probability: float
    recommendation: str
    model_used: str
    details: Dict

# City encoding mapping (you should load this from your training data)
# This is a placeholder - you should save and load the actual encoder
CITY_ENCODING = {
    "tunis": 1, "ariana": 2, "ben arous": 3, "bizerte": 4,
    "sousse": 5, "sfax": 6, "nabeul": 7, "monastir": 8,
    "kairouan": 9, "gabes": 10, "gafsa": 11, "jendouba": 12,
    "kasserine": 12, "kebili": 13, "le kef": 14, "mahdia": 15,
    "manouba": 16, "medenine": 17, "sidi bouzid": 18, "siliana": 19,
    "tataouine": 20, "tozeur": 21, "zaghouan": 22
}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if best_model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model": model_status,
        "model_type": model_type
    }

@router.get("/governorates")
async def get_governorates():
    """List all Tunisian governorates"""
    return sorted([
        "Ariana", "Béja", "Ben Arous", "Bizerte", "Gabès", "Gafsa", "Jendouba",
        "Kairouan", "Kasserine", "Kébili", "Le Kef", "Mahdia", "Manouba",
        "Médenine", "Monastir", "Nabeul", "Sfax", "Sidi Bouzid", "Siliana",
        "Sousse", "Tataouine", "Tozeur", "Tunis", "Zaghouan"
    ])

@router.post("/predict", response_model=PredictResponse)
async def predict_danger(request: PredictRequest):
    try:
        # Check if model is loaded
        if best_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Parse date if provided
        pred_date = datetime.strptime(request.date, "%Y-%m-%d") if request.date else datetime.now()
        
        # Encode city (case-insensitive)
        city_lower = request.city.lower()
        city_encoded = CITY_ENCODING.get(city_lower, 0)
        
        # Use provided values or reasonable defaults
        # IMPORTANT: Use the exact feature names from training with _fb suffix
        temp_max_fb = request.temp_max if request.temp_max is not None else 25.0
        temp_min_fb = request.temp_min if request.temp_min is not None else 18.0
        humidity_percent_hist_fb = request.humidity if request.humidity is not None else 60.0
        wind_speed_kmh_hist_fb = request.wind_speed if request.wind_speed is not None else 15.0
        
        # Create feature array with EXACT order from training
        # Order: temp_max_fb, temp_min_fb, humidity_percent_hist_fb, wind_speed_kmh_hist_fb, city_encoded
        features = np.array([[
            temp_max_fb,                    # temp_max_fb
            temp_min_fb,                     # temp_min_fb
            humidity_percent_hist_fb,         # humidity_percent_hist_fb
            wind_speed_kmh_hist_fb,           # wind_speed_kmh_hist_fb
            city_encoded                      # city_encoded
        ]], dtype=np.float32)
        
        logger.info(f"Prediction features: {features}")
        logger.info(f"Feature names: ['temp_max_fb', 'temp_min_fb', 'humidity_percent_hist_fb', 'wind_speed_kmh_hist_fb', 'city_encoded']")
        
        # Make prediction
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(features)[0]
            # Get probability of positive class (danger)
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # If no predict_proba, use predict and convert to probability
            pred = best_model.predict(features)[0]
            probability = 1.0 if pred == 1 else 0.0
        
        logger.info(f"Prediction probability: {probability}")
        
        # Map probability to risk levels
        if probability < 0.2:
            level = "green"
            rec = "Conditions normales - aucune mesure particulière"
        elif probability < 0.4:
            level = "yellow"
            rec = "Soyez vigilant - conditions potentiellement dangereuses"
        elif probability < 0.6:
            level = "orange"
            rec = "Risque élevé - limitez les déplacements non essentiels"
        elif probability < 0.8:
            level = "orange_red"
            rec = "Risque très élevé - préparez-vous à des mesures d'urgence"
        else:
            level = "red"
            rec = "DANGER EXTRÊME - suivez les instructions des autorités"
        
        return PredictResponse(
            risk_level=level,
            probability=round(probability, 3),
            recommendation=rec,
            model_used=model_type,
            details={
                "city": request.city,
                "date": str(pred_date.date()),
                "temperature_max_fb": temp_max_fb,
                "temperature_min_fb": temp_min_fb,
                "humidity_percent_hist_fb": humidity_percent_hist_fb,
                "wind_speed_kmh_hist_fb": wind_speed_kmh_hist_fb,
                "city_encoded": city_encoded,
                "raw_probability": probability
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")