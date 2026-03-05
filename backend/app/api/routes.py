# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import Optional, Dict, List
# from datetime import datetime
# from ..utils.mlflow_loader import best_model, model_type
# import numpy as np
# import logging

# logger = logging.getLogger(__name__)
# router = APIRouter(prefix="/api", tags=["predictions"])

# class PredictRequest(BaseModel):
#     city: str
#     date: Optional[str] = None
#     temp_max_fb: Optional[float] = None  
#     temp_min_fb: Optional[float] = None  
#     humidity_percent_hist_fb: Optional[float] = None  
#     wind_speed_kmh_hist_fb: Optional[float] = None  
    
# class PredictResponse(BaseModel):
#     risk_level: str
#     probability: float
#     recommendation: str
#     model_used: str
#     details: Dict

# # City encoding mapping (you should load this from your training data)
# # This is a placeholder - you should save and load the actual encoder
# CITY_ENCODING = {
#     "tunis": 1, "ariana": 2, "ben arous": 3, "bizerte": 4,
#     "sousse": 5, "sfax": 6, "nabeul": 7, "monastir": 8,
#     "kairouan": 9, "gabes": 10, "gafsa": 11, "jendouba": 12,
#     "kasserine": 12, "kebili": 13, "le kef": 14, "mahdia": 15,
#     "manouba": 16, "medenine": 17, "sidi bouzid": 18, "siliana": 19,
#     "tataouine": 20, "tozeur": 21, "zaghouan": 22
# }

# @router.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     model_status = "loaded" if best_model is not None else "not_loaded"
#     return {
#         "status": "healthy",
#         "model": model_status,
#         "model_type": model_type
#     }

# @router.get("/governorates")
# async def get_governorates():
#     """List all Tunisian governorates"""
#     return sorted([
#         "Ariana", "Béja", "Ben Arous", "Bizerte", "Gabès", "Gafsa", "Jendouba",
#         "Kairouan", "Kasserine", "Kébili", "Le Kef", "Mahdia", "Manouba",
#         "Médenine", "Monastir", "Nabeul", "Sfax", "Sidi Bouzid", "Siliana",
#         "Sousse", "Tataouine", "Tozeur", "Tunis", "Zaghouan"
#     ])

# @router.post("/predict", response_model=PredictResponse)
# async def predict_danger(request: PredictRequest):
#     try:
#         # Check if model is loaded
#         if best_model is None:
#             raise HTTPException(status_code=503, detail="Model not loaded")
        
#         # Parse date if provided
#         pred_date = datetime.strptime(request.date, "%Y-%m-%d") if request.date else datetime.now()
        
#         # Encode city (case-insensitive)
#         city_lower = request.city.lower()
#         city_encoded = CITY_ENCODING.get(city_lower, 0)
        
#         # Use provided values or reasonable defaults
#         # IMPORTANT: Use the exact feature names from training with _fb suffix
#         temp_max_fb = request.temp_max if request.temp_max is not None else 25.0
#         temp_min_fb = request.temp_min if request.temp_min is not None else 18.0
#         humidity_percent_hist_fb = request.humidity if request.humidity is not None else 60.0
#         wind_speed_kmh_hist_fb = request.wind_speed if request.wind_speed is not None else 15.0
        
#         # Create feature array with EXACT order from training
#         # Order: temp_max_fb, temp_min_fb, humidity_percent_hist_fb, wind_speed_kmh_hist_fb, city_encoded
#         features = np.array([[
#             temp_max_fb,                    # temp_max_fb
#             temp_min_fb,                     # temp_min_fb
#             humidity_percent_hist_fb,         # humidity_percent_hist_fb
#             wind_speed_kmh_hist_fb,           # wind_speed_kmh_hist_fb
#             city_encoded                      # city_encoded
#         ]], dtype=np.float32)
        
#         logger.info(f"Prediction features: {features}")
#         logger.info(f"Feature names: ['temp_max_fb', 'temp_min_fb', 'humidity_percent_hist_fb', 'wind_speed_kmh_hist_fb', 'city_encoded']")
        
#         # Make prediction
#         if hasattr(best_model, "predict_proba"):
#             proba = best_model.predict_proba(features)[0]
#             # Get probability of positive class (danger)
#             probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
#         else:
#             # If no predict_proba, use predict and convert to probability
#             pred = best_model.predict(features)[0]
#             probability = 1.0 if pred == 1 else 0.0
        
#         logger.info(f"Prediction probability: {probability}")
        
#         # Map probability to risk levels
#         if probability < 0.2:
#             level = "green"
#             rec = "Conditions normales - aucune mesure particulière"
#         elif probability < 0.4:
#             level = "yellow"
#             rec = "Soyez vigilant - conditions potentiellement dangereuses"
#         elif probability < 0.6:
#             level = "orange"
#             rec = "Risque élevé - limitez les déplacements non essentiels"
#         elif probability < 0.8:
#             level = "orange_red"
#             rec = "Risque très élevé - préparez-vous à des mesures d'urgence"
#         else:
#             level = "red"
#             rec = "DANGER EXTRÊME - suivez les instructions des autorités"
        
#         return PredictResponse(
#             risk_level=level,
#             probability=round(probability, 3),
#             recommendation=rec,
#             model_used=model_type,
#             details={
#                 "city": request.city,
#                 "date": str(pred_date.date()),
#                 "temperature_max_fb": temp_max_fb,
#                 "temperature_min_fb": temp_min_fb,
#                 "humidity_percent_hist_fb": humidity_percent_hist_fb,
#                 "wind_speed_kmh_hist_fb": wind_speed_kmh_hist_fb,
#                 "city_encoded": city_encoded,
#                 "raw_probability": probability
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")



from fastapi import APIRouter, HTTPException
import mlflow
import numpy as np
from typing import Dict, Any
import os
from pydantic import BaseModel
from datetime import datetime
import random

router = APIRouter()

@router.get("/predict")
async def predict():
    """
    Exemple d'endpoint de prédiction avec logging MLflow
    """
    try:
        # Démarrer un run MLflow
        with mlflow.start_run(run_name="api_prediction") as run:
            # Loguer des paramètres
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("api_call", "predict")
            
            # Simulation de prédiction (à remplacer par votre vrai modèle)
            prediction = np.random.random()
            
            # Loguer la métrique
            mlflow.log_metric("prediction_value", float(prediction))
            
            return {
                "prediction": float(prediction),
                "run_id": run.info.run_id,
                "status": "success"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experiments")
async def list_experiments():
    """
    Liste toutes les expériences MLflow
    """
    try:
        experiments = mlflow.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "tags": exp.tags
            }
            for exp in experiments
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/runs/{experiment_id}")
async def list_runs(experiment_id: str):
    """
    Liste tous les runs d'une expérience
    """
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        return runs.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/test")
async def test():
    return {"message": "API is working"}

@router.get("/governorates")
async def get_governorates():
    """Liste des gouvernorats de Tunisie"""
    governorates = [
        "Ariana", "Béja", "Ben Arous", "Bizerte", "Gabès", "Gafsa",
        "Jendouba", "Kairouan", "Kasserine", "Kébili", "Le Kef", "Mahdia",
        "La Manouba", "Médenine", "Monastir", "Nabeul", "Sfax", "Sidi Bouzid",
        "Siliana", "Sousse", "Tataouine", "Tozeur", "Tunis", "Zaghouan"
    ]
    return {"governorates": governorates}



# Modèles de données
class ForecastRequest(BaseModel):
    date: str
    city: str

class ForecastResponse(BaseModel):
    forecast_date: str
    city: str
    risk_level: str
    confidence: int
    probabilities: Dict[str, int]
    weather: Dict[str, float]

@router.post("/forecast-by-date", response_model=ForecastResponse)
async def forecast_by_date(request: ForecastRequest):
    """
    Prédiction personnalisée pour une date et une ville spécifiques
    """
    try:
        # Valider la date
        try:
            parsed_date = datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Format de date invalide. Utilisez YYYY-MM-DD")
        
        # Ici, vous appellerez votre vrai modèle ML
        # Pour l'instant, on utilise des données simulées
        # Mais basées sur la ville et la date pour que ce soit cohérent
        
        # Utiliser la ville et la date pour générer une prédiction semi-déterministe
        city_seed = sum(ord(c) for c in request.city)
        date_seed = parsed_date.timetuple().tm_yday
        random.seed(city_seed + date_seed)
        
        # Générer les probabilités
        probs = {
            "GREEN": random.randint(0, 100),
            "YELLOW": random.randint(0, 100),
            "ORANGE": random.randint(0, 100),
            "RED": random.randint(0, 100),
            "PURPLE": random.randint(0, 100)
        }
        
        # Normaliser pour que la somme = 100
        total = sum(probs.values())
        probs = {k: int(v * 100 / total) for k, v in probs.items()}
        
        # Ajuster pour que la somme soit exactement 100
        diff = 100 - sum(probs.values())
        if diff != 0:
            # Ajouter la différence au plus grand
            max_key = max(probs, key=probs.get)
            probs[max_key] += diff
        
        # Déterminer le niveau de risque dominant
        risk_level = max(probs, key=probs.get)
        
        # Générer les conditions météo (simulées)
        weather = {
            "temp_max": round(random.uniform(15, 35), 1),
            "temp_min": round(random.uniform(5, 20), 1),
            "temp_avg": round(random.uniform(10, 25), 1),
            "wind_speed": round(random.uniform(0, 30), 1),
            "humidity": random.randint(30, 90)
        }
        
        # Réinitialiser le seed aléatoire
        random.seed()
        
        return {
            "forecast_date": request.date,
            "city": request.city,
            "risk_level": risk_level,
            "confidence": random.randint(65, 95),
            "probabilities": probs,
            "weather": weather
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# Optionnel: endpoint pour les dates disponibles
@router.get("/available-dates/{city}")
async def get_available_dates(city: str):
    """
    Retourne les dates disponibles pour une ville
    """
    from datetime import datetime, timedelta
    
    today = datetime.now()
    dates = []
    for i in range(5):  # 5 jours à l'avance
        date = today + timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
    
    return {"city": city, "available_dates": dates}