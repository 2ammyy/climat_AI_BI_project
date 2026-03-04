# backend/app/api/weather_risk.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import os
import traceback
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="WeatherGuardTN API")

# ✅ Add CORS middleware to allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL_PATH = "mlartifacts/clean_data_model.pkl"
SCALER_PATH = "mlartifacts/scaler.pkl"
ENCODER_PATH = "mlartifacts/label_encoder.pkl"
FEATURES_PATH = "mlartifacts/feature_columns.pkl"

print("📦 Loading model and preprocessors...")
try:
    # Check if files exist
    for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model).__name__}")
    print(f"📊 Features: {feature_cols}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()
    model = None
    scaler = None
    encoder = None
    feature_cols = None

# Weather API configuration - USING YOUR API KEY DIRECTLY
WEATHER_API_KEY = "139fef2236c773191352b491bd53a624"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Tunisian cities with their coordinates
CITY_COORDINATES = {
    "Tunis": {"lat": 36.8065, "lon": 10.1815},
    "Sfax": {"lat": 34.7400, "lon": 10.7600},
    "Sousse": {"lat": 35.8256, "lon": 10.6411},
    "Bizerte": {"lat": 37.2744, "lon": 9.8739},
    "Jendouba": {"lat": 36.5011, "lon": 8.7803},
    "Gabes": {"lat": 33.8815, "lon": 10.0982},
    "Ariana": {"lat": 36.8667, "lon": 10.2000},
    "Ben Arous": {"lat": 36.7533, "lon": 10.2219},
    "Manouba": {"lat": 36.8081, "lon": 10.0972},
    "Nabeul": {"lat": 36.4561, "lon": 10.7378},
    "Zaghouan": {"lat": 36.4029, "lon": 10.1429},
    "Beja": {"lat": 36.7256, "lon": 9.1817},
    "Kef": {"lat": 36.1741, "lon": 8.7049},
    "Siliana": {"lat": 36.0849, "lon": 9.3708},
    "Kairouan": {"lat": 35.6781, "lon": 10.0963},
    "Kasserine": {"lat": 35.1676, "lon": 8.8365},
    "Sidi Bouzid": {"lat": 35.0359, "lon": 9.4858},
    "Monastir": {"lat": 35.7833, "lon": 10.8333},
    "Mahdia": {"lat": 35.5047, "lon": 11.0622},
    "Medenine": {"lat": 33.3549, "lon": 10.5055},
    "Tataouine": {"lat": 32.9297, "lon": 10.4518},
    "Gafsa": {"lat": 34.4250, "lon": 8.7842},
    "Tozeur": {"lat": 33.9197, "lon": 8.1335},
    "Kebili": {"lat": 33.7044, "lon": 8.9692}
}

# Define request model for manual weather input
class WeatherData(BaseModel):
    temp_max: float = Field(..., description="Maximum temperature in °C", example=35.0)
    temp_min: float = Field(..., description="Minimum temperature in °C", example=24.0)
    temp_avg: float = Field(..., description="Average temperature in °C", example=29.5)
    wind_speed: float = Field(..., description="Wind speed in km/h", example=45.0)
    humidity: float = Field(..., description="Humidity percentage (0-100)", example=60.0)
    city: str = Field("Unknown", description="City name", example="Tunis")

# New request model for date-only input
class DateForecastRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format", example="2024-03-15")
    city: str = Field(..., description="City name", example="Tunis")

# Define nested weather info model for response
class WeatherInfo(BaseModel):
    """Weather data included in response"""
    temp_max: float
    temp_min: float
    temp_avg: float
    wind_speed: float
    humidity: float
    city: str

# Define response model
class RiskResponse(BaseModel):
    risk_level: str
    risk_code: int
    confidence: float
    probabilities: Dict[str, float]
    city: str
    weather: WeatherInfo

# Risk level mapping
RISK_NAMES = {0: "GREEN", 1: "YELLOW", 2: "ORANGE", 3: "RED", 4: "PURPLE"}
RISK_COLORS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🟣"}
RISK_DESCRIPTIONS = {
    "GREEN": "Normal conditions - No action needed",
    "YELLOW": "Be aware - Monitor weather updates",
    "ORANGE": "Be prepared - Possible disruptions",
    "RED": "Take action - Protect life and property",
    "PURPLE": "Emergency - Immediate response needed"
}

@app.get("/")
def root():
    return {
        "message": "WeatherGuardTN API",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=RiskResponse)
async def predict_risk(weather: WeatherData):
    """Predict weather risk level based on manually entered weather data"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        print(f"📥 Received request: {weather}")
        
        # Prepare features in the exact order the model expects
        features = np.array([[
            float(weather.temp_max),
            float(weather.temp_min),
            float(weather.temp_avg),
            float(weather.wind_speed),
            float(weather.humidity)
        ]])
        
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Features: {features}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        print(f"📊 Scaled features shape: {features_scaled.shape}")
        
        # Predict
        pred = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        print(f"🎯 Prediction: {pred}")
        print(f"📊 Probabilities: {probabilities}")
        
        # Get risk level
        risk_code = int(pred)
        risk_level = RISK_NAMES[risk_code]
        confidence = float(max(probabilities) * 100)
        
        # Create probability dictionary
        prob_dict = {}
        for i in range(len(probabilities)):
            prob_value = float(probabilities[i] * 100)
            if prob_value > 0.1:  # Show all non-zero probabilities
                prob_dict[RISK_NAMES[i]] = round(prob_value, 2)
        
        # Create response with proper typing
        response = RiskResponse(
            risk_level=risk_level,
            risk_code=risk_code,
            confidence=round(confidence, 2),
            probabilities=prob_dict,
            city=weather.city,
            weather=WeatherInfo(
                temp_max=round(weather.temp_max, 1),
                temp_min=round(weather.temp_min, 1),
                temp_avg=round(weather.temp_avg, 1),
                wind_speed=round(weather.wind_speed, 1),
                humidity=round(weather.humidity, 1),
                city=weather.city
            )
        )
        
        print(f"✅ Response prepared")
        return response
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/forecast-by-date")
async def forecast_by_date(request: DateForecastRequest):
    """
    Get weather forecast for a specific date and predict risk level
    Enter only date and city - weather data is fetched automatically
    """
    try:
        print(f"📅 Processing forecast for {request.city} on {request.date}")
        
        # Validate city
        if request.city not in CITY_COORDINATES:
            raise HTTPException(status_code=400, detail=f"City '{request.city}' not found. Available cities: {', '.join(sorted(CITY_COORDINATES.keys()))}")
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Prediction model not loaded")
        
        # Get coordinates
        coords = CITY_COORDINATES[request.city]
        
        # Fetch weather forecast from OpenWeatherMap
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": WEATHER_API_KEY,
            "units": "metric",  # For Celsius
            "cnt": 40  # Number of timestamps (5 days * 8 per day)
        }
        
        print(f"🌐 Fetching weather data from API...")
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=502, detail=f"Failed to fetch weather data: {response.status_code}")
        
        forecast_data = response.json()
        
        # Find forecast for the requested date
        target_date = datetime.strptime(request.date, "%Y-%m-%d").date()
        day_forecasts = []
        
        for item in forecast_data["list"]:
            item_date = datetime.fromtimestamp(item["dt"]).date()
            if item_date == target_date:
                day_forecasts.append(item)
        
        if not day_forecasts:
            # Check if date is within range
            available_dates = set()
            for item in forecast_data["list"]:
                available_dates.add(datetime.fromtimestamp(item["dt"]).date())
            
            available_dates_str = [d.strftime("%Y-%m-%d") for d in sorted(available_dates)]
            
            raise HTTPException(
                status_code=404, 
                detail=f"No forecast found for date {request.date}. Available dates: {', '.join(available_dates_str[:5])}"
            )
        
        # Calculate daily averages
        temps_max = [f["main"]["temp_max"] for f in day_forecasts]
        temps_min = [f["main"]["temp_min"] for f in day_forecasts]
        temps_avg = [(f["main"]["temp_max"] + f["main"]["temp_min"]) / 2 for f in day_forecasts]
        wind_speeds = [f["wind"]["speed"] * 3.6 for f in day_forecasts]  # Convert m/s to km/h
        humidities = [f["main"]["humidity"] for f in day_forecasts]
        
        # Calculate daily averages
        weather_data = {
            "temp_max": round(max(temps_max), 1),
            "temp_min": round(min(temps_min), 1),
            "temp_avg": round(sum(temps_avg) / len(temps_avg), 1),
            "wind_speed": round(sum(wind_speeds) / len(wind_speeds), 1),
            "humidity": round(sum(humidities) / len(humidities), 1),
            "city": request.city
        }
        
        print(f"📊 Calculated weather for {request.date}: {weather_data}")
        
        # Prepare features for prediction
        features = np.array([[
            weather_data["temp_max"],
            weather_data["temp_min"],
            weather_data["temp_avg"],
            weather_data["wind_speed"],
            weather_data["humidity"]
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        pred = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        risk_code = int(pred)
        risk_level = RISK_NAMES[risk_code]
        confidence = float(max(probabilities) * 100)
        
        # Create probability dictionary
        prob_dict = {}
        for i in range(len(probabilities)):
            prob_value = float(probabilities[i] * 100)
            if prob_value > 0.1:
                prob_dict[RISK_NAMES[i]] = round(prob_value, 2)
        
        # Create response
        response = {
            "risk_level": risk_level,
            "risk_code": risk_code,
            "confidence": round(confidence, 2),
            "probabilities": prob_dict,
            "city": request.city,
            "forecast_date": request.date,
            "weather": {
                "temp_max": weather_data["temp_max"],
                "temp_min": weather_data["temp_min"],
                "temp_avg": weather_data["temp_avg"],
                "wind_speed": weather_data["wind_speed"],
                "humidity": weather_data["humidity"],
                "city": request.city
            }
        }
        
        print(f"✅ Forecast and prediction complete")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in forecast-by-date: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/governorates")
def get_governorates():
    """List of Tunisian governorates"""
    governorates = sorted([
        "Tunis", "Ariana", "Ben Arous", "Manouba", "Nabeul", "Zaghouan",
        "Bizerte", "Beja", "Jendouba", "Kef", "Siliana", "Sousse",
        "Monastir", "Mahdia", "Sfax", "Kairouan", "Kasserine", "Sidi Bouzid",
        "Gabes", "Medenine", "Tataouine", "Gafsa", "Tozeur", "Kebili"
    ])
    return {"governorates": governorates}

@app.get("/risk-info")
def get_risk_info():
    """Information about risk levels"""
    return {
        "levels": [
            {
                "name": name,
                "code": code,
                "color": RISK_COLORS[code],
                "description": RISK_DESCRIPTIONS[name]
            }
            for code, name in RISK_NAMES.items()
        ]
    }

@app.get("/debug/model-info")
def model_info():
    """Debug endpoint to check model info"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": type(model).__name__,
        "features": feature_cols,
        "classes": list(RISK_NAMES.values()),
        "n_features": len(feature_cols) if feature_cols else 0,
        "model_loaded": True
    }

@app.get("/debug/test-prediction")
def test_prediction():
    """Test endpoint with sample data"""
    if model is None:
        return {"error": "Model not loaded"}
    
    test_cases = [
        {"name": "Normal Day", "data": {"temp_max": 22, "temp_min": 15, "temp_avg": 18, "wind_speed": 15, "humidity": 65}},
        {"name": "Hot Day", "data": {"temp_max": 38, "temp_min": 24, "temp_avg": 31, "wind_speed": 20, "humidity": 40}},
        {"name": "Windy Day", "data": {"temp_max": 24, "temp_min": 18, "temp_avg": 21, "wind_speed": 55, "humidity": 70}},
    ]
    
    results = []
    for test in test_cases:
        try:
            features = np.array([[
                test["data"]["temp_max"],
                test["data"]["temp_min"],
                test["data"]["temp_avg"],
                test["data"]["wind_speed"],
                test["data"]["humidity"]
            ]])
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            results.append({
                "name": test["name"],
                "prediction": int(pred),
                "risk_level": RISK_NAMES[int(pred)],
                "confidence": float(max(proba) * 100),
                "success": True
            })
        except Exception as e:
            results.append({
                "name": test["name"],
                "error": str(e),
                "success": False
            })
    
    return {"test_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)